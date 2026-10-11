# What changed on `clean-pipeline`, and why

This branch starts from `k.karimli` and makes the project actually runnable
with one command, as the README promises.

`k.karimli` had the two hand-written models and nothing else. The data
pipeline files were empty placeholders, so `python -m src.run_all` printed
nothing and exited — there was no way to produce a single number or figure.

Everything below is the list of changes, why each was needed, and what it did
to the results.

---

## 1. Summary of the changes

| File | Before (`k.karimli`) | After | What it is |
|---|---|---|---|
| `src/run_all.py` | 0 bytes (empty) | 183 lines | **new** — runs the whole pipeline |
| `src/evaluate.py` | 32 bytes (one comment) | 580 lines | **rewritten** — scores our models |
| `src/ensemble.py` | missing | 240 lines | **new** — bagging over our tree |
| `src/data_prep.py` | 70 bytes (stub) | 706 lines | restored from `data-preprocessing` |
| `src/validate.py` | missing | 23 KB | restored from `data-preprocessing` |
| `src/__init__.py` | missing | 0 bytes | restored — makes `python -m src.*` work |
| `explore.py` | missing | 55 KB | restored from `data-preprocessing` |
| `report/data_report.tex` | missing | 25 KB | restored from `data-preprocessing` |
| `data_prep.py` (repo root) | 0 bytes | **deleted** | empty leftover, confusing duplicate |
| `src/decision_tree.py` | 13.7 KB | **unchanged** | your tree, kept exactly as it was |
| `src/svm.py` | 3.7 KB | **unchanged** | your SVM, kept exactly as it was |

**Neither model file was touched.** All the work was around them.

---

## 2. Making the pipeline exist at all

**Problem.** `src/run_all.py` was an empty file. `src/data_prep.py`,
`src/evaluate.py` were one-line comments. `src/validate.py`, `explore.py` and
`src/__init__.py` did not exist. The real versions of these were sitting
unmerged on the `data-preprocessing` branch.

**Fix.** Brought those files over and wrote a new `run_all.py` that chains the
steps together:

```
data_prep  ->  validate  ->  explore (EDA)  ->  evaluate  ->  summary
```

**Result.** `python -m src.run_all` goes from the raw 141 MB CSV to every
number and figure in **6.8 minutes**.

Reproducibility check — the rebuild produced exactly the same data as the
original preprocessing run:

| | Value |
|---|---|
| Raw rows | 100,775 |
| After removing duplicate scrapes | 63,241 |
| After keeping Baku + 3 residential categories | 57,069 |
| After domain filters | **57,003** |
| Split | train 39,902 / valid 8,550 / test 8,551 |
| Features | 82 |
| Tier threshold | 209,000 AZN |

All of `validate.py`'s integrity checks pass.

---

## 3. Putting OUR models in charge (the important one)

**Problem.** The old `evaluate.py` from `data-preprocessing` measured
**scikit-learn's** `DecisionTreeRegressor` and reported that as the result.
Our own `DecisionTree` and `PegasosSVM` were never scored at all. The README's
Golden rule says the opposite: our NumPy code is the model, scikit-learn is
only a comparison baseline.

**Fix.** Rewrote `evaluate.py` so it fits `DecisionTree` and `PegasosSVM` from
`src/`, and keeps sklearn only as a baseline to compare against.

**Result.** We can finally answer the question the project is actually about —
does our code work? It does:

| | Our code | sklearn (same settings) | Gap |
|---|---|---|---|
| Task A RMSE | 0.2287 | 0.2285 | **0.0002** |
| Task B accuracy | 0.9113 | 0.9111 | **0.0002** |

Our from-scratch implementations match a professional library to four decimal
places. This is a better result than beating it would have been — it means the
implementations are correct.

---

## 4. Making the comparison fair

**Problem.** The first time we compared our tree to sklearn, our tree looked
better. That was not real. We were running our tree with `min_samples_leaf=5`
and sklearn with its default of `1`, so we were comparing two different
settings and giving ourselves the better one.

**Fix.** The sklearn baseline is now built with the same `min_samples_leaf` as
ours.

**Result.** The fake advantage disappeared, which is the honest outcome:

| Comparison | RMSE (ours) | RMSE (sklearn) | Reading |
|---|---|---|---|
| Unfair (`min_samples_leaf` 5 vs 1) | 0.2300 | 0.2399 | "we beat sklearn by 4%" — **wrong** |
| Fair (both 5) | 0.2300 | 0.2295 | tied, sklearn ahead by 0.2% |

Do not claim we beat scikit-learn in the report. Claim we **match** it.

---

## 5. Making it finish in minutes instead of hours

**Problem.** Our tree is correct but slow. `_best_split` tries every midpoint
between every pair of neighbouring values, for every feature, at every node.
On 39,902 rows that is a very large number of checks. The old evaluation swept
depths 1 to 30 for two tasks — with our tree that would have taken about
**3 hours**.

**Fix.** Choose the depth on a fixed 6,000-row sample (fast), then train the
final model at that depth on all 39,902 rows (accurate). Added a `--fast` flag
that shrinks everything for a quick check that the pipeline works.

**Result.**

| Mode | Time | Use |
|---|---|---|
| `--fast` | ~0.4 min | check nothing is broken |
| full run | **6.8 min** | the real numbers |
| old approach | ~3 hours | not usable |

Final fits: regression 135.5s, classification 159.3s, SVM 4.8s.

---

## 6. Tuning the models properly

Before, the settings were picked by hand. Now they are searched.

### Tree depth

| Setting | Source | RMSE (valid) | R² |
|---|---|---|---|
| depth 13 | guessed by hand | 0.2300 | 0.8676 |
| **depth 16** | chosen by sweep | **0.2287** | **0.8691** |

Small but free improvement. The validation curve is nearly flat from depth 10
onward (0.2795 at 10 vs 0.2794 at 16 on the tuning sample), so this choice is
not sensitive.

### SVM regularisation (lambda)

This was the biggest single gain:

| Setting | Source | Accuracy (valid) | ROC-AUC |
|---|---|---|---|
| lambda = 0.01 | guessed by hand | 0.8896 | 0.9571 |
| **lambda = 0.001** | chosen by sweep | **0.9029** | **0.9673** |

**+1.3 points of accuracy and +1.0 of AUC**, just from searching one number.

### Classification depth — a real trade-off

The sweep chose depth 8 instead of the 15 we had been using by hand. This
makes accuracy slightly worse but ranking clearly better:

| Depth | Accuracy (valid) | ROC-AUC (valid) |
|---|---|---|
| 15 (hand-picked) | 0.9204 | 0.9481 |
| **8 (chosen)** | 0.9113 | **0.9664** |

**Why:** a shallow tree has more samples sitting in each leaf, so
`predict_proba` returns a graded score like 0.7. A deep tree has leaves that
are nearly pure, so it only ever outputs values close to 0 or 1. Accuracy does
not care, but ROC-AUC measures *ranking*, and you cannot rank well with only
two possible scores.

So we lose 0.9 points of accuracy and gain 1.8 points of AUC. If the report
wants accuracy as its headline number instead, raise `TUNE_SUBSAMPLE` in
`src/evaluate.py` and the sweep will pick a deeper tree.

---

## 7. Fixing the SVM decision threshold

**Problem.** Pegasos classifies using `score >= 0`. But Pegasos optimises
hinge loss, not accuracy, so 0 is not necessarily the best place to cut.

**Fix.** Added `tune_svm_threshold`, which picks the cut that maximises
validation accuracy.

**Result.** Honest answer: it barely matters now.

| | Accuracy |
|---|---|
| Default cut at 0 | 0.9029 |
| Tuned cut at −0.0586 | 0.9043 |

Only +0.14 points. At the old lambda = 0.01 the gap was about 3 points, so
this mattered more before the lambda sweep fixed the real problem. Keep it,
but do not make a claim about it in the report.

---

## 7b. Bagging our own trees (the biggest win)

**Problem.** A single tree has low bias but high variance — move a few rows
and the splits change. Scikit-learn's RandomForest on the *same 82 features*
scores MSE 0.03440 against our single tree's 0.05221. That gap is the model,
not the data.

**Fix.** Added `src/ensemble.py` with `BaggedTrees`: train many copies of our
own `DecisionTree` on bootstrap samples and average them. `decision_tree.py`
is not modified.

**Result.**

| Task | Metric | One tree | Bagged (30 x 6,000) | Change |
|---|---|---|---|---|
| A (valid) | MSE | 0.05232 | **0.04201** | **−19.7%** |
| A (valid) | R² | 0.8691 | **0.8949** | +0.026 |
| A (valid) | typical error | 15.9% | **14.6%** | −1.3 pp |
| A (test) | MSE | 0.05547 | **0.04734** | **−14.7%** |
| B (valid) | accuracy | 0.9113 | **0.9284** | +1.7 pp |
| B (valid) | ROC-AUC | 0.9664 | **0.9808** | +1.4 pp |
| B (test) | accuracy | 0.9159 | **0.9336** | +1.8 pp |

Cost: about 4 minutes per task.

**It also removed the trade-off from section 6.** A single tree had to choose
between accuracy (depth 15: 0.9204 acc / 0.9481 AUC) and ranking (depth 8:
0.9113 / 0.9664). Bagging beats both on both at once, because averaging 30
trees produces finely graded probabilities *and* a more accurate vote.

**Out-of-bag scores.** Each training row is scored using only the trees whose
bootstrap sample excluded it, which gives a held-out estimate using no
validation data: R² 0.8883 (regression), accuracy 0.9291 (classification).
Both land close to the real validation numbers, which is independent evidence
the ensemble is not overfitting. Worth a sentence in the report.

### Two choices we made by measuring, not guessing

**No feature subsampling.** A real random forest also picks a random subset of
features at each node. We tested it:

| Design | MSE | Needs editing `decision_tree.py`? |
|---|---|---|
| **plain bagging** | **0.04213** | no |
| per-node 50% features | 0.04290 | yes |
| per-tree 50% columns | 0.05431 | no |

Plain bagging won, so Kenan's tree stays untouched. 59 of our 82 features are
sparse location one-hots, and dropping them loses more signal than the extra
decorrelation gains.

**Members are grown deeper than the single tree.** Our first attempt passed
the swept depth (8) to the ensemble. That was wrong. The sweep answers "how
deep before *one* tree overfits", but averaging already controls variance, so
members should fit hard and let the ensemble cancel the noise — this is why
random forests grow their trees out.

| Member depth | Accuracy | ROC-AUC | OOB |
|---|---|---|---|
| 8 (swept, wrong) | 0.9145 | 0.9757 | 0.9165 |
| **16 (fixed)** | **0.9284** | **0.9808** | **0.9291** |

Costing 1.4 points of accuracy. `BAG_MAX_DEPTH` in `src/evaluate.py` now sets
this separately, with the measurement recorded beside it.

---

## 8. Final numbers

Full run, 39,902 training rows, seed 42, `fast_mode: false`.

### Task A — predicting price (natural log of price)

| Split | MSE | RMSE | MAE | R² | Typical error |
|---|---|---|---|---|---|
| Validation | 0.05232 | 0.2287 | 0.1472 | 0.8691 | **15.9%** |
| Test | 0.05547 | 0.2355 | 0.1469 | 0.8665 | 15.8% |

"Typical error" is `exp(MAE) - 1`. Because the target is log price, an error
is a ratio, so this is the plain-language version: the prediction is usually
within about 16% of the real price. At the median listing of 209,000 AZN that
is roughly ±33,000 AZN.

**Do not quote MSE on its own.** It is in squared log units and means nothing
by itself. The same model measured on raw manat would score about 8.5 billion,
which is not worse — just a different ruler. Use R² and the percentage.

Bagging 30 of our trees improves this to MSE 0.04201 / R² 0.8949 on
validation (0.04734 / 0.8861 on test) — see section 7b.

### Task B — predicting price tier (premium vs standard)

| Model | Split | Accuracy | ROC-AUC |
|---|---|---|---|
| our `DecisionTree` | validation | 0.9113 | 0.9664 |
| our `DecisionTree` | test | 0.9159 | 0.9695 |
| our `PegasosSVM` | validation | 0.9029 | 0.9673 |
| our `PegasosSVM` | test | 0.9084 | 0.9698 |
| **our `BaggedTrees`** | validation | **0.9284** | **0.9808** |
| **our `BaggedTrees`** | test | **0.9336** | **0.9827** |

SVM details: 11,094 support vectors, margin width 0.3023, lambda 0.001.

The two models are close. The tree is slightly more accurate, the SVM ranks
slightly better. The SVM trains in 4.8 seconds versus the tree's 159 seconds.

### Baselines (validation, same settings as ours)

| Model | RMSE | Accuracy |
|---|---|---|
| **ours** | **0.2287** | **0.9113** |
| sklearn tree | 0.2285 | 0.9111 |
| Ridge / LinearSVC | 0.2407 | 0.9076 |
| Dummy | 0.6322 | 0.4936 |

Note that Ridge — a plain linear model — gets RMSE 0.2407 against our 0.2287.
That is only 5% worse. Worth mentioning in the report: most of the signal in
this dataset is close to linear once `log_area_m2` is used as a feature, which
is why the tree's advantage is modest.

---

## 9. State of the data

Checked the processed arrays directly. Almost everything is clean:

| Check | Result |
|---|---|
| Missing or infinite values | 0 |
| Shapes line up with targets | yes |
| Constant or duplicate columns | none |
| Feature scaling | mean ≈ 0, std ≤ 1 |
| Tier balance | ~50% in every split |
| **Price leakage** | **none** — highest correlation is `log_area_m2` at 0.73 |

The leakage check is the important one. `unit_price` and `total_price` were
dropped correctly. If they had leaked in, R² would be about 0.99 and the whole
model would be meaningless.

**One real defect:** 44 rows appear in both train and validation, and 40 in
both train and test (about 0.5%). These are rows with identical features —
most likely genuinely different flats that happen to match on every column,
rather than missed duplicates. Too small to change the numbers, but it should
be mentioned in the report since we cannot prove they are distinct properties.

---

## 10. Known limitations

1. **Depth is tuned on 6,000 rows but used on 39,902.** The best depth usually
   grows with the amount of data, so our chosen depths are probably slightly
   too shallow. This is a deliberate trade for runtime, and it is why
   classification sits at depth 8.
2. **Bagging members are not separately tuned.** `BAG_MAX_DEPTH` is a fixed
   16, chosen by measurement rather than by its own sweep. A proper sweep over
   member depth and tree count would likely gain a little more.
3. **Our tree is slow.** `_best_split` rechecks the full column for every
   candidate threshold. Sorting each feature once and sweeping with running
   totals would make it far faster. Not required for the assignment, but it is
   the honest explanation for the 159-second fit.
4. **`min_impurity_decrease` is not weighted.** scikit-learn scales it by
   `n_node / n_total`; ours compares the raw gain. Same idea, so the same
   number will not prune identically in both. Not a bug.
5. **`scipy` is not pinned** in `requirements.txt` (installed: 1.18.1). Every
   other dependency matches its pin.

---

## 11. How to run it

```bash
source .venv/bin/activate

python -m src.run_all                        # everything, ~14 min
python -m src.run_all --fast                 # quick check, ~30 s, NOT real numbers
python -m src.run_all --skip-prep            # reuse data/processed
python -m src.run_all --skip-prep --skip-eda # only re-score the models
python -m src.run_all --no-ensemble          # skip bagging (saves ~8 min)
```

Then read:

```bash
cat reports/summary.md             # one page
cat reports/figures/metrics.md     # full tables
```

Outputs:

- `reports/figures/fig1-5_*.png` — bias/variance, predictions, ROC and
  confusion matrix, model comparison, error analysis
- `reports/eda/` — EDA report, validation report, cleanliness plots
- `report/figures/` — the four figures `report/data_report.tex` includes
- `reports/figures/metrics.json` — every number, machine-readable

If `metrics.json` has `"fast_mode": true`, you are looking at smoke-test
numbers, not real ones.

# PegaSold presentation

`slides.pdf` is the current 15-slide Madrid Beamer deck. It uses the supplied sans-serif style, white content pages, deep Colombian blue with yellow/red accents, and a full-width tricolor/footer panel. Team names appear on the title slide only. The sequence follows the report: task/data, cleaning, features and split, from-scratch models, validation, held-out metrics, limitations and reproducibility.

Regenerate the report and slides from the latest experiment outputs with `python -m src.run_all`. The run also rebuilds `site-data.js` for the project website.

## Balanced four-speaker plan

The allocation below aims for similar effort, not identical slide counts. Each speaker owns one technical workstream and its results. Target effort is 25% each; the plan is not a record of completed individual contributions. Keep speaking time near 3–4 minutes per person.

| Speaker / member | Owned project work | Slides and supporting discussion | Effort target |
|---|---|---|---:|
| Speaker 1 — Emiliya Abdulhalimova | Raw EDA, cleaning, leakage review, feature design and split framing | 2–5 | 25% |
| Speaker 2 — Aygun Asvarova | NumPy decision tree for both tasks, split criteria, tuning, feature importance and AZN error interpretation | 6–8 and tree interpretation in 13 | 25% |
| Speaker 3 — Murad Bagirzade | Pegasos hinge-loss optimizer, margin/support vectors and random Fourier features | 9–10 and SVM comparison in 12 | 25% |
| Speaker 4 — Kanan Karimli | Evaluation, matched baselines, CV, error analysis, tests and reproducibility | 11–13 and report/handoff in 14–15 | 25% |

Slides 14–15 are shared for conclusions and handoff. Adjust the spoken handoffs so technical explanations and evidence remain balanced. Record actual work and review evidence separately in `contribution_report.md`.

## Slide order

1. Title and PegaSold team
2. Project question and two tasks
3. Cleaning and leakage control
4. Exploration and feature design
5. Split, targets and validation protocol
6. Decision tree from scratch
7. Tree results and generalization
8. Interpreting the tree's AZN price error
9. Linear Pegasos SVM
10. Nonlinear SVM and decision margin
11. Fair comparison and uncertainty
12. Held-out classification comparison
13. Validation sweeps and model interpretation
14. Conclusions and limitations
15. Reproducibility and four-part handoff

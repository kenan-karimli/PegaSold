"""ONE command reproduces every number/figure: prep -> validate -> EDA -> evaluate."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sh(*cmd: str) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Reproduce every number and figure.")
    p.add_argument("--data", default="data/bina_az_sale.csv")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--eda-out", default="reports/eda")
    p.add_argument("--fig-out", default="reports/figures")
    a = p.parse_args()

    # 1. clean + split + model-ready arrays (seeds fixed in src/data_prep.py)
    from data_prep import run as prep_run

    prep_run(a.data, a.processed)

    # 2. validation + cleanliness plots + validation_report.md
    sh(sys.executable, "-m", "src.validate", "--dir", a.processed, "--out", a.eda_out)

    # 3. EDA report + figures (read-only, never modifies data/)
    sh(sys.executable, "explore.py", "--data", a.data, "--out", a.eda_out)

    # 4. model diagnostics + metrics.json/md (sklearn baselines until DT/SVM land)
    sh(sys.executable, "-m", "src.evaluate", "--dir", a.processed, "--out", a.fig_out)

    # 5. one-page summary pointing at the artifacts
    prep = json.loads((Path(a.processed) / "preprocessing_report.json").read_text())
    metrics = json.loads((Path(a.fig_out) / "metrics.json").read_text())
    s = metrics["task_a_val"]
    t = metrics["task_b_val"]
    (ROOT / "reports" / "summary.md").write_text(
        "# Run summary (reproduced by `python -m src.run_all`)\n\n"
        f"- Cleaned rows: {prep['final_rows']:,} "
        f"(raw {prep['raw_rows']:,}, dedup -{prep['rows_removed_by_item_deduplication']:,}, "
        f"scope -{prep['scope']['rows_removed_by_scope']:,}, "
        f"domain -{prep['domain_filters']['total_removed_by_domain_filters']:,})\n"
        f"- Split: {prep['split_rows']}\n"
        f"- Features: {prep['n_processed_features']} "
        f"(numeric={len(prep['numeric_features'])})\n"
        f"- Tier threshold: {prep['tier_threshold_AZN']:,.0f} AZN\n"
        f"- Task A valid: RMSE={s['rmse']:.4f} MAE={s['mae']:.4f} R²={s['r2']:.4f}\n"
        f"- Task B valid: Acc={t['acc']:.4f} ROC-AUC={t['roc_auc']:.4f}\n"
        f"- synthetic={metrics['synthetic']}\n\n"
        "Artifacts: `reports/eda/eda_report.txt`, "
        "`reports/eda/validation_report.md`, `reports/eda/figures/val_*.png`, "
        "`reports/figures/fig1-5_*.png`, `reports/figures/metrics.json/md`.\n",
        encoding="utf-8",
    )
    print("\nAll numbers/figures reproduced. See reports/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

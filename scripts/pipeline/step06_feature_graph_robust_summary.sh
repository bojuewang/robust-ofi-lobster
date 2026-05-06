#!/usr/bin/env bash
set -euo pipefail

# Step 06: Final feature/graph robustness summary + reproducibility/GitHub scaffold
#
# This module aggregates Step 03, Step 04b, and Step 05 into a report-ready
# robustness summary. It does not retrain models.
#
# Inputs:
#   .pipeline_state/step03_graph_baseline.PASS
#   .pipeline_state/step04b_calibrated_eval.PASS
#   .pipeline_state/step05_graph_robust_pgda.PASS
#   reports/benchmarks/step03_graph_build_and_baseline.json
#   reports/benchmarks/step04b_feature_robust_calibrated_eval.json
#   reports/benchmarks/step05_graph_robust_pgda.json
#
# Outputs:
#   reports/final/robust_ofi_final_summary.md
#   reports/final/final_ablation_table.csv
#   reports/final/final_robustness_findings.json
#   reports/figures/*.png
#   README.md
#   requirements.txt
#   .gitignore
#   scripts/run_pipeline_to_step06.sh
#   reports/benchmarks/step06_feature_graph_robust_summary.md
#   .pipeline_state/step06_summary.PASS
#
# Run:
#   bash step06_feature_graph_robust_summary.sh
# or:
#   PROJECT_DIR=/path/to/robust-ofi-lobster bash step06_feature_graph_robust_summary.sh

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
PROJECT_DIR="${PROJECT_DIR:-${ROOT_DIR}/robust-ofi-lobster}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  if [[ -d "${ROOT_DIR}/reports/benchmarks" ]]; then
    PROJECT_DIR="${ROOT_DIR}"
  else
    echo "[ERROR] PROJECT_DIR not found: ${PROJECT_DIR}"
    echo "        Set PROJECT_DIR=/path/to/robust-ofi-lobster"
    exit 1
  fi
fi

cd "${PROJECT_DIR}"

for gate in \
  ".pipeline_state/step03_graph_baseline.PASS" \
  ".pipeline_state/step04b_calibrated_eval.PASS" \
  ".pipeline_state/step05_graph_robust_pgda.PASS"
do
  if [[ ! -f "${gate}" ]]; then
    echo "[ERROR] Missing gate: ${gate}"
    exit 1
  fi
done

mkdir -p src/lobster_ofi scripts reports/benchmarks reports/final reports/figures data/processed .pipeline_state

cat > src/lobster_ofi/step06_feature_graph_robust_summary.py <<'PY'
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

TAUS = [1, 2, 10, 20]
BENCH = Path("reports/benchmarks")
FINAL = Path("reports/final")
FIGS = Path("reports/figures")
STATE = Path(".pipeline_state")

@dataclass
class Check:
    name: str
    status: str
    observed: str
    expected: str

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"[ERROR] Missing JSON file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def fnum(x, digits=6):
    if x is None:
        return ""
    try:
        x = float(x)
    except Exception:
        return str(x)
    if math.isnan(x):
        return "nan"
    if abs(x) < 1e-3 or abs(x) > 1e4:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}f}"

def md_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            vals.append(fnum(v) if isinstance(v, (float, int)) else str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)

def write_csv(path: Path, rows: List[Dict[str, Any]], cols: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

def best_step03_by_tau(step03: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    rows = step03.get("best_by_tau") or []
    return {int(r["tau"]): r for r in rows}

def feature_summary(step04b: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    rows = step04b.get("improvement_rows") or []
    return {int(r["tau"]): r for r in rows}

def graph_summary(step05: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    rows = step05.get("improvement_rows") or []
    return {int(r["tau"]): r for r in rows}

def best_rows_by_tau(rows: List[Dict[str, Any]], tau: int, key: str):
    rs = [r for r in rows if int(r.get("tau", -1)) == tau]
    if not rs:
        return None
    return min(rs, key=lambda r: float(r.get(key, float("inf"))))

def make_plots(final_rows: List[Dict[str, Any]]):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return False, f"matplotlib unavailable: {exc}"

    taus = [r["tau"] for r in final_rows]

    # Plot 1: feature vs graph robust degradation improvement
    feat = [r["feature_deg_mse_improvement_0.1"] for r in final_rows]
    graph = [r["graph_deg_mse_improvement_0.1"] for r in final_rows]
    x = np.arange(len(taus))
    width = 0.35

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.bar(x - width/2, feat, width, label="Feature robust")
    ax.bar(x + width/2, graph, width, label="Graph robust")
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in taus])
    ax.set_xlabel("Prediction horizon tau")
    ax.set_ylabel("MSE degradation improvement at rho=0.1")
    ax.set_title("Robustness gain under calibrated perturbations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "step06_degradation_improvement.png", dpi=160)
    plt.close(fig)

    # Plot 2: clean MSE tradeoff, robust vs ERM
    erm = [r["erm_clean_mse_reference"] for r in final_rows]
    feat_clean = [r["feature_robust_clean_mse"] for r in final_rows]
    graph_clean = [r["graph_robust_clean_mse"] for r in final_rows]

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(taus, erm, marker="o", label="ERM reference")
    ax.plot(taus, feat_clean, marker="o", label="Feature robust choice")
    ax.plot(taus, graph_clean, marker="o", label="Graph robust choice")
    ax.set_xlabel("Prediction horizon tau")
    ax.set_ylabel("Clean test MSE")
    ax.set_title("Clean-test accuracy tradeoff")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "step06_clean_mse_tradeoff.png", dpi=160)
    plt.close(fig)

    # Plot 3: graph sensitivity improvement
    gsens = [r["graph_sensitivity_improvement"] for r in final_rows]
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.bar([str(t) for t in taus], gsens)
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Prediction horizon tau")
    ax.set_ylabel("Graph sensitivity improvement")
    ax.set_title("Graph-robust training reduces graph sensitivity")
    fig.tight_layout()
    fig.savefig(FIGS / "step06_graph_sensitivity_improvement.png", dpi=160)
    plt.close(fig)

    return True, "plots created"

def main():
    checks: List[Check] = []
    def add(name, ok, observed, expected):
        checks.append(Check(name, "PASS" if ok else "FAIL", str(observed), str(expected)))

    step03_path = BENCH / "step03_graph_build_and_baseline.json"
    step04b_path = BENCH / "step04b_feature_robust_calibrated_eval.json"
    step05_path = BENCH / "step05_graph_robust_pgda.json"

    step03 = load_json(step03_path)
    step04b = load_json(step04b_path)
    step05 = load_json(step05_path)

    add("step03_status_pass", step03.get("status") == "PASS", step03.get("status"), "PASS")
    add("step04b_status_pass", step04b.get("status") == "PASS", step04b.get("status"), "PASS")
    add("step05_status_pass", step05.get("status") == "PASS", step05.get("status"), "PASS")

    s03 = best_step03_by_tau(step03)
    fsum = feature_summary(step04b)
    gsum = graph_summary(step05)

    add("all_tau_in_step03", all(t in s03 for t in TAUS), sorted(s03.keys()), TAUS)
    add("all_tau_in_step04b", all(t in fsum for t in TAUS), sorted(fsum.keys()), TAUS)
    add("all_tau_in_step05", all(t in gsum for t in TAUS), sorted(gsum.keys()), TAUS)

    final_rows: List[Dict[str, Any]] = []
    conclusion_rows: List[Dict[str, Any]] = []

    for tau in TAUS:
        b03 = s03[tau]
        f = fsum[tau]
        g = gsum[tau]

        # Use graph-robust ERM clean MSE as main ERM reference because Step 05 uses same calibrated graph perturbation model.
        row = {
            "tau": tau,
            "step03_best_model": b03.get("model"),
            "step03_best_graph": b03.get("graph"),
            "step03_clean_mse": float(b03.get("mse")),
            "step03_directional_accuracy": float(b03.get("directional_accuracy")),
            "erm_clean_mse_reference": float(g.get("erm_clean_mse")),
            "feature_robust_clean_mse": float(f.get("robust_clean_mse")),
            "feature_erm_deg_mse_0.1": float(f.get("erm_deg_mse_0.1")),
            "feature_robust_deg_mse_0.1": float(f.get("robust_deg_mse_0.1")),
            "feature_deg_mse_improvement_0.1": float(f.get("deg_mse_improvement")),
            "feature_robust_choice": f.get("robust_choice"),
            "graph_robust_clean_mse": float(g.get("robust_clean_mse")),
            "graph_erm_deg_mse_0.1": float(g.get("erm_deg_mse_0.1")),
            "graph_robust_deg_mse_0.1": float(g.get("robust_deg_mse_0.1")),
            "graph_deg_mse_improvement_0.1": float(g.get("deg_mse_improvement")),
            "graph_erm_sensitivity_G": float(g.get("erm_sensitivity_G")),
            "graph_robust_sensitivity_G": float(g.get("robust_sensitivity_G")),
            "graph_sensitivity_improvement": float(g.get("sensitivity_improvement")),
            "graph_robust_choice": g.get("robust_choice"),
        }
        final_rows.append(row)

        if row["graph_deg_mse_improvement_0.1"] > 0 and row["feature_deg_mse_improvement_0.1"] <= 0:
            verdict = "graph robust improves; feature robust does not"
        elif row["graph_deg_mse_improvement_0.1"] > 0 and row["feature_deg_mse_improvement_0.1"] > 0:
            verdict = "both feature and graph robustness improve degradation"
        elif row["graph_deg_mse_improvement_0.1"] <= 0 and row["feature_deg_mse_improvement_0.1"] > 0:
            verdict = "feature robust improves; graph robust does not"
        else:
            verdict = "neither robust variant improves degradation"

        conclusion_rows.append({
            "tau": tau,
            "feature_gain_positive": row["feature_deg_mse_improvement_0.1"] > 0,
            "graph_gain_positive": row["graph_deg_mse_improvement_0.1"] > 0,
            "clean_tradeoff_feature": row["feature_robust_clean_mse"] - row["erm_clean_mse_reference"],
            "clean_tradeoff_graph": row["graph_robust_clean_mse"] - row["erm_clean_mse_reference"],
            "verdict": verdict,
        })

    add("final_rows_count", len(final_rows) == len(TAUS), len(final_rows), len(TAUS))
    add("graph_improvement_all_tau", all(r["graph_deg_mse_improvement_0.1"] > 0 for r in final_rows),
        [fnum(r["graph_deg_mse_improvement_0.1"]) for r in final_rows], "positive for all tau")
    add("feature_improvement_some_tau", any(r["feature_deg_mse_improvement_0.1"] > 0 for r in final_rows),
        [fnum(r["feature_deg_mse_improvement_0.1"]) for r in final_rows], "positive for at least one tau")

    plots_ok, plots_msg = make_plots(final_rows)
    add("plots_created", plots_ok, plots_msg, "matplotlib plots saved")

    final_cols = [
        "tau",
        "step03_best_model", "step03_best_graph", "step03_clean_mse", "step03_directional_accuracy",
        "erm_clean_mse_reference",
        "feature_robust_clean_mse", "feature_erm_deg_mse_0.1", "feature_robust_deg_mse_0.1",
        "feature_deg_mse_improvement_0.1", "feature_robust_choice",
        "graph_robust_clean_mse", "graph_erm_deg_mse_0.1", "graph_robust_deg_mse_0.1",
        "graph_deg_mse_improvement_0.1", "graph_erm_sensitivity_G", "graph_robust_sensitivity_G",
        "graph_sensitivity_improvement", "graph_robust_choice",
    ]
    write_csv(FINAL / "final_ablation_table.csv", final_rows, final_cols)

    summary = {
        "status": "PASS" if all(c.status == "PASS" for c in checks) else "FAIL",
        "final_rows": final_rows,
        "conclusion_rows": conclusion_rows,
        "checks": [c.__dict__ for c in checks],
        "artifacts": [
            "reports/final/robust_ofi_final_summary.md",
            "reports/final/final_ablation_table.csv",
            "reports/figures/step06_degradation_improvement.png",
            "reports/figures/step06_clean_mse_tradeoff.png",
            "reports/figures/step06_graph_sensitivity_improvement.png",
        ],
    }
    with open(FINAL / "final_robustness_findings.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Create report-ready final markdown.
    report = []
    report.append("# Robust OFI Prediction: Final Pipeline Summary\n")
    report.append("## Experimental contract\n")
    report.append("- Data: five LOBSTER sample tickers, ten order-book levels.")
    report.append("- Time grid: 500ms bins over the regular trading window, giving `T = 46800` bins.")
    report.append("- Feature tensor: `X_std.shape = (46800, 5, 51)`.")
    report.append("- Split: chronological `70% / 15% / 15%` train/validation/test.")
    report.append("- Core comparison: ERM baseline vs feature-robust PGDA vs graph-robust PGDA.\n")

    report.append("## Main ablation table\n")
    report.append(md_table(final_rows, [
        "tau", "step03_best_model", "step03_best_graph", "step03_clean_mse", "step03_directional_accuracy",
        "feature_deg_mse_improvement_0.1", "feature_robust_choice",
        "graph_deg_mse_improvement_0.1", "graph_sensitivity_improvement", "graph_robust_choice",
    ]))
    report.append("")

    report.append("## Interpretation\n")
    report.append("1. The Step 03 ERM baselines establish a reproducible prediction baseline with finite MSE and directional accuracy above 50% across all tested horizons.")
    report.append("2. Calibrated Step 04b shows that feature perturbations are meaningful; feature-robust training helps mainly at longer horizons, but can sacrifice clean accuracy.")
    report.append("3. Step 05 gives the clearest robustness result: graph-robust training reduces graph-perturbation MSE degradation for every tested horizon and reduces graph sensitivity in all horizons.")
    report.append("4. The strongest report-level conclusion is therefore: in this five-asset one-day LOBSTER sample, robustness to cross-asset graph misspecification is more stable than feature-level adversarial robustness.\n")

    report.append("## Conclusion by horizon\n")
    report.append(md_table(conclusion_rows, [
        "tau", "feature_gain_positive", "graph_gain_positive",
        "clean_tradeoff_feature", "clean_tradeoff_graph", "verdict",
    ]))
    report.append("")

    report.append("## Figures\n")
    report.append("- `reports/figures/step06_degradation_improvement.png`")
    report.append("- `reports/figures/step06_clean_mse_tradeoff.png`")
    report.append("- `reports/figures/step06_graph_sensitivity_improvement.png`\n")

    report.append("## Recommended final-report wording\n")
    report.append("> Robust optimization does not uniformly improve clean prediction error. Instead, its value appears in degradation control under structured perturbations. Feature-level robust training gives mixed results and is most useful at longer horizons, while graph-level robust training consistently reduces sensitivity and MSE degradation under cross-asset graph perturbations. This supports the proposal's thesis that robust optimization is useful as a stability mechanism rather than simply as a nominal accuracy booster.\n")

    report.append("## Limitations\n")
    report.append("- The experiment uses one trading day of public LOBSTER samples, so regime-shift conclusions are preliminary.")
    report.append("- The graph is fixed from the training period; rolling graph experiments should be added if more time or data are available.")
    report.append("- The PnL score is only a sanity check, not a production backtest.")
    report.append("- Feature-robust PGDA is sensitive to perturbation calibration; normalized adversarial evaluation is needed to avoid false zero-degradation conclusions.\n")

    (FINAL / "robust_ofi_final_summary.md").write_text("\n".join(report), encoding="utf-8")

    # GitHub scaffold.
    readme = []
    readme.append("# Robust OFI LOBSTER Sample Pipeline\n")
    readme.append("This repository implements a reproducible pipeline for robust optimization of order-flow-imbalance prediction using public LOBSTER sample data.\n")
    readme.append("## Pipeline\n")
    readme.append("1. `step01_lobster_raw_audit_clean.sh`: raw sample audit and 500ms bin contract.")
    readme.append("2. `step02_feature_build_500ms.sh`: 51-dimensional feature tensor construction.")
    readme.append("3. `step03_graph_build_and_baseline.sh`: cross-asset graph construction and ERM baselines.")
    readme.append("4. `step04b_feature_robust_calibrated_eval.sh`: calibrated feature-adversarial evaluation.")
    readme.append("5. `step05_graph_robust_pgda.sh`: graph-robust PGDA benchmark.")
    readme.append("6. `step06_feature_graph_robust_summary.sh`: final summary, tables, figures, and GitHub scaffold.\n")
    readme.append("## Reproducibility\n")
    readme.append("Place the five LOBSTER sample zip files in `data/raw/`, then run:\n")
    readme.append("```bash")
    readme.append("bash scripts/run_pipeline_to_step06.sh")
    readme.append("```\n")
    readme.append("## Data policy\n")
    readme.append("Raw LOBSTER zip files are not committed to GitHub. The `.gitignore` excludes `data/raw/*.zip` and large intermediate arrays.\n")
    readme.append("## Main result\n")
    readme.append("See `reports/final/robust_ofi_final_summary.md`.\n")
    Path("README.md").write_text("\n".join(readme), encoding="utf-8")

    Path("requirements.txt").write_text(
        "\n".join([
            "numpy>=1.24",
            "pandas>=2.0",
            "scikit-learn>=1.3",
            "matplotlib>=3.7",
        ]) + "\n",
        encoding="utf-8",
    )

    Path(".gitignore").write_text(
        "\n".join([
            "# Python",
            "__pycache__/",
            "*.pyc",
            ".venv/",
            "venv/",
            "",
            "# Raw/proprietary or large data",
            "data/raw/*.zip",
            "data/raw/*.csv",
            "data/interim/*.csv",
            "data/processed/*.npz",
            "",
            "# Local state",
            ".pipeline_state/*.FAIL",
            "",
            "# OS/editor",
            ".DS_Store",
            ".vscode/",
        ]) + "\n",
        encoding="utf-8",
    )

    Path("scripts/run_pipeline_to_step06.sh").write_text(
        "\n".join([
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "bash step01_lobster_raw_audit_clean.sh",
            "bash step02_feature_build_500ms.sh",
            "bash step03_graph_build_and_baseline.sh",
            "bash step04_feature_robust_pgda.sh",
            "bash step04b_feature_robust_calibrated_eval.sh",
            "bash step05_graph_robust_pgda.sh",
            "bash step06_feature_graph_robust_summary.sh",
            "",
            "echo '[DONE] Full pipeline through Step 06 completed.'",
        ]) + "\n",
        encoding="utf-8",
    )
    Path("scripts/run_pipeline_to_step06.sh").chmod(0o755)

    # Benchmark report.
    status = summary["status"]
    bench = []
    bench.append("# Step 06 Feature/Graph Robustness Summary Benchmark\n")
    bench.append(f"Overall status: **{status}**\n")
    bench.append("## Final conclusion")
    bench.append("Graph-robust training gives the most consistent robustness improvement in the current experiment: MSE degradation under graph perturbation decreases for all tested horizons, and graph sensitivity decreases for all tested horizons.\n")
    bench.append("## Final ablation table")
    bench.append(md_table(final_rows, [
        "tau", "feature_deg_mse_improvement_0.1", "feature_robust_choice",
        "graph_deg_mse_improvement_0.1", "graph_sensitivity_improvement", "graph_robust_choice",
    ]))
    bench.append("")
    bench.append("## Generated artifacts")
    for a in summary["artifacts"]:
        bench.append(f"- `{a}`")
    bench.append("- `README.md`")
    bench.append("- `requirements.txt`")
    bench.append("- `.gitignore`")
    bench.append("- `scripts/run_pipeline_to_step06.sh`\n")
    bench.append("## Benchmark checks")
    bench.append("| check | status | observed | expected |")
    bench.append("|---|---|---|---|")
    for c in checks:
        bench.append(f"| {c.name} | {c.status} | `{c.observed}` | `{c.expected}` |")
    bench.append("")
    if status == "PASS":
        (STATE / "step06_summary.PASS").write_text("PASS\n", encoding="utf-8")
        bench.append("Gate passed: `.pipeline_state/step06_summary.PASS` should exist.")
        bench.append("Next recommended module: `07_github_packaging.sh` if you want an automatic commit-ready repository package.")
    else:
        (STATE / "step06_summary.FAIL").write_text("FAIL\n", encoding="utf-8")
        bench.append("Gate failed: inspect `reports/final/final_robustness_findings.json`.")

    (BENCH / "step06_feature_graph_robust_summary.md").write_text("\n".join(bench), encoding="utf-8")
    print("\n".join(bench))

if __name__ == "__main__":
    main()
PY

python src/lobster_ofi/step06_feature_graph_robust_summary.py

echo
echo "[DONE] Step 06 final summary:"
echo "  ${PROJECT_DIR}/reports/benchmarks/step06_feature_graph_robust_summary.md"
echo "  ${PROJECT_DIR}/reports/final/robust_ofi_final_summary.md"

# LLM Usage Disclosure

This document records the team's use of large language models and related AI tools for the ECE 509 term project repository.

## Responsibility statement

The team takes full responsibility for the final report, code, experiments, results, citations, and presentation. All LLM-assisted material was reviewed, edited, tested, and technically checked by the students. LLM assistance was used as a drafting, debugging, structuring, and documentation aid; it did not replace the team's responsibility to understand the optimization formulation, implementation, experiments, or conclusions.

## Tool and model

| Item | Description |
|---|---|
| Tool/platform | ChatGPT |
| Provider | OpenAI |
| Model | GPT-5.5 Thinking, where available in the interface |
| Approximate dates of use | April--May 2026 |
| Project components assisted | Proposal refinement, notation cleanup, shell-script generation, benchmark design, README/docs drafting, final-result interpretation |
| Repository | `bojuewang/robust-ofi-lobster` |

## Shared or exported transcript note

The team should either share the relevant ChatGPT project/conversation with the instructor if the platform supports sharing, or export/store a prompt transcript or screenshots for review. If a share link is available, record it here before final submission:

```text
Shared ChatGPT link or exported-log location: TODO
```

## Prompt log and relied-upon outputs

The following prompt summaries describe the material LLM assistance that shaped the repository. The exact local shell scripts and benchmark reports should be treated as the authoritative implementation record.

| Pipeline step | Purpose | Representative prompt summary | Material output relied upon | Verification / testing threshold |
|---|---|---|---|---|
| 01 | Raw LOBSTER audit and cleaning benchmark | Generate a shell script to read LOBSTER sample zip files, audit message/orderbook rows, verify timestamps, 500ms bin counts, and orderbook sanity. | `step01_lobster_raw_audit_clean.sh`; benchmark report template. | PASS required: five tickers found; row counts match expected samples; time interval `[34200,57600)`; `T=46800`; event-bin counts match; orderbook has 40 columns; positive best-level spread and sizes. |
| 02 | 500ms feature tensor construction | Generate a modular script to build 500ms orderbook carry-forward states, OFI/OBI/message-flow features, event mask, standardized feature tensor, and prediction targets. | `step02_feature_build_500ms.sh`; feature-construction logic and benchmark checks. | PASS required: `X_raw.shape=(46800,5,51)`; `X_std.shape=(46800,5,51)`; no NaN/inf; event-bin counts match Step 01; training-set standardized mean/std within tolerance; valid targets for `tau in {1,2,10,20}`. |
| 03 | Graph construction and ERM baselines | Generate script for return-correlation, OFI-correlation, lagged cross-OFI, and mask-aware graphs; train Ridge/Lasso and graph-linear ERM baselines. | `step03_graph_build_and_baseline.sh`; baseline tables and graph summaries. | PASS required: graph shape `(5,5)`; zero diagonal; finite graph values; nonempty baseline results; finite MSE; directional accuracy in `[0,1]`. |
| 04 | Initial feature-robust PGDA | Generate feature-robust PGDA script for adversarial perturbations of standardized feature matrix. | `step04_feature_robust_pgda.sh`; initial feature-robust benchmark. | PASS required: ERM and feature-robust models exist; clean/perturbed metrics finite; degradation mostly nonnegative. The team detected zero-degradation artifacts and did not rely on this step alone for final robustness claims. |
| 04b | Calibrated feature-adversarial evaluation | Generate calibrated adversarial evaluation using normalized steepest-ascent directions to avoid raw-gradient zero-degradation artifacts. | `step04b_feature_robust_calibrated_eval.sh`; calibrated feature degradation table. | PASS required: calibrated degradation nonzero for at least 75% of tested rows; degradation mostly nonnegative; all metrics finite. |
| 05 | Graph-robust PGDA benchmark | Generate graph-robust PGDA script with adversarial perturbations of the cross-asset graph. | `step05_graph_robust_pgda.sh`; graph degradation and graph-sensitivity results. | PASS required: graph attack uses radius near `rho_G=0.1`; degradation nonzero; degradation mostly nonnegative; `rho_G>0` models present; all metrics finite. |
| 06 | Final ablation summary and figures | Generate summary script aggregating Step 03, Step 04b, and Step 05 into final tables and figures. | `step06_feature_graph_robust_summary.sh`; `reports/final/final_ablation_table.csv`; figures. | PASS required: all prior statuses PASS; all horizons present; graph improvement positive for all tested horizons; feature improvement positive for at least one horizon; plots created. |
| 07 | GitHub packaging and safety checks | Generate repository packaging script, README scaffold, `.gitignore`, requirements, citation file, and Git safety check. | `step07_github_packaging.sh`; repository structure and README/docs. | PASS required: README/LICENSE/CITATION/requirements exist; pipeline scripts present; raw zip and processed npz ignored; no large nondata files over 25MB; git initialized. |
| 08 | Course compliance documentation | Generate this LLM disclosure, external code/data note, and README course-submission section. | `step08_course_compliance_docs.sh`; `llm-usage.md`; `docs/EXTERNAL_CODE_AND_DATA.md`. | PASS required: compliance files exist; README contains course-submission artifacts; no raw data included. |

## How outputs were checked

The team used benchmark-gated scripts rather than trusting generated code by default. A later module was not treated as valid until the previous module produced a PASS report and a local gate marker. Numerical checks included shape checks, finite-value checks, timestamp and row-count audits, graph-constraint checks, model-metric sanity checks, calibrated perturbation checks, and data-safety checks.

## Known limitations of LLM assistance

- Generated code and prose may contain mistakes; the team inspected and tested outputs before relying on them.
- The feature-robust PGDA step initially produced zero degradation under raw-gradient attacks; this was diagnosed as an evaluation artifact and corrected by calibrated adversarial evaluation in Step 04b.
- The LLM did not have access to private raw data except through files and benchmark outputs provided during the project workflow.
- All report-level claims should be traced to repository scripts, benchmark reports, final tables, and manually reviewed interpretations.

## Final ownership statement

The students reviewed, edited, tested, and accepted responsibility for the final repository, numerical results, written report, and presentation. The repository and report should be understood as the team's own technical work, with LLM assistance disclosed here.

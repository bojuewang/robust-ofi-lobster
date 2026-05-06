from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

EPS = 1e-12
T_EXPECTED = 46800
N_EXPECTED = 5
D_EXPECTED = 51
TRAIN_END = 32760
VAL_END = 39780
TAUS = [1, 2, 10, 20]
TICKERS = ["AMZN", "AAPL", "GOOG", "INTC", "MSFT"]
DEFAULT_GRAPH = "return_abs"

# Feature-robust PGDA grid.
RHO_X_GRID = [0.0, 0.05, 0.10, 0.20]
R_ADV_GRID = [1, 3]
EPOCHS = 8
BATCH_SIZE = 512
LR = 0.03
WEIGHT_DECAY = 1e-4
ETA_U_FACTOR = 1.25
SEED = 7


@dataclass
class Check:
    name: str
    status: str
    observed: str
    expected: str


def find_key(keys: List[str], candidates: List[str], contains_all: Optional[List[str]] = None) -> Optional[str]:
    lower = {k.lower(): k for k in keys}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if contains_all:
        for k in keys:
            kl = k.lower()
            if all(s.lower() in kl for s in contains_all):
                return k
    return None


def load_step02(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    keys = list(z.keys())

    xstd_key = find_key(keys, ["X_std", "X_tilde", "X_standardized", "features_std"], ["x", "std"])
    xraw_key = find_key(keys, ["X_raw", "features_raw"], ["x", "raw"])
    if xstd_key is None:
        raise SystemExit(f"[ERROR] Could not find X_std in {npz_path}; keys={keys}")

    X_std = np.asarray(z[xstd_key], dtype=np.float32)
    X_raw = np.asarray(z[xraw_key], dtype=np.float32) if xraw_key else None

    y_logret: Dict[int, np.ndarray] = {}
    for tau in TAUS:
        candidates = [
            f"y_logret_tau_{tau}",
            f"logret_tau_{tau}",
            f"Y_logret_tau_{tau}",
            f"target_logret_tau_{tau}",
            f"y_tau_{tau}",
        ]
        k = find_key(keys, candidates, ["log", str(tau)])
        if k is not None:
            y_logret[tau] = np.asarray(z[k], dtype=np.float32)

    if not y_logret:
        mid_key = find_key(keys, ["mid_price", "mid", "m"], ["mid"])
        if mid_key is None:
            raise SystemExit(f"[ERROR] No log-return targets and no mid_price found; keys={keys}")
        mid = np.asarray(z[mid_key], dtype=np.float64)
        for tau in TAUS:
            y = np.full_like(mid, np.nan, dtype=np.float32)
            y[:-tau] = np.log(mid[tau:] + EPS) - np.log(mid[:-tau] + EPS)
            y_logret[tau] = y

    return X_std, X_raw, y_logret, keys, xstd_key


def train_val_test_indices(tau: int):
    train = np.arange(0, TRAIN_END - tau)
    val = np.arange(TRAIN_END, VAL_END - tau)
    test = np.arange(VAL_END, T_EXPECTED - tau)
    return train, val, test


def row_normalize_graph(G: np.ndarray) -> np.ndarray:
    G = np.asarray(G, dtype=np.float32)
    denom = np.sum(np.abs(G), axis=1, keepdims=True)
    return G / (denom + EPS)


def make_design(X: np.ndarray, Gnorm: np.ndarray) -> np.ndarray:
    # X: B,N,D. Return Dmat: B,N,2D+1.
    X_nei = np.einsum("ij,bjd->bid", Gnorm, X, optimize=True)
    ones = np.ones((*X.shape[:2], 1), dtype=np.float32)
    return np.concatenate([X, X_nei, ones], axis=2)


def predict_from_X(X: np.ndarray, theta: np.ndarray, Gnorm: np.ndarray) -> np.ndarray:
    Dmat = make_design(X, Gnorm)
    return np.einsum("bnd,k->bn", Dmat, theta, optimize=True)


def project_U(U: np.ndarray, rho_x: float, rho_row: float) -> np.ndarray:
    if rho_x <= 0:
        return np.zeros_like(U)
    # Row-wise per asset projection for each time sample.
    row_norm = np.linalg.norm(U, axis=2, keepdims=True)
    row_scale = np.minimum(1.0, rho_row / (row_norm + EPS))
    U = U * row_scale
    # Per-time Frobenius projection.
    fro = np.linalg.norm(U.reshape(U.shape[0], -1), axis=1, keepdims=True).reshape(-1, 1, 1)
    fro_scale = np.minimum(1.0, rho_x / (fro + EPS))
    return U * fro_scale


def grad_loss_wrt_X(Xadv: np.ndarray, y: np.ndarray, theta: np.ndarray, Gnorm: np.ndarray) -> np.ndarray:
    # Model: pred_i = X_i dot w0 + sum_j G_ij X_j dot w1 + b.
    # dL/dX_a,k = mean_factor * sum_i error_i * (1_{i=a} w0_k + G_{i,a} w1_k).
    B, N, D = Xadv.shape
    w0 = theta[:D]
    w1 = theta[D:2*D]
    pred = predict_from_X(Xadv, theta, Gnorm)
    err = pred - y
    coeff = (2.0 / float(B * N)) * err  # B,N
    grad_self = coeff[:, :, None] * w0[None, None, :]  # B,N,D
    # For neighbor term: grad_X_a += sum_i coeff_i G_i,a w1
    neigh_coeff = np.einsum("bi,ia->ba", coeff, Gnorm, optimize=True)  # B,N-as-a
    grad_neigh = neigh_coeff[:, :, None] * w1[None, None, :]
    return (grad_self + grad_neigh).astype(np.float32)


def pgda_inner_X(X: np.ndarray, y: np.ndarray, theta: np.ndarray, Gnorm: np.ndarray, rho_x: float, r_adv: int) -> np.ndarray:
    if rho_x <= 0:
        return np.zeros_like(X)
    rho_row = rho_x / math.sqrt(N_EXPECTED)
    eta_u = ETA_U_FACTOR * rho_x / max(r_adv, 1)
    U = np.zeros_like(X, dtype=np.float32)
    for _ in range(r_adv):
        gradU = grad_loss_wrt_X(X + U, y, theta, Gnorm)
        U = U + eta_u * gradU
        U = project_U(U, rho_x=rho_x, rho_row=rho_row)
    return U.astype(np.float32)


def eval_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    mse = float(np.mean((yp - yt) ** 2))
    mae = float(np.mean(np.abs(yp - yt)))
    nonzero = np.abs(yt) > 0
    da = float(np.mean(np.sign(yp[nonzero]) == np.sign(yt[nonzero]))) if np.any(nonzero) else float("nan")
    # reshape after valid horizon use; y_true shape B,N.
    yp2 = y_pred.reshape(-1, N_EXPECTED)
    yt2 = y_true.reshape(-1, N_EXPECTED)
    denom = np.sum(np.abs(yp2), axis=1, keepdims=True) + EPS
    w = yp2 / denom
    pnl = np.sum(w * yt2, axis=1)
    sharpe = float(np.mean(pnl) / (np.std(pnl) + EPS))
    return {"mse": mse, "mae": mae, "directional_accuracy": da, "sharpe_proxy_no_tc": sharpe}


def fit_closed_form_ridge(X: np.ndarray, y: np.ndarray, Gnorm: np.ndarray, idx: np.ndarray, alpha: float = 1e-2):
    Dmat = make_design(X[idx], Gnorm).reshape(len(idx) * N_EXPECTED, 2 * D_EXPECTED + 1)
    yy = y[idx].reshape(len(idx) * N_EXPECTED)
    mask = np.isfinite(yy) & np.all(np.isfinite(Dmat), axis=1)
    A = Dmat[mask].astype(np.float64)
    b = yy[mask].astype(np.float64)
    reg = alpha * np.eye(A.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0  # do not regularize intercept coordinate
    theta = np.linalg.solve(A.T @ A + reg, A.T @ b)
    return theta.astype(np.float32)


def train_feature_robust_sgd(
    X: np.ndarray,
    y: np.ndarray,
    Gnorm: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    rho_x: float,
    r_adv: int,
    seed: int = SEED,
):
    rng = np.random.default_rng(seed)
    theta = fit_closed_form_ridge(X, y, Gnorm, train_idx, alpha=1e-2)
    p = theta.shape[0]
    n = len(train_idx)

    best_theta = theta.copy()
    best_val = float("inf")

    for epoch in range(EPOCHS):
        perm = rng.permutation(train_idx)
        for start in range(0, n, BATCH_SIZE):
            batch_idx = perm[start:start+BATCH_SIZE]
            Xb = X[batch_idx].astype(np.float32)
            yb = y[batch_idx].astype(np.float32)
            if not np.all(np.isfinite(yb)):
                valid = np.all(np.isfinite(yb), axis=1)
                Xb = Xb[valid]
                yb = yb[valid]
            if len(Xb) == 0:
                continue

            U = pgda_inner_X(Xb, yb, theta, Gnorm, rho_x=rho_x, r_adv=r_adv)
            Xadv = Xb + U
            Dmat = make_design(Xadv, Gnorm)  # B,N,p
            pred = np.einsum("bnp,p->bn", Dmat, theta, optimize=True)
            err = pred - yb
            grad_theta = (2.0 / float(err.size)) * np.einsum("bnp,bn->p", Dmat, err, optimize=True)
            grad_theta[:-1] += 2.0 * WEIGHT_DECAY * theta[:-1]
            # Simple clipped gradient for stability.
            gn = np.linalg.norm(grad_theta)
            if gn > 1.0:
                grad_theta = grad_theta / (gn + EPS)
            theta = theta - LR * grad_theta.astype(np.float32)

        val_pred = predict_from_X(X[val_idx], theta, Gnorm)
        val_mse = eval_predictions(y[val_idx], val_pred)["mse"]
        if val_mse < best_val:
            best_val = val_mse
            best_theta = theta.copy()

    return best_theta, best_val


def adversarial_test_metrics(X: np.ndarray, y: np.ndarray, theta: np.ndarray, Gnorm: np.ndarray, test_idx: np.ndarray, rho_eval: float):
    Xte = X[test_idx]
    yte = y[test_idx]
    clean_pred = predict_from_X(Xte, theta, Gnorm)
    clean = eval_predictions(yte, clean_pred)
    U = pgda_inner_X(Xte, yte, theta, Gnorm, rho_x=rho_eval, r_adv=5)
    pert_pred = predict_from_X(Xte + U, theta, Gnorm)
    pert = eval_predictions(yte, pert_pred)
    return {
        "clean_mse": clean["mse"],
        "clean_da": clean["directional_accuracy"],
        "clean_sharpe_proxy": clean["sharpe_proxy_no_tc"],
        "pert_mse": pert["mse"],
        "pert_da": pert["directional_accuracy"],
        "pert_sharpe_proxy": pert["sharpe_proxy_no_tc"],
        "deg_mse": pert["mse"] - clean["mse"],
        "deg_da": clean["directional_accuracy"] - pert["directional_accuracy"],
        "deg_sharpe": clean["sharpe_proxy_no_tc"] - pert["sharpe_proxy_no_tc"],
    }


def sensitivity_score(X: np.ndarray, y: np.ndarray, theta: np.ndarray, Gnorm: np.ndarray, idx: np.ndarray, max_batches: int = 8):
    # Average ||grad_X loss||_F over a small deterministic subset of batches.
    vals = []
    for start in np.linspace(0, max(0, len(idx) - BATCH_SIZE), num=max_batches, dtype=int):
        batch_idx = idx[start:start+BATCH_SIZE]
        if len(batch_idx) == 0:
            continue
        Xb = X[batch_idx]
        yb = y[batch_idx]
        g = grad_loss_wrt_X(Xb, yb, theta, Gnorm)
        per_t = np.linalg.norm(g.reshape(g.shape[0], -1), axis=1)
        vals.append(float(np.mean(per_t)))
    return float(np.mean(vals)) if vals else float("nan")


def markdown_table(rows: List[Dict], columns: List[str]) -> str:
    out = []
    out.append("| " + " | ".join(columns) + " |")
    out.append("|" + "|".join(["---"] * len(columns)) + "|")
    for r in rows:
        vals = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                if math.isnan(v):
                    vals.append("nan")
                elif abs(v) < 1e-3 or abs(v) > 1e4:
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main():
    project = Path(".").resolve()
    processed = project / "data" / "processed"
    reports = project / "reports" / "benchmarks"
    state = project / ".pipeline_state"

    checks: List[Check] = []

    def add(name, ok, observed, expected):
        checks.append(Check(name, "PASS" if ok else "FAIL", str(observed), str(expected)))

    X, Xraw, y_logret, npz_keys, xstd_key = load_step02(processed / "features_500ms_step02.npz")
    gnpz = np.load(processed / "graphs_step03.npz")
    if DEFAULT_GRAPH not in gnpz.files:
        raise SystemExit(f"[ERROR] {DEFAULT_GRAPH} not found in graphs_step03.npz. Keys={gnpz.files}")
    G = np.asarray(gnpz[DEFAULT_GRAPH], dtype=np.float32)
    Gnorm = row_normalize_graph(G)

    add("step03_gate_exists", (state / "step03_graph_baseline.PASS").exists(), ".pipeline_state/step03_graph_baseline.PASS", "exists")
    add("X_shape", tuple(X.shape) == (T_EXPECTED, N_EXPECTED, D_EXPECTED), tuple(X.shape), (T_EXPECTED, N_EXPECTED, D_EXPECTED))
    add("graph_key_exists", DEFAULT_GRAPH in gnpz.files, DEFAULT_GRAPH, f"in {gnpz.files}")
    add("graph_shape", tuple(G.shape) == (N_EXPECTED, N_EXPECTED), tuple(G.shape), (N_EXPECTED, N_EXPECTED))
    add("graph_diag_zero", float(np.max(np.abs(np.diag(G)))) < 1e-8, float(np.max(np.abs(np.diag(G)))), "0")
    add("targets_available", all(t in y_logret for t in TAUS), sorted(y_logret.keys()), TAUS)

    results = []
    best_by_tau = []
    improvement_rows = []

    # Train robust models. Keep default graph fixed from Step 03.
    for tau in TAUS:
        y = y_logret[tau]
        train_idx, val_idx, test_idx = train_val_test_indices(tau)

        # ERM reference: rho=0.
        tau_rows = []
        for rho_x in RHO_X_GRID:
            r_grid = [1] if rho_x == 0.0 else R_ADV_GRID
            for r_adv in r_grid:
                theta, val_mse = train_feature_robust_sgd(
                    X, y, Gnorm, train_idx, val_idx, rho_x=rho_x, r_adv=r_adv, seed=SEED + tau + int(1000*rho_x) + r_adv
                )
                sens = sensitivity_score(X, y, theta, Gnorm, test_idx)
                m0 = adversarial_test_metrics(X, y, theta, Gnorm, test_idx, rho_eval=0.0)
                m05 = adversarial_test_metrics(X, y, theta, Gnorm, test_idx, rho_eval=0.05)
                m10 = adversarial_test_metrics(X, y, theta, Gnorm, test_idx, rho_eval=0.10)
                rec = {
                    "tau": tau,
                    "model": "graph_linear_feature_pgda" if rho_x > 0 else "graph_linear_sgd_erm",
                    "graph": DEFAULT_GRAPH,
                    "rho_x_train": rho_x,
                    "r_adv": r_adv,
                    "val_mse": val_mse,
                    "clean_mse": m0["clean_mse"],
                    "clean_da": m0["clean_da"],
                    "clean_sharpe_proxy": m0["clean_sharpe_proxy"],
                    "sensitivity_X": sens,
                    "eval_rho_0.05_mse": m05["pert_mse"],
                    "eval_rho_0.05_deg_mse": m05["deg_mse"],
                    "eval_rho_0.05_deg_da": m05["deg_da"],
                    "eval_rho_0.10_mse": m10["pert_mse"],
                    "eval_rho_0.10_deg_mse": m10["deg_mse"],
                    "eval_rho_0.10_deg_da": m10["deg_da"],
                }
                results.append(rec)
                tau_rows.append(rec)

        best = min(tau_rows, key=lambda r: r["val_mse"])
        best_by_tau.append(best)

        erm = min([r for r in tau_rows if r["rho_x_train"] == 0.0], key=lambda r: r["val_mse"])
        robust_candidates = [r for r in tau_rows if r["rho_x_train"] > 0.0]
        robust_best_deg = min(robust_candidates, key=lambda r: r["eval_rho_0.10_deg_mse"])
        improvement_rows.append({
            "tau": tau,
            "erm_clean_mse": erm["clean_mse"],
            "robust_clean_mse": robust_best_deg["clean_mse"],
            "erm_deg_mse_rho_0.10": erm["eval_rho_0.10_deg_mse"],
            "robust_deg_mse_rho_0.10": robust_best_deg["eval_rho_0.10_deg_mse"],
            "deg_mse_improvement": erm["eval_rho_0.10_deg_mse"] - robust_best_deg["eval_rho_0.10_deg_mse"],
            "erm_sensitivity_X": erm["sensitivity_X"],
            "robust_sensitivity_X": robust_best_deg["sensitivity_X"],
            "sensitivity_improvement": erm["sensitivity_X"] - robust_best_deg["sensitivity_X"],
            "robust_choice": f"rho={robust_best_deg['rho_x_train']},R={robust_best_deg['r_adv']}",
        })

    # Benchmarks: not requiring robust to outperform, only that degradation/sensitivity are computed and finite.
    finite_fields = []
    for r in results:
        for k in ["val_mse", "clean_mse", "clean_da", "sensitivity_X", "eval_rho_0.05_deg_mse", "eval_rho_0.10_deg_mse"]:
            finite_fields.append(np.isfinite(r[k]))
    add("results_nonempty", len(results) > 0, len(results), "> 0")
    add("all_core_metrics_finite", bool(np.all(finite_fields)), f"{sum(finite_fields)}/{len(finite_fields)} finite", "all finite")
    da_vals = [r["clean_da"] for r in results]
    add("clean_da_range", min(da_vals) >= 0.0 and max(da_vals) <= 1.0, f"min={min(da_vals):.4f}, max={max(da_vals):.4f}", "[0,1]")
    deg_vals = [r["eval_rho_0.10_deg_mse"] for r in results]
    add("degradation_nonnegative_mostly", np.mean(np.array(deg_vals) >= -1e-12) >= 0.90, f"{np.mean(np.array(deg_vals) >= -1e-12):.2%}", ">= 90%")
    add("has_feature_robust_models", any(r["rho_x_train"] > 0 for r in results), "rho_x > 0 present", "True")

    status = "PASS" if all(c.status == "PASS" for c in checks) else "FAIL"

    # Save CSV.
    cols = [
        "tau", "model", "graph", "rho_x_train", "r_adv", "val_mse",
        "clean_mse", "clean_da", "clean_sharpe_proxy", "sensitivity_X",
        "eval_rho_0.05_mse", "eval_rho_0.05_deg_mse", "eval_rho_0.05_deg_da",
        "eval_rho_0.10_mse", "eval_rho_0.10_deg_mse", "eval_rho_0.10_deg_da",
    ]
    with open(processed / "feature_robust_results_step04.csv", "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    summary = {
        "status": status,
        "graph": DEFAULT_GRAPH,
        "rho_x_grid": RHO_X_GRID,
        "r_adv_grid": R_ADV_GRID,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "results": results,
        "best_by_tau": best_by_tau,
        "improvement_rows": improvement_rows,
        "checks": [c.__dict__ for c in checks],
    }
    with open(reports / "step04_feature_robust_pgda.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md = []
    md.append("# Step 04 Feature-Robust PGDA Benchmark\n")
    md.append(f"Overall status: **{status}**\n")
    md.append("## Input contract")
    md.append(f"- Feature tensor: `{X.shape}` from key `{xstd_key}`")
    md.append(f"- Graph file: `data/processed/graphs_step03.npz`")
    md.append(f"- Default graph: `{DEFAULT_GRAPH}`")
    md.append(f"- Training grid: `rho_X={RHO_X_GRID}`, `R_adv={R_ADV_GRID}`, epochs `{EPOCHS}`, batch size `{BATCH_SIZE}`")
    md.append("")

    md.append("## Best validation-selected model by horizon")
    md.append(markdown_table(best_by_tau, [
        "tau", "model", "rho_x_train", "r_adv", "val_mse", "clean_mse", "clean_da",
        "sensitivity_X", "eval_rho_0.10_deg_mse", "eval_rho_0.10_deg_da"
    ]))
    md.append("")

    md.append("## ERM vs best robust degradation comparison")
    md.append(markdown_table(improvement_rows, [
        "tau", "erm_clean_mse", "robust_clean_mse",
        "erm_deg_mse_rho_0.10", "robust_deg_mse_rho_0.10", "deg_mse_improvement",
        "erm_sensitivity_X", "robust_sensitivity_X", "sensitivity_improvement", "robust_choice"
    ]))
    md.append("")

    md.append("## All feature-robust results")
    md.append(markdown_table(results, [
        "tau", "rho_x_train", "r_adv", "val_mse", "clean_mse", "clean_da",
        "sensitivity_X", "eval_rho_0.05_deg_mse", "eval_rho_0.10_deg_mse"
    ]))
    md.append("")

    md.append("## Benchmark checks")
    md.append("| check | status | observed | expected |")
    md.append("|---|---|---|---|")
    for c in checks:
        md.append(f"| {c.name} | {c.status} | `{c.observed}` | `{c.expected}` |")
    md.append("")

    if status == "PASS":
        (state / "step04_feature_robust_pgda.PASS").write_text("PASS\n", encoding="utf-8")
        md.append("Gate passed: `.pipeline_state/step04_feature_robust_pgda.PASS` should exist.")
        md.append("Next module should be `05_graph_robust_pgda.sh`, but do not run it until this report is reviewed.")
    else:
        (state / "step04_feature_robust_pgda.FAIL").write_text("FAIL\n", encoding="utf-8")
        md.append("Gate failed: inspect `reports/benchmarks/step04_feature_robust_pgda.json`.")

    report_text = "\n".join(md)
    (reports / "step04_feature_robust_pgda.md").write_text(report_text, encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

EPS = 1e-12
T_EXPECTED = 46800
N = 5
D = 51
TRAIN_END = 32760
VAL_END = 39780
TAUS = [1, 2, 10, 20]

# Keep the first graph-robust pass small and reproducible.
# return_abs was the strongest Step 03 graph in several horizons.
BASE_GRAPH_KEY = "return_abs"
RHO_G_TRAIN_GRID = [0.0, 0.05, 0.10, 0.20]
R_ADV_GRID = [1, 3]
RHO_G_EVAL_GRID = [0.01, 0.05, 0.10, 0.20]
EPOCHS = 7
BATCH_SIZE = 512
LR = 0.03
WEIGHT_DECAY = 1e-4
SEED = 23

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

def load_step02():
    z = np.load("data/processed/features_500ms_step02.npz", allow_pickle=True)
    keys = list(z.keys())
    xkey = find_key(keys, ["X_std", "X_tilde", "X_standardized", "features_std"], ["x", "std"])
    if xkey is None:
        raise SystemExit(f"[ERROR] Cannot find X_std in feature NPZ. Keys={keys}")
    X = np.asarray(z[xkey], dtype=np.float32)

    y = {}
    for tau in TAUS:
        k = find_key(keys, [f"y_logret_tau_{tau}", f"logret_tau_{tau}", f"Y_logret_tau_{tau}", f"target_logret_tau_{tau}", f"y_tau_{tau}"], ["log", str(tau)])
        if k is None:
            raise SystemExit(f"[ERROR] Cannot find log-return target for tau={tau}. Keys={keys}")
        y[tau] = np.asarray(z[k], dtype=np.float32)
    return X, y, xkey, keys

def split_idx(tau: int):
    train = np.arange(0, TRAIN_END - tau)
    val = np.arange(TRAIN_END, VAL_END - tau)
    test = np.arange(VAL_END, T_EXPECTED - tau)
    return train, val, test

def row_normalize_graph(G: np.ndarray) -> np.ndarray:
    G = np.asarray(G, dtype=np.float32)
    denom = np.sum(np.abs(G), axis=1, keepdims=True)
    out = G / (denom + EPS)
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 0.0)
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)

def project_delta(Delta: np.ndarray, G_base: np.ndarray, rho: float) -> np.ndarray:
    if rho <= 0:
        return np.zeros_like(Delta, dtype=np.float32)
    A = G_base + Delta
    A = 0.5 * (A + A.T)
    A = np.clip(A, 0.0, 1.0)
    np.fill_diagonal(A, 0.0)
    Delta = A - G_base
    fro = float(np.linalg.norm(Delta))
    if fro > rho:
        Delta = Delta * (rho / (fro + EPS))
    # Convex combination with G_base preserves [0,1] and zero diagonal after scaling.
    A2 = G_base + Delta
    A2 = 0.5 * (A2 + A2.T)
    A2 = np.clip(A2, 0.0, 1.0)
    np.fill_diagonal(A2, 0.0)
    return (A2 - G_base).astype(np.float32)

def make_design(X: np.ndarray, G: np.ndarray) -> np.ndarray:
    X_nei = np.einsum("ij,tjd->tid", G, X, optimize=True)
    ones = np.ones((*X.shape[:2], 1), dtype=np.float32)
    return np.concatenate([X, X_nei, ones], axis=2)

def predict(X: np.ndarray, theta: np.ndarray, G: np.ndarray) -> np.ndarray:
    return np.einsum("tnp,p->tn", make_design(X, G), theta, optimize=True)

def metric(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(pred)
    yy = y[mask]
    pp = pred[mask]
    mse = float(np.mean((pp - yy) ** 2))
    mae = float(np.mean(np.abs(pp - yy)))
    nz = np.abs(yy) > 0
    da = float(np.mean(np.sign(pp[nz]) == np.sign(yy[nz]))) if np.any(nz) else float("nan")
    pp2 = pred.reshape(-1, N)
    yy2 = y.reshape(-1, N)
    w = pp2 / (np.sum(np.abs(pp2), axis=1, keepdims=True) + EPS)
    pnl = np.sum(w * yy2, axis=1)
    sharpe = float(np.mean(pnl) / (np.std(pnl) + EPS))
    return {"mse": mse, "mae": mae, "da": da, "sharpe": sharpe}

def fit_ridge(X: np.ndarray, y: np.ndarray, G: np.ndarray, idx: np.ndarray, alpha: float = 1e-2) -> np.ndarray:
    Dmat = make_design(X[idx], G).reshape(len(idx) * N, 2 * D + 1)
    yy = y[idx].reshape(len(idx) * N)
    mask = np.isfinite(yy) & np.all(np.isfinite(Dmat), axis=1)
    A = Dmat[mask].astype(np.float64)
    b = yy[mask].astype(np.float64)
    reg = alpha * np.eye(A.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    theta = np.linalg.solve(A.T @ A + reg, A.T @ b)
    return theta.astype(np.float32)

def grad_theta(X: np.ndarray, y: np.ndarray, theta: np.ndarray, G: np.ndarray) -> np.ndarray:
    Dmat = make_design(X, G)
    pred = np.einsum("tnp,p->tn", Dmat, theta, optimize=True)
    err = pred - y
    grad = 2.0 * np.einsum("tnp,tn->p", Dmat, err, optimize=True) / float(err.size)
    grad[:-1] += 2.0 * WEIGHT_DECAY * theta[:-1]
    return grad.astype(np.float32)

def grad_G(X: np.ndarray, y: np.ndarray, theta: np.ndarray, G: np.ndarray) -> np.ndarray:
    # pred_i = X_i w0 + sum_j G_ij X_j w1 + b
    # dL/dG_ij = sum_t coeff_{t,i} * <X_{t,j}, w1>
    w1 = theta[D:2*D]
    pred = predict(X, theta, G)
    err = pred - y
    coeff = 2.0 * err / float(err.size)  # T,N
    xw1 = np.einsum("tjd,d->tj", X, w1, optimize=True)  # T,N
    g = np.einsum("ti,tj->ij", coeff, xw1, optimize=True)
    # Symmetric graph constraint: use symmetric part and zero diagonal.
    g = 0.5 * (g + g.T)
    np.fill_diagonal(g, 0.0)
    return g.astype(np.float32)

def normalized_graph_attack(X: np.ndarray, y: np.ndarray, theta: np.ndarray, G_base: np.ndarray, rho: float, steps: int) -> np.ndarray:
    if rho <= 0:
        return np.zeros_like(G_base, dtype=np.float32)
    Delta = np.zeros_like(G_base, dtype=np.float32)
    step = rho / float(max(steps, 1))
    for _ in range(steps):
        g = grad_G(X, y, theta, G_base + Delta)
        gn = float(np.linalg.norm(g))
        direction = g / (gn + EPS)
        Delta = Delta + step * direction
        Delta = project_delta(Delta, G_base, rho)
    return Delta.astype(np.float32)

def sensitivity_G(X: np.ndarray, y: np.ndarray, theta: np.ndarray, G_base: np.ndarray, idx: np.ndarray, max_batches: int = 8) -> float:
    vals = []
    starts = np.linspace(0, max(0, len(idx) - BATCH_SIZE), num=max_batches, dtype=int)
    for s in starts:
        bi = idx[s:s+BATCH_SIZE]
        if len(bi) == 0:
            continue
        g = grad_G(X[bi], y[bi], theta, G_base)
        vals.append(float(np.linalg.norm(g)))
    return float(np.mean(vals)) if vals else float("nan")

def train_graph_robust(
    X: np.ndarray,
    y: np.ndarray,
    G_base: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    rho_g: float,
    r_adv: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    theta = fit_ridge(X, y, G_base, train, alpha=1e-2)
    best_theta = theta.copy()
    best_val = float("inf")

    for epoch in range(EPOCHS):
        perm = rng.permutation(train)
        for start in range(0, len(perm), BATCH_SIZE):
            bi = perm[start:start+BATCH_SIZE]
            Xb = X[bi]
            yb = y[bi]
            valid = np.all(np.isfinite(yb), axis=1)
            Xb = Xb[valid]
            yb = yb[valid]
            if len(Xb) == 0:
                continue

            Delta = normalized_graph_attack(Xb, yb, theta, G_base, rho=rho_g, steps=r_adv)
            Gadv = G_base + Delta
            gt = grad_theta(Xb, yb, theta, Gadv)
            gn = float(np.linalg.norm(gt))
            if gn > 1.0:
                gt = gt / (gn + EPS)
            theta = theta - LR * gt.astype(np.float32)

        val_pred = predict(X[val], theta, G_base)
        val_mse = metric(y[val], val_pred)["mse"]
        if val_mse < best_val:
            best_val = val_mse
            best_theta = theta.copy()

    return best_theta, best_val

def md_table(rows: List[Dict], cols: List[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        vals = []
        for c in cols:
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
    reports = Path("reports/benchmarks")
    processed = Path("data/processed")
    state = Path(".pipeline_state")
    checks: List[Check] = []

    def add(name, ok, observed, expected):
        checks.append(Check(name, "PASS" if ok else "FAIL", str(observed), str(expected)))

    X, ys, xkey, npz_keys = load_step02()
    graphs = np.load(processed / "graphs_step03.npz")
    if BASE_GRAPH_KEY not in graphs.files:
        raise SystemExit(f"[ERROR] Base graph key `{BASE_GRAPH_KEY}` not found. Keys={graphs.files}")
    G_raw = np.asarray(graphs[BASE_GRAPH_KEY], dtype=np.float32)
    G_base = row_normalize_graph(G_raw)

    add("step04b_gate_exists", (state / "step04b_calibrated_eval.PASS").exists(), ".pipeline_state/step04b_calibrated_eval.PASS", "exists")
    add("X_shape", tuple(X.shape) == (T_EXPECTED, N, D), tuple(X.shape), (T_EXPECTED, N, D))
    add("base_graph_key_exists", BASE_GRAPH_KEY in graphs.files, BASE_GRAPH_KEY, "present in graphs_step03.npz")
    add("base_graph_shape", tuple(G_base.shape) == (N, N), tuple(G_base.shape), (N, N))
    add("base_graph_diag_zero", float(np.max(np.abs(np.diag(G_base)))) < 1e-8, float(np.max(np.abs(np.diag(G_base)))), "0")
    add("base_graph_range", float(np.min(G_base)) >= -1e-8 and float(np.max(G_base)) <= 1.0 + 1e-8, f"[{float(np.min(G_base)):.4g},{float(np.max(G_base)):.4g}]", "[0,1]")

    rows: List[Dict] = []
    best_by_tau: List[Dict] = []
    improvement_rows: List[Dict] = []

    for tau in TAUS:
        y = ys[tau]
        train, val, test = split_idx(tau)
        tau_rows: List[Dict] = []

        for rho_g in RHO_G_TRAIN_GRID:
            rlist = [1] if rho_g == 0 else R_ADV_GRID
            for r_adv in rlist:
                theta, val_mse = train_graph_robust(
                    X, y, G_base, train, val,
                    rho_g=rho_g, r_adv=r_adv,
                    seed=SEED + tau + int(1000*rho_g) + r_adv,
                )
                clean_pred = predict(X[test], theta, G_base)
                clean = metric(y[test], clean_pred)
                rec = {
                    "tau": tau,
                    "rho_g_train": rho_g,
                    "r_adv": r_adv,
                    "val_mse": val_mse,
                    "clean_mse": clean["mse"],
                    "clean_da": clean["da"],
                    "clean_sharpe": clean["sharpe"],
                    "sensitivity_G": sensitivity_G(X, y, theta, G_base, test),
                }
                for rho_eval in RHO_G_EVAL_GRID:
                    Delta = normalized_graph_attack(X[test], y[test], theta, G_base, rho=rho_eval, steps=7)
                    pert_pred = predict(X[test], theta, G_base + Delta)
                    pert = metric(y[test], pert_pred)
                    rec[f"pert_mse_{rho_eval}"] = pert["mse"]
                    rec[f"deg_mse_{rho_eval}"] = pert["mse"] - clean["mse"]
                    rec[f"deg_da_{rho_eval}"] = clean["da"] - pert["da"]
                    rec[f"delta_fro_{rho_eval}"] = float(np.linalg.norm(Delta))
                rows.append(rec)
                tau_rows.append(rec)

        best = min(tau_rows, key=lambda r: r["val_mse"])
        best_by_tau.append(best)

        erm = [r for r in tau_rows if r["rho_g_train"] == 0.0][0]
        robust = min([r for r in tau_rows if r["rho_g_train"] > 0.0], key=lambda r: r["deg_mse_0.1"])
        improvement_rows.append({
            "tau": tau,
            "erm_clean_mse": erm["clean_mse"],
            "robust_clean_mse": robust["clean_mse"],
            "erm_deg_mse_0.1": erm["deg_mse_0.1"],
            "robust_deg_mse_0.1": robust["deg_mse_0.1"],
            "deg_mse_improvement": erm["deg_mse_0.1"] - robust["deg_mse_0.1"],
            "erm_sensitivity_G": erm["sensitivity_G"],
            "robust_sensitivity_G": robust["sensitivity_G"],
            "sensitivity_improvement": erm["sensitivity_G"] - robust["sensitivity_G"],
            "robust_choice": f"rho={robust['rho_g_train']},R={robust['r_adv']}",
        })

    finite_vals = []
    for r in rows:
        for k, v in r.items():
            if isinstance(v, float):
                finite_vals.append(np.isfinite(v))

    degs = np.array([r["deg_mse_0.1"] for r in rows], dtype=float)
    deltas = np.array([r["delta_fro_0.1"] for r in rows], dtype=float)
    add("results_nonempty", len(rows) > 0, len(rows), ">0")
    add("metrics_finite", bool(np.all(finite_vals)), f"{int(np.sum(finite_vals))}/{len(finite_vals)} finite", "all finite")
    add("graph_attack_uses_radius", np.mean(deltas > 0.09) >= 0.75, f"mean delta_fro_0.1={float(np.mean(deltas)):.4g}", "most attacks near rho=0.1")
    add("graph_degradation_nonzero", np.mean(np.abs(degs) > 1e-14) >= 0.75, f"{float(np.mean(np.abs(degs) > 1e-14)):.2%}", ">=75% nonzero")
    add("graph_degradation_nonnegative_mostly", np.mean(degs >= -1e-14) >= 0.80, f"{float(np.mean(degs >= -1e-14)):.2%}", ">=80%")
    add("has_graph_robust_models", any(r["rho_g_train"] > 0 for r in rows), "rho_g > 0 present", "True")

    status = "PASS" if all(c.status == "PASS" for c in checks) else "FAIL"

    cols = [
        "tau", "rho_g_train", "r_adv", "val_mse", "clean_mse", "clean_da", "clean_sharpe",
        "sensitivity_G", "deg_mse_0.01", "deg_mse_0.05", "deg_mse_0.1", "deg_mse_0.2",
        "deg_da_0.1", "delta_fro_0.1",
    ]
    with open(processed / "graph_robust_results_step05.csv", "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    summary = {
        "status": status,
        "base_graph_key": BASE_GRAPH_KEY,
        "rho_g_train_grid": RHO_G_TRAIN_GRID,
        "rho_g_eval_grid": RHO_G_EVAL_GRID,
        "r_adv_grid": R_ADV_GRID,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "results": rows,
        "best_by_tau": best_by_tau,
        "improvement_rows": improvement_rows,
        "checks": [c.__dict__ for c in checks],
    }
    with open(reports / "step05_graph_robust_pgda.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md = []
    md.append("# Step 05 Graph-Robust PGDA Benchmark\n")
    md.append(f"Overall status: **{status}**\n")
    md.append("## Purpose")
    md.append("This module tests graph perturbation robustness with fixed standardized features and adversarial perturbations of the cross-asset graph. The base graph is constrained to the unsigned graph set with zero diagonal.\n")
    md.append("## Input contract")
    md.append(f"- Feature tensor: `{X.shape}` from key `{xkey}`")
    md.append(f"- Base graph: `{BASE_GRAPH_KEY}` from `data/processed/graphs_step03.npz`")
    md.append(f"- Base graph shape/range: `{G_base.shape}`, min `{float(np.min(G_base)):.6g}`, max `{float(np.max(G_base)):.6g}`")
    md.append(f"- Training grid: `rho_G={RHO_G_TRAIN_GRID}`, `R_adv={R_ADV_GRID}`, epochs `{EPOCHS}`, batch size `{BATCH_SIZE}`")
    md.append("")

    md.append("## Best validation-selected model by horizon")
    md.append(md_table(best_by_tau, [
        "tau", "rho_g_train", "r_adv", "val_mse", "clean_mse", "clean_da",
        "sensitivity_G", "deg_mse_0.1", "deg_da_0.1", "delta_fro_0.1"
    ]))
    md.append("")

    md.append("## ERM vs best graph-robust degradation comparison")
    md.append(md_table(improvement_rows, [
        "tau", "erm_clean_mse", "robust_clean_mse",
        "erm_deg_mse_0.1", "robust_deg_mse_0.1", "deg_mse_improvement",
        "erm_sensitivity_G", "robust_sensitivity_G", "sensitivity_improvement", "robust_choice"
    ]))
    md.append("")

    md.append("## All graph-robust results")
    md.append(md_table(rows, cols))
    md.append("")

    md.append("## Benchmark checks")
    md.append("| check | status | observed | expected |")
    md.append("|---|---|---|---|")
    for c in checks:
        md.append(f"| {c.name} | {c.status} | `{c.observed}` | `{c.expected}` |")
    md.append("")

    if status == "PASS":
        (state / "step05_graph_robust_pgda.PASS").write_text("PASS\n", encoding="utf-8")
        md.append("Gate passed: `.pipeline_state/step05_graph_robust_pgda.PASS` should exist.")
        md.append("Next module should be `06_feature_graph_robust_summary.sh`, but do not run it until this report is reviewed.")
    else:
        (state / "step05_graph_robust_pgda.FAIL").write_text("FAIL\n", encoding="utf-8")
        md.append("Gate failed: inspect `reports/benchmarks/step05_graph_robust_pgda.json`.")

    text = "\n".join(md)
    (reports / "step05_graph_robust_pgda.md").write_text(text, encoding="utf-8")
    print(text)

if __name__ == "__main__":
    main()

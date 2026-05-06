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
RHO_EVAL_GRID = [0.01, 0.05, 0.10, 0.20]
RHO_TRAIN_GRID = [0.0, 0.05, 0.10, 0.20]
R_ADV_GRID = [1, 3]
DEFAULT_GRAPH = "return_abs"
SEED = 11

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

def load_data():
    z = np.load("data/processed/features_500ms_step02.npz", allow_pickle=True)
    keys = list(z.keys())
    xkey = find_key(keys, ["X_std", "X_tilde", "X_standardized"], ["x", "std"])
    if xkey is None:
        raise SystemExit(f"[ERROR] Cannot find X_std. Keys={keys}")
    X = np.asarray(z[xkey], dtype=np.float32)

    y = {}
    for tau in TAUS:
        k = find_key(keys, [f"y_logret_tau_{tau}", f"logret_tau_{tau}", f"Y_logret_tau_{tau}", f"y_tau_{tau}"], ["log", str(tau)])
        if k is None:
            raise SystemExit(f"[ERROR] Cannot find log-return target for tau={tau}. Keys={keys}")
        y[tau] = np.asarray(z[k], dtype=np.float32)
    return X, y, xkey, keys

def idx_split(tau: int):
    train = np.arange(0, TRAIN_END - tau)
    val = np.arange(TRAIN_END, VAL_END - tau)
    test = np.arange(VAL_END, T_EXPECTED - tau)
    return train, val, test

def row_normalize_graph(G):
    denom = np.sum(np.abs(G), axis=1, keepdims=True)
    return G / (denom + EPS)

def make_design(X, G):
    Xn = np.einsum("ij,tjd->tid", G, X, optimize=True)
    ones = np.ones((*X.shape[:2], 1), dtype=np.float32)
    return np.concatenate([X, Xn, ones], axis=2)

def predict(X, theta, G):
    return np.einsum("tnp,p->tn", make_design(X, G), theta, optimize=True)

def fit_ridge(X, y, G, idx, alpha=1e-2):
    Dmat = make_design(X[idx], G).reshape(len(idx)*N, 2*D+1)
    yy = y[idx].reshape(len(idx)*N)
    mask = np.isfinite(yy) & np.all(np.isfinite(Dmat), axis=1)
    A = Dmat[mask].astype(np.float64)
    b = yy[mask].astype(np.float64)
    reg = alpha * np.eye(A.shape[1])
    reg[-1, -1] = 0
    theta = np.linalg.solve(A.T @ A + reg, A.T @ b)
    return theta.astype(np.float32)

def grad_X(X, y, theta, G):
    B = X.shape[0]
    w0 = theta[:D]
    w1 = theta[D:2*D]
    pred = predict(X, theta, G)
    err = pred - y
    coeff = 2.0 * err / float(B*N)
    grad_self = coeff[:, :, None] * w0[None, None, :]
    neigh_coeff = np.einsum("ti,ia->ta", coeff, G, optimize=True)
    grad_neigh = neigh_coeff[:, :, None] * w1[None, None, :]
    return (grad_self + grad_neigh).astype(np.float32)

def project_to_feature_ball(U, rho):
    if rho <= 0:
        return np.zeros_like(U)
    # Per sample Frobenius ball with row-wise cap.
    rho_row = rho / math.sqrt(N)
    row_norm = np.linalg.norm(U, axis=2, keepdims=True)
    U = U * np.minimum(1.0, rho_row/(row_norm + EPS))
    fro = np.linalg.norm(U.reshape(U.shape[0], -1), axis=1).reshape(-1,1,1)
    U = U * np.minimum(1.0, rho/(fro + EPS))
    return U.astype(np.float32)

def normalized_pgd_attack(X, y, theta, G, rho, steps=5):
    if rho <= 0:
        return np.zeros_like(X)
    U = np.zeros_like(X, dtype=np.float32)
    step = rho / float(max(steps, 1))
    for _ in range(steps):
        g = grad_X(X + U, y, theta, G)
        gnorm = np.linalg.norm(g.reshape(g.shape[0], -1), axis=1).reshape(-1,1,1)
        # Normalize per time sample, so every valid sample moves in steepest-ascent direction.
        direction = g / (gnorm + EPS)
        U = U + step * direction
        U = project_to_feature_ball(U, rho)
    return U

def train_calibrated_robust(X, y, G, train_idx, val_idx, rho_train, r_adv, epochs=6, batch_size=512, lr=0.03, seed=0):
    rng = np.random.default_rng(seed)
    theta = fit_ridge(X, y, G, train_idx, alpha=1e-2)
    best_theta = theta.copy()
    best_val = np.inf

    for ep in range(epochs):
        perm = rng.permutation(train_idx)
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start+batch_size]
            Xb = X[idx]
            yb = y[idx]
            valid = np.all(np.isfinite(yb), axis=1)
            Xb = Xb[valid]
            yb = yb[valid]
            if len(Xb) == 0:
                continue
            U = normalized_pgd_attack(Xb, yb, theta, G, rho_train, steps=r_adv)
            Dmat = make_design(Xb + U, G)
            pred = np.einsum("tnp,p->tn", Dmat, theta, optimize=True)
            err = pred - yb
            grad = 2.0 * np.einsum("tnp,tn->p", Dmat, err, optimize=True) / float(err.size)
            grad[:-1] += 2e-4 * theta[:-1]
            gn = np.linalg.norm(grad)
            if gn > 1.0:
                grad = grad / (gn + EPS)
            theta = theta - lr * grad.astype(np.float32)

        vp = predict(X[val_idx], theta, G)
        vmse = metric(y[val_idx], vp)["mse"]
        if vmse < best_val:
            best_val = vmse
            best_theta = theta.copy()

    return best_theta, best_val

def metric(y, pred):
    mask = np.isfinite(y) & np.isfinite(pred)
    yy = y[mask]
    pp = pred[mask]
    mse = float(np.mean((pp-yy)**2))
    mae = float(np.mean(np.abs(pp-yy)))
    nz = np.abs(yy) > 0
    da = float(np.mean(np.sign(pp[nz]) == np.sign(yy[nz]))) if np.any(nz) else float("nan")
    pp2 = pred.reshape(-1, N)
    yy2 = y.reshape(-1, N)
    w = pp2 / (np.sum(np.abs(pp2), axis=1, keepdims=True) + EPS)
    pnl = np.sum(w * yy2, axis=1)
    sharpe = float(np.mean(pnl)/(np.std(pnl)+EPS))
    return {"mse": mse, "mae": mae, "da": da, "sharpe": sharpe}

def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"]*len(cols)) + "|"]
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
    state = Path(".pipeline_state")
    processed = Path("data/processed")
    checks = []

    def add(name, ok, observed, expected):
        checks.append(Check(name, "PASS" if ok else "FAIL", str(observed), str(expected)))

    X, ydict, xkey, keys = load_data()
    gnpz = np.load("data/processed/graphs_step03.npz")
    G = row_normalize_graph(np.asarray(gnpz[DEFAULT_GRAPH], dtype=np.float32))

    add("step04_gate_exists", Path(".pipeline_state/step04_feature_robust_pgda.PASS").exists(), ".pipeline_state/step04_feature_robust_pgda.PASS", "exists")
    add("X_shape", tuple(X.shape) == (T_EXPECTED,N,D), tuple(X.shape), (T_EXPECTED,N,D))
    add("graph_shape", tuple(G.shape) == (N,N), tuple(G.shape), (N,N))

    rows = []
    best_by_tau = []
    improvement_rows = []

    for tau in TAUS:
        y = ydict[tau]
        train, val, test = idx_split(tau)
        tau_rows = []
        for rho_train in RHO_TRAIN_GRID:
            rlist = [1] if rho_train == 0 else R_ADV_GRID
            for r_adv in rlist:
                theta, val_mse = train_calibrated_robust(X, y, G, train, val, rho_train, r_adv, seed=SEED+tau+int(1000*rho_train)+r_adv)
                clean_pred = predict(X[test], theta, G)
                clean = metric(y[test], clean_pred)
                rec = {
                    "tau": tau, "rho_x_train": rho_train, "r_adv": r_adv,
                    "val_mse": val_mse, "clean_mse": clean["mse"],
                    "clean_da": clean["da"], "clean_sharpe": clean["sharpe"],
                }
                for rho_eval in RHO_EVAL_GRID:
                    U = normalized_pgd_attack(X[test], y[test], theta, G, rho_eval, steps=7)
                    pp = predict(X[test] + U, theta, G)
                    mm = metric(y[test], pp)
                    rec[f"deg_mse_{rho_eval}"] = mm["mse"] - clean["mse"]
                    rec[f"deg_da_{rho_eval}"] = clean["da"] - mm["da"]
                    rec[f"pert_mse_{rho_eval}"] = mm["mse"]
                rows.append(rec)
                tau_rows.append(rec)

        best = min(tau_rows, key=lambda r: r["val_mse"])
        best_by_tau.append(best)

        erm = [r for r in tau_rows if r["rho_x_train"] == 0][0]
        robust = min([r for r in tau_rows if r["rho_x_train"] > 0], key=lambda r: r["deg_mse_0.1"])
        improvement_rows.append({
            "tau": tau,
            "erm_clean_mse": erm["clean_mse"],
            "robust_clean_mse": robust["clean_mse"],
            "erm_deg_mse_0.1": erm["deg_mse_0.1"],
            "robust_deg_mse_0.1": robust["deg_mse_0.1"],
            "deg_mse_improvement": erm["deg_mse_0.1"] - robust["deg_mse_0.1"],
            "robust_choice": f"rho={robust['rho_x_train']},R={robust['r_adv']}",
        })

    finite = []
    for r in rows:
        for k,v in r.items():
            if isinstance(v, float):
                finite.append(np.isfinite(v))
    degs = [r["deg_mse_0.1"] for r in rows]
    nonzero_deg = [abs(x) > 1e-14 for x in degs]

    add("results_nonempty", len(rows) > 0, len(rows), ">0")
    add("metrics_finite", bool(np.all(finite)), f"{sum(finite)}/{len(finite)} finite", "all finite")
    add("calibrated_degradation_nonzero", np.mean(nonzero_deg) >= 0.75, f"{np.mean(nonzero_deg):.2%}", ">=75% nonzero")
    add("degradation_nonnegative_mostly", np.mean(np.array(degs) >= -1e-14) >= 0.90, f"{np.mean(np.array(degs) >= -1e-14):.2%}", ">=90%")

    status = "PASS" if all(c.status == "PASS" for c in checks) else "FAIL"

    cols = ["tau","rho_x_train","r_adv","val_mse","clean_mse","clean_da","clean_sharpe",
            "deg_mse_0.01","deg_mse_0.05","deg_mse_0.1","deg_mse_0.2","deg_da_0.1"]
    with open(processed/"feature_robust_calibrated_eval_step04b.csv","w",encoding="utf-8") as f:
        f.write(",".join(cols)+"\n")
        for r in rows:
            f.write(",".join(str(r.get(c,"")) for c in cols)+"\n")

    summary = {
        "status": status, "rho_eval_grid": RHO_EVAL_GRID, "rho_train_grid": RHO_TRAIN_GRID,
        "rows": rows, "best_by_tau": best_by_tau, "improvement_rows": improvement_rows,
        "checks": [c.__dict__ for c in checks]
    }
    with open(reports/"step04b_feature_robust_calibrated_eval.json","w",encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md = []
    md.append("# Step 04b Calibrated Feature-Adversarial Evaluation\n")
    md.append(f"Overall status: **{status}**\n")
    md.append("## Purpose")
    md.append("This module recalibrates feature adversarial evaluation by using normalized steepest-ascent directions on the standardized feature tensor. It is designed to avoid the zero-degradation artifact caused by tiny raw gradients.\n")
    md.append("## Best validation-selected model by horizon")
    md.append(md_table(best_by_tau, ["tau","rho_x_train","r_adv","val_mse","clean_mse","clean_da","deg_mse_0.1","deg_da_0.1"]))
    md.append("")
    md.append("## ERM vs best robust degradation comparison")
    md.append(md_table(improvement_rows, ["tau","erm_clean_mse","robust_clean_mse","erm_deg_mse_0.1","robust_deg_mse_0.1","deg_mse_improvement","robust_choice"]))
    md.append("")
    md.append("## All calibrated degradation results")
    md.append(md_table(rows, cols))
    md.append("")
    md.append("## Benchmark checks")
    md.append("| check | status | observed | expected |")
    md.append("|---|---|---|---|")
    for c in checks:
        md.append(f"| {c.name} | {c.status} | `{c.observed}` | `{c.expected}` |")
    md.append("")
    if status == "PASS":
        (state/"step04b_calibrated_eval.PASS").write_text("PASS\n",encoding="utf-8")
        md.append("Gate passed: `.pipeline_state/step04b_calibrated_eval.PASS` should exist.")
        md.append("Next module should be `05_graph_robust_pgda.sh`.")
    else:
        (state/"step04b_calibrated_eval.FAIL").write_text("FAIL\n",encoding="utf-8")
        md.append("Gate failed: inspect `reports/benchmarks/step04b_feature_robust_calibrated_eval.json`.")

    text = "\n".join(md)
    (reports/"step04b_feature_robust_calibrated_eval.md").write_text(text,encoding="utf-8")
    print(text)

if __name__ == "__main__":
    main()

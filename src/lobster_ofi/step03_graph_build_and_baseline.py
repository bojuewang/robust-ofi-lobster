from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.metrics import mean_squared_error
except Exception as exc:
    raise SystemExit(
        "[ERROR] scikit-learn is required for Step 03. Install with: pip install scikit-learn\n"
        f"Original error: {exc}"
    )

EPS = 1e-12
T_EXPECTED = 46800
N_EXPECTED = 5
H_EXPECTED = 10
D_EXPECTED = 51
TRAIN_END = 32760
VAL_END = 39780
TAUS = [1, 2, 10, 20]
TICKERS = ["AMZN", "AAPL", "GOOG", "INTC", "MSFT"]


@dataclass
class Check:
    name: str
    status: str
    observed: str
    expected: str


def corr_matrix(X: np.ndarray) -> np.ndarray:
    """Column correlation with robust handling of constant columns."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"corr_matrix expects 2D array, got {X.shape}")
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    std = np.nanstd(Xc, axis=0, keepdims=True)
    good = (std.reshape(-1) > 1e-10)
    Z = np.zeros_like(Xc)
    Z[:, good] = Xc[:, good] / (std[:, good] + EPS)
    denom = max(X.shape[0] - 1, 1)
    C = (Z.T @ Z) / denom
    C[~np.isfinite(C)] = 0.0
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 0.0)
    return C


def sym_clip(C: np.ndarray, signed: bool = True) -> np.ndarray:
    A = 0.5 * (C + C.T)
    if signed:
        A = np.clip(A, -1.0, 1.0)
    else:
        A = np.abs(A)
        A = np.clip(A, 0.0, 1.0)
    np.fill_diagonal(A, 0.0)
    A[~np.isfinite(A)] = 0.0
    return A.astype(np.float32)


def row_normalize_graph(G: np.ndarray) -> np.ndarray:
    """Normalize graph rows by L1 mass for stable graph-linear regression."""
    G = np.asarray(G, dtype=np.float64)
    denom = np.sum(np.abs(G), axis=1, keepdims=True)
    return (G / (denom + EPS)).astype(np.float32)


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
    mask_key = find_key(keys, ["event_mask", "chi_evt", "mask", "event_activity_mask"], ["mask"])

    if xstd_key is None:
        raise SystemExit(f"[ERROR] Could not find standardized feature tensor key in NPZ. Keys: {keys}")
    X_std = np.asarray(z[xstd_key], dtype=np.float32)
    X_raw = np.asarray(z[xraw_key], dtype=np.float32) if xraw_key else None

    if mask_key is not None:
        event_mask = np.asarray(z[mask_key])
    else:
        # Proposal feature order appends chi_evt as the final scalar feature.
        event_mask = (X_raw[:, :, -1] if X_raw is not None else X_std[:, :, -1]) > 0.5
    event_mask = event_mask.astype(np.float32)

    # Load targets robustly.
    y_logret: Dict[int, np.ndarray] = {}
    y_spread: Dict[int, np.ndarray] = {}
    for tau in TAUS:
        candidates_log = [
            f"y_logret_tau_{tau}",
            f"logret_tau_{tau}",
            f"Y_logret_tau_{tau}",
            f"target_logret_tau_{tau}",
            f"y_tau_{tau}",
        ]
        candidates_spread = [
            f"y_spreadret_tau_{tau}",
            f"spreadret_tau_{tau}",
            f"Y_spreadret_tau_{tau}",
            f"target_spreadret_tau_{tau}",
            f"z_tau_{tau}",
        ]
        klog = find_key(keys, candidates_log, ["log", str(tau)])
        kspr = find_key(keys, candidates_spread, ["spread", str(tau)])
        if klog:
            y_logret[tau] = np.asarray(z[klog], dtype=np.float32)
        if kspr:
            y_spread[tau] = np.asarray(z[kspr], dtype=np.float32)

    # If tau=1 target not found, try mid_price.
    if not y_logret:
        mid_key = find_key(keys, ["mid_price", "mid", "m"], ["mid"])
        if mid_key:
            mid = np.asarray(z[mid_key], dtype=np.float64)
            for tau in TAUS:
                y = np.full_like(mid, np.nan, dtype=np.float32)
                y[:-tau] = np.log(mid[tau:] + EPS) - np.log(mid[:-tau] + EPS)
                y_logret[tau] = y
        else:
            raise SystemExit(f"[ERROR] Could not find target arrays or mid_price in NPZ. Keys: {keys}")

    return z, keys, X_std, X_raw, event_mask, y_logret, y_spread, xstd_key, xraw_key, mask_key


def train_val_test_indices(tau: int):
    train = np.arange(0, TRAIN_END - tau)
    val = np.arange(TRAIN_END, VAL_END - tau)
    test = np.arange(VAL_END, T_EXPECTED - tau)
    return train, val, test


def build_graphs(X_std: np.ndarray, event_mask: np.ndarray, y_logret: Dict[int, np.ndarray], tau: int = 1):
    # Proposal order: first H features are share-based OFI Phi^H.
    phi_bar = X_std[:, :, :H_EXPECTED].mean(axis=2).astype(np.float64)
    y1 = np.asarray(y_logret[1], dtype=np.float64)

    train, _, _ = train_val_test_indices(tau)
    train_lag = np.arange(0, TRAIN_END - tau)

    # Return graph from past one-bin log-return.
    valid_ret = np.all(np.isfinite(y1[train]), axis=1)
    C_ret = corr_matrix(y1[train][valid_ret])

    # OFI graph from integrated OFI.
    valid_phi = np.all(np.isfinite(phi_bar[train]), axis=1)
    C_ofi = corr_matrix(phi_bar[train][valid_phi])

    # Lagged cross-OFI graph: corr(phi_u,i, y_{u,j}^{tau}) on training only.
    ytau = np.asarray(y_logret[tau], dtype=np.float64)
    valid_cross = np.all(np.isfinite(phi_bar[train_lag]), axis=1) & np.all(np.isfinite(ytau[train_lag]), axis=1)
    Phi = phi_bar[train_lag][valid_cross]
    Y = ytau[train_lag][valid_cross]
    Phi_c = Phi - Phi.mean(axis=0, keepdims=True)
    Y_c = Y - Y.mean(axis=0, keepdims=True)
    Phi_s = Phi_c.std(axis=0, keepdims=True)
    Y_s = Y_c.std(axis=0, keepdims=True)
    Zp = Phi_c / (Phi_s + EPS)
    Zy = Y_c / (Y_s + EPS)
    C_cross = (Zp.T @ Zy) / max(Phi.shape[0] - 1, 1)
    C_cross[~np.isfinite(C_cross)] = 0.0
    C_cross = np.clip(C_cross, -1.0, 1.0)
    np.fill_diagonal(C_cross, 0.0)

    # Rolling coactivity on training period for fixed mask-aware graph.
    coact = (event_mask[:TRAIN_END].T @ event_mask[:TRAIN_END]) / float(TRAIN_END)
    coact = np.asarray(coact, dtype=np.float64)
    np.fill_diagonal(coact, 0.0)
    coact = np.clip(coact, 0.0, 1.0)

    graphs = {}
    for name, C in [
        ("return", C_ret),
        ("ofi", C_ofi),
        ("cross_ofi", 0.5 * (C_cross + C_cross.T)),
    ]:
        G_signed = sym_clip(C, signed=True)
        G_abs = sym_clip(C, signed=False)
        graphs[f"{name}_signed"] = G_signed
        graphs[f"{name}_abs"] = G_abs
        graphs[f"{name}_signed_mask"] = (G_signed * coact).astype(np.float32)
        graphs[f"{name}_abs_mask"] = (G_abs * coact).astype(np.float32)

    return graphs, coact.astype(np.float32), phi_bar.astype(np.float32)


def flatten_own_asset(X: np.ndarray, y: np.ndarray, idx: np.ndarray):
    # X: T,N,d, y:T,N -> rows T*N
    Xb = X[idx].reshape(len(idx) * X.shape[1], X.shape[2])
    yb = y[idx].reshape(len(idx) * X.shape[1])
    mask = np.isfinite(yb) & np.all(np.isfinite(Xb), axis=1)
    return Xb[mask], yb[mask]


def make_graph_linear_design(X: np.ndarray, G: np.ndarray, idx: np.ndarray):
    Gnorm = row_normalize_graph(G)
    X_self = X[idx]  # B,N,d
    X_nei = np.einsum("ij,tjd->tid", Gnorm, X_self)
    D = np.concatenate([X_self, X_nei, np.ones((*X_self.shape[:2], 1), dtype=X_self.dtype)], axis=2)
    return D.reshape(len(idx) * X.shape[1], 2 * X.shape[2] + 1)


def eval_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    mse = float(np.mean((yp - yt) ** 2))
    mae = float(np.mean(np.abs(yp - yt)))
    # Directional accuracy: ignore exact zero signs if any.
    nonzero = np.abs(yt) > 0
    if np.any(nonzero):
        da = float(np.mean(np.sign(yp[nonzero]) == np.sign(yt[nonzero])))
    else:
        da = float("nan")
    denom = np.sum(np.abs(yp.reshape(-1, N_EXPECTED)), axis=1, keepdims=True) + EPS
    w = yp.reshape(-1, N_EXPECTED) / denom
    yy = yt.reshape(-1, N_EXPECTED)
    pnl = np.sum(w * yy, axis=1)
    sharpe = float(np.mean(pnl) / (np.std(pnl) + EPS))
    return {"mse": mse, "mae": mae, "directional_accuracy": da, "sharpe_proxy_no_tc": sharpe}


def run_models(X_std: np.ndarray, y_logret: Dict[int, np.ndarray], graphs: Dict[str, np.ndarray], tau: int):
    y = y_logret[tau]
    train, val, test = train_val_test_indices(tau)

    results = []
    checks = []

    # Baseline 0: own-feature Ridge and Lasso.
    Xtr, ytr = flatten_own_asset(X_std, y, train)
    Xva, yva = flatten_own_asset(X_std, y, val)
    Xte, yte = flatten_own_asset(X_std, y, test)

    alpha_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best = None
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha, fit_intercept=True, random_state=0)
        model.fit(Xtr, ytr)
        pred_va = model.predict(Xva)
        va = eval_predictions(yva.reshape(-1, N_EXPECTED), pred_va.reshape(-1, N_EXPECTED))
        rec = ("ridge_own", alpha, va["mse"], model)
        if best is None or rec[2] < best[2]:
            best = rec
    assert best is not None
    name, alpha, val_mse, model = best
    pred_te = model.predict(Xte)
    metrics = eval_predictions(yte.reshape(-1, N_EXPECTED), pred_te.reshape(-1, N_EXPECTED))
    results.append({"model": name, "graph": "none", "tau": tau, "alpha": alpha, "val_mse": val_mse, **metrics})

    best_lasso = None
    for alpha in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
        model_l = Lasso(alpha=alpha, fit_intercept=True, max_iter=5000, random_state=0)
        model_l.fit(Xtr, ytr)
        pred_va = model_l.predict(Xva)
        va = eval_predictions(yva.reshape(-1, N_EXPECTED), pred_va.reshape(-1, N_EXPECTED))
        rec = ("lasso_own", alpha, va["mse"], model_l)
        if best_lasso is None or rec[2] < best_lasso[2]:
            best_lasso = rec
    name, alpha, val_mse, model_l = best_lasso
    pred_te = model_l.predict(Xte)
    metrics = eval_predictions(yte.reshape(-1, N_EXPECTED), pred_te.reshape(-1, N_EXPECTED))
    results.append({"model": name, "graph": "none", "tau": tau, "alpha": alpha, "val_mse": val_mse, **metrics})

    # Baseline 2: graph-linear Ridge on selected graphs.
    for gname in ["return_abs", "ofi_abs", "cross_ofi_abs", "return_abs_mask", "ofi_abs_mask", "cross_ofi_abs_mask"]:
        G = graphs[gname]
        Dtr = make_graph_linear_design(X_std, G, train)
        Dva = make_graph_linear_design(X_std, G, val)
        Dte = make_graph_linear_design(X_std, G, test)
        ytr_flat = y[train].reshape(len(train) * N_EXPECTED)
        yva_flat = y[val].reshape(len(val) * N_EXPECTED)
        yte_flat = y[test].reshape(len(test) * N_EXPECTED)

        mtr = np.isfinite(ytr_flat) & np.all(np.isfinite(Dtr), axis=1)
        mva = np.isfinite(yva_flat) & np.all(np.isfinite(Dva), axis=1)
        mte = np.isfinite(yte_flat) & np.all(np.isfinite(Dte), axis=1)

        best_g = None
        for alpha in alpha_grid:
            model_g = Ridge(alpha=alpha, fit_intercept=False, random_state=0)
            model_g.fit(Dtr[mtr], ytr_flat[mtr])
            pred_va = model_g.predict(Dva[mva])
            va = eval_predictions(yva_flat[mva].reshape(-1, N_EXPECTED), pred_va.reshape(-1, N_EXPECTED))
            rec = (alpha, va["mse"], model_g)
            if best_g is None or rec[1] < best_g[1]:
                best_g = rec
        alpha, val_mse, model_g = best_g
        pred_te = model_g.predict(Dte[mte])
        metrics = eval_predictions(yte_flat[mte].reshape(-1, N_EXPECTED), pred_te.reshape(-1, N_EXPECTED))
        results.append({"model": "graph_linear_ridge", "graph": gname, "tau": tau, "alpha": alpha, "val_mse": val_mse, **metrics})

    return results


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
    npz_path = processed / "features_500ms_step02.npz"

    z, keys, X_std, X_raw, event_mask, y_logret, y_spread, xstd_key, xraw_key, mask_key = load_step02(npz_path)

    checks: List[Check] = []

    def add(name, ok, observed, expected):
        checks.append(Check(name, "PASS" if ok else "FAIL", str(observed), str(expected)))

    add("step02_gate_exists", (state / "step02_feature_build.PASS").exists(), ".pipeline_state/step02_feature_build.PASS", "exists")
    add("xstd_key_found", xstd_key is not None, xstd_key, "standardized feature tensor key")
    add("X_std_shape", tuple(X_std.shape) == (T_EXPECTED, N_EXPECTED, D_EXPECTED), tuple(X_std.shape), (T_EXPECTED, N_EXPECTED, D_EXPECTED))
    add("event_mask_shape", tuple(event_mask.shape) == (T_EXPECTED, N_EXPECTED), tuple(event_mask.shape), (T_EXPECTED, N_EXPECTED))
    add("X_std_finite", bool(np.all(np.isfinite(X_std))), bool(np.all(np.isfinite(X_std))), True)
    add("event_mask_binary", bool(np.all((event_mask == 0) | (event_mask == 1))), np.unique(event_mask).tolist()[:5], "{0,1}")
    add("tau1_target_exists", 1 in y_logret, sorted(y_logret.keys()), "tau=1 target available")
    for tau in TAUS:
        add(f"tau_{tau}_target_exists", tau in y_logret, sorted(y_logret.keys()), f"tau={tau}")
        if tau in y_logret:
            add(f"tau_{tau}_target_shape", tuple(y_logret[tau].shape) == (T_EXPECTED, N_EXPECTED), tuple(y_logret[tau].shape), (T_EXPECTED, N_EXPECTED))

    graphs, coact, phi_bar = build_graphs(X_std, event_mask, y_logret, tau=1)
    np.savez_compressed(
        processed / "graphs_step03.npz",
        coactivity_train=coact,
        phi_bar=phi_bar,
        **graphs,
    )

    graph_summaries = []
    for name, G in graphs.items():
        add(f"graph_{name}_shape", tuple(G.shape) == (N_EXPECTED, N_EXPECTED), tuple(G.shape), (N_EXPECTED, N_EXPECTED))
        add(f"graph_{name}_finite", bool(np.all(np.isfinite(G))), bool(np.all(np.isfinite(G))), True)
        add(f"graph_{name}_diag_zero", float(np.max(np.abs(np.diag(G)))) < 1e-7, float(np.max(np.abs(np.diag(G)))), "0")
        graph_summaries.append({
            "graph": name,
            "min": float(np.min(G)),
            "max": float(np.max(G)),
            "mean_abs": float(np.mean(np.abs(G))),
            "nonzero_edges": int(np.sum(np.abs(G) > 1e-10)),
        })

    # Run baselines for all taus.
    all_results = []
    for tau in TAUS:
        all_results.extend(run_models(X_std, y_logret, graphs, tau=tau))

    # Determine sanity checks.
    mses = [r["mse"] for r in all_results if np.isfinite(r["mse"])]
    das = [r["directional_accuracy"] for r in all_results if np.isfinite(r["directional_accuracy"])]
    add("baseline_results_nonempty", len(all_results) > 0, len(all_results), "> 0")
    add("baseline_mse_finite", bool(len(mses) > 0 and np.all(np.isfinite(mses))), f"count={len(mses)}", "all finite")
    add("baseline_da_range", bool(len(das) > 0 and min(das) >= 0.0 and max(das) <= 1.0), f"min={min(das):.4f}, max={max(das):.4f}", "[0,1]")

    # Best rows by tau.
    best_by_tau = []
    for tau in TAUS:
        rows = [r for r in all_results if r["tau"] == tau]
        best = min(rows, key=lambda r: r["val_mse"])
        best_by_tau.append(best)

    # Save artifacts.
    def json_default(o):
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(type(o).__name__)

    summary = {
        "status": "PASS" if all(c.status == "PASS" for c in checks) else "FAIL",
        "npz_keys": keys,
        "graphs": graph_summaries,
        "results": all_results,
        "best_by_tau": best_by_tau,
        "checks": [c.__dict__ for c in checks],
    }
    with open(reports / "step03_graph_build_and_baseline.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default)

    # Also save CSV-like results without pandas.
    cols = ["model", "graph", "tau", "alpha", "val_mse", "mse", "mae", "directional_accuracy", "sharpe_proxy_no_tc"]
    with open(processed / "baseline_results_step03.csv", "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in all_results:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    status = summary["status"]
    md = []
    md.append("# Step 03 Graph Build + ERM Baseline Benchmark\n")
    md.append(f"Overall status: **{status}**\n")
    md.append("## Input contract")
    md.append(f"- Feature NPZ: `data/processed/features_500ms_step02.npz`")
    md.append(f"- NPZ standardized key: `{xstd_key}`")
    md.append(f"- Raw feature tensor: `{None if X_raw is None else X_raw.shape}`")
    md.append(f"- Standardized feature tensor: `{X_std.shape}`")
    md.append(f"- Event mask: `{event_mask.shape}`")
    md.append(f"- Tickers: `{', '.join(TICKERS)}`")
    md.append(f"- Time bins/assets/features: `{T_EXPECTED} / {N_EXPECTED} / {D_EXPECTED}`\n")

    md.append("## Graph summaries")
    md.append(markdown_table(graph_summaries, ["graph", "min", "max", "mean_abs", "nonzero_edges"]))
    md.append("")

    md.append("## Best validation-selected model by horizon")
    md.append(markdown_table(best_by_tau, ["tau", "model", "graph", "alpha", "val_mse", "mse", "mae", "directional_accuracy", "sharpe_proxy_no_tc"]))
    md.append("")

    md.append("## All baseline results")
    md.append(markdown_table(all_results, ["tau", "model", "graph", "alpha", "val_mse", "mse", "mae", "directional_accuracy", "sharpe_proxy_no_tc"]))
    md.append("")

    md.append("## Benchmark checks")
    md.append("| check | status | observed | expected |")
    md.append("|---|---|---|---|")
    for c in checks:
        md.append(f"| {c.name} | {c.status} | `{c.observed}` | `{c.expected}` |")
    md.append("")

    if status == "PASS":
        (state / "step03_graph_baseline.PASS").write_text("PASS\n", encoding="utf-8")
        md.append("Gate passed: `.pipeline_state/step03_graph_baseline.PASS` should exist.")
        md.append("Next module should be `04_feature_robust_pgda.sh`, but do not run it until this report is reviewed.")
    else:
        fail_path = state / "step03_graph_baseline.FAIL"
        fail_path.write_text("FAIL\n", encoding="utf-8")
        md.append("Gate failed: inspect `reports/benchmarks/step03_graph_build_and_baseline.json`.")

    (reports / "step03_graph_build_and_baseline.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path.cwd()
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "benchmarks"
STATE_DIR = ROOT / ".pipeline_state"

TICKERS = ["AMZN", "AAPL", "GOOG", "INTC", "MSFT"]
EXPECTED_EVENT_BINS = {
    "AMZN": 29801,
    "AAPL": 34800,
    "GOOG": 22678,
    "INTC": 36086,
    "MSFT": 37005,
}
EXPECTED_ANY_ASSET_EVENT_BINS = 46290

START_SEC = 34200.0
END_SEC = 57600.0
BIN_WIDTH = 0.5
T = int(round((END_SEC - START_SEC) / BIN_WIDTH))
N = len(TICKERS)
H = 10
D = 4 * H + 11
TRAIN_END = 32760
VAL_END = 39780
TEST_END = 46800
TAUS = [1, 2, 10, 20]
EPS = 1e-8

MSG_COLS = ["time", "event_type", "order_id", "size", "price", "direction"]
EVENT_VALID = {1, 2, 3, 4, 5}
EVENT_SUB = {1}
EVENT_CAN = {2, 3}
EVENT_EXE = {4, 5}

FEATURE_NAMES: List[str] = (
    [f"ofi_share_h{h}" for h in range(1, H + 1)]
    + [f"ofi_dollar_h{h}" for h in range(1, H + 1)]
    + [f"obi_share_h{h}" for h in range(1, H + 1)]
    + [f"obi_dollar_h{h}" for h in range(1, H + 1)]
    + ["spread", "weighted_mid"]
    + [
        "V_execution_volume",
        "v_execution_traded_value",
        "deltaP_submission",
        "Q_submission_signed",
        "deltaP_cancellation",
        "Q_cancellation_signed",
        "deltaP_execution",
        "Q_execution_signed",
        "event_activity_mask",
    ]
)

assert len(FEATURE_NAMES) == D


@dataclass
class TickerBuildResult:
    ticker: str
    X_raw: np.ndarray
    mid: np.ndarray
    spread: np.ndarray
    weighted_mid: np.ndarray
    event_mask: np.ndarray
    summary: Dict[str, object]


class Gate:
    def __init__(self) -> None:
        self.rows: List[Dict[str, object]] = []

    def add(self, name: str, ok: bool, observed: object, expected: object) -> None:
        self.rows.append(
            {
                "check": name,
                "status": "PASS" if bool(ok) else "FAIL",
                "observed": observed,
                "expected": expected,
            }
        )

    @property
    def passed(self) -> bool:
        return all(row["status"] == "PASS" for row in self.rows)


def find_zip(ticker: str) -> Path:
    matches = sorted(RAW_DIR.glob(f"*{ticker}*2012-06-21*10*.zip"))
    if not matches:
        raise FileNotFoundError(f"No LOBSTER zip found for {ticker} in {RAW_DIR}")
    # Prefer exact non-duplicated name when available, but any matching zip is acceptable.
    return matches[0]


def zip_members(zip_path: Path) -> Tuple[str, str]:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    msg = [n for n in names if "message" in n.lower() and n.lower().endswith(".csv")]
    ob = [n for n in names if "orderbook" in n.lower() and n.lower().endswith(".csv")]
    if not msg or not ob:
        raise ValueError(f"Cannot find message/orderbook CSV in {zip_path}")
    return msg[0], ob[0]


def read_message(zip_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            df = pd.read_csv(
                fh,
                header=None,
                names=MSG_COLS,
                dtype={
                    "time": "float64",
                    "event_type": "int16",
                    "order_id": "int64",
                    "size": "float64",
                    "price": "float64",
                    "direction": "int16",
                },
            )
    return df


def read_orderbook_selected(zip_path: Path, member: str, selected_indices: np.ndarray) -> Tuple[np.ndarray, int]:
    """Read only selected orderbook rows from the zipped CSV.

    LOBSTER orderbook files are event-level and can be large. For 500ms binning we
    only need the last orderbook row in each bin after carry-forward. This function
    streams through the CSV and parses only those selected rows, which is much faster
    and lighter than loading every event-level orderbook row into memory.
    """
    selected = np.asarray(selected_indices, dtype=np.int64)
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError("selected_indices must be a nonempty one-dimensional array")
    order = np.argsort(selected)
    selected_sorted = selected[order]
    if np.any(np.diff(selected_sorted) == 0):
        raise ValueError("selected_indices should be unique before calling read_orderbook_selected")

    rows_sorted = np.empty((len(selected_sorted), 4 * H), dtype=np.float64)
    target_pos = 0
    line_count = 0
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            for line_no, raw in enumerate(fh):
                line_count = line_no + 1
                if target_pos >= len(selected_sorted):
                    # Continue no further; all requested rows have already been read.
                    break
                target = int(selected_sorted[target_pos])
                if line_no < target:
                    continue
                if line_no == target:
                    vals = np.fromstring(raw.decode("ascii"), sep=",", dtype=np.float64)
                    if vals.shape[0] != 4 * H:
                        raise ValueError(
                            f"{zip_path.name}:{member}: row {line_no} has {vals.shape[0]} columns; expected {4 * H}"
                        )
                    rows_sorted[target_pos] = vals
                    target_pos += 1
                elif line_no > target:
                    raise RuntimeError("Internal selected-row streaming error")

    if target_pos != len(selected_sorted):
        raise ValueError(
            f"Only read {target_pos} selected orderbook rows out of {len(selected_sorted)} requested from {zip_path}"
        )
    rows = np.empty_like(rows_sorted)
    rows[order] = rows_sorted
    return rows, line_count


def to_bin_index(times: np.ndarray) -> np.ndarray:
    # np.floor is intentional because bins are half-open [start+0.5(t-1), start+0.5t).
    return np.floor((times - START_SEC) / BIN_WIDTH).astype(np.int64)


def forward_fill_last_indices(bin_idx: np.ndarray, original_row_idx: np.ndarray) -> np.ndarray:
    last = np.full(T, -1, dtype=np.int64)
    if len(bin_idx) > 0:
        np.maximum.at(last, bin_idx, original_row_idx)
    if np.all(last < 0):
        raise ValueError("No orderbook rows fall inside the requested time window.")
    filled = np.maximum.accumulate(last)
    if filled[0] < 0:
        first = int(np.flatnonzero(last >= 0)[0])
        filled[:first] = last[first]
    return filled


def aggregate_event_class(
    msg: pd.DataFrame,
    mid: np.ndarray,
    event_set: set[int],
    sign_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return class size sum, relative size-weighted price, and signed queue."""
    event_type = msg["event_type"].to_numpy()
    time = msg["time"].to_numpy()
    size = msg["size"].to_numpy(dtype=np.float64)
    price = msg["price"].to_numpy(dtype=np.float64) / 10000.0
    direction = msg["direction"].to_numpy(dtype=np.float64)

    keep = (
        (time >= START_SEC)
        & (time < END_SEC)
        & np.isin(event_type, list(event_set))
        & np.isfinite(size)
        & np.isfinite(price)
        & (size >= 0)
    )
    bins = to_bin_index(time[keep])
    valid = (bins >= 0) & (bins < T)
    bins = bins[valid]
    size_k = size[keep][valid]
    price_k = price[keep][valid]
    direction_k = direction[keep][valid]

    size_sum = np.zeros(T, dtype=np.float64)
    px_size_sum = np.zeros(T, dtype=np.float64)
    signed_q = np.zeros(T, dtype=np.float64)

    if sign_mode == "sub":
        sigma = direction_k
    elif sign_mode in {"can", "exe"}:
        sigma = -direction_k
    else:
        raise ValueError(f"Unknown sign mode {sign_mode}")

    if len(bins):
        np.add.at(size_sum, bins, size_k)
        np.add.at(px_size_sum, bins, price_k * size_k)
        np.add.at(signed_q, bins, sigma * size_k)

    delta_price = np.zeros(T, dtype=np.float64)
    has = size_sum > 0
    delta_price[has] = px_size_sum[has] / (size_sum[has] + EPS) - mid[has]
    return size_sum, delta_price, signed_q


def build_ticker(ticker: str) -> TickerBuildResult:
    zip_path = find_zip(ticker)
    msg_member, ob_member = zip_members(zip_path)
    print(f"[INFO] {ticker}: reading message file {msg_member}", flush=True)
    msg = read_message(zip_path, msg_member)
    time_all = msg["time"].to_numpy(dtype=np.float64)
    in_window = (time_all >= START_SEC) & (time_all < END_SEC)
    bins_all = to_bin_index(time_all[in_window])
    original_idx = np.flatnonzero(in_window).astype(np.int64)
    valid_bins = (bins_all >= 0) & (bins_all < T)
    bins_all = bins_all[valid_bins]
    original_idx = original_idx[valid_bins]
    last_idx = forward_fill_last_indices(bins_all, original_idx)

    unique_idx, inverse_idx = np.unique(last_idx, return_inverse=True)
    print(
        f"[INFO] {ticker}: streaming {len(unique_idx)} selected orderbook rows from {ob_member}",
        flush=True,
    )
    ob_unique, orderbook_rows_seen = read_orderbook_selected(zip_path, ob_member, unique_idx)
    if orderbook_rows_seen != len(msg):
        raise ValueError(f"{ticker}: message rows {len(msg)} != orderbook rows seen {orderbook_rows_seen}")
    ob_bin = ob_unique[inverse_idx]

    ask_p = ob_bin[:, 0::4] / 10000.0
    ask_q = ob_bin[:, 1::4]
    bid_p = ob_bin[:, 2::4] / 10000.0
    bid_q = ob_bin[:, 3::4]

    best_valid = (
        np.isfinite(ask_p[:, 0])
        & np.isfinite(bid_p[:, 0])
        & np.isfinite(ask_q[:, 0])
        & np.isfinite(bid_q[:, 0])
        & (ask_p[:, 0] > 0)
        & (bid_p[:, 0] > 0)
        & (ask_q[:, 0] > 0)
        & (bid_q[:, 0] > 0)
        & (ask_p[:, 0] > bid_p[:, 0])
    )
    if not np.all(best_valid):
        bad = int((~best_valid).sum())
        raise ValueError(f"{ticker}: {bad} binned best-level states are invalid or crossed/locked.")

    valid_level = (
        np.isfinite(ask_p)
        & np.isfinite(bid_p)
        & np.isfinite(ask_q)
        & np.isfinite(bid_q)
        & (ask_p > 0)
        & (bid_p > 0)
        & (ask_q > 0)
        & (bid_q > 0)
        & (ask_p > bid_p)
    )
    pa = np.where(valid_level, ask_p, 0.0)
    pb = np.where(valid_level, bid_p, 0.0)
    qa = np.where(valid_level, ask_q, 0.0)
    qb = np.where(valid_level, bid_q, 0.0)

    mid = ((ask_p[:, 0] + bid_p[:, 0]) / 2.0).astype(np.float64)
    spread = (ask_p[:, 0] - bid_p[:, 0]).astype(np.float64)
    weighted_mid = (
        (ask_p[:, 0] * bid_q[:, 0] + bid_p[:, 0] * ask_q[:, 0])
        / (ask_q[:, 0] + bid_q[:, 0] + EPS)
    ).astype(np.float64)

    # Event-activity mask: valid message event types 1--5 only.
    event_type = msg["event_type"].to_numpy()
    event_keep = in_window & np.isin(event_type, list(EVENT_VALID))
    event_bins = to_bin_index(time_all[event_keep])
    event_bins = event_bins[(event_bins >= 0) & (event_bins < T)]
    event_mask = np.zeros(T, dtype=np.uint8)
    if len(event_bins):
        event_mask[np.unique(event_bins)] = 1

    # Message-flow features: V, v, deltaP/Q for submission/cancellation/execution.
    V_exe, deltaP_exe, Q_exe = aggregate_event_class(msg, mid, EVENT_EXE, "exe")
    V_sub, deltaP_sub, Q_sub = aggregate_event_class(msg, mid, EVENT_SUB, "sub")
    V_can, deltaP_can, Q_can = aggregate_event_class(msg, mid, EVENT_CAN, "can")
    # Traded value uses execution events only.
    exe_keep = in_window & np.isin(event_type, list(EVENT_EXE))
    exe_bins = to_bin_index(time_all[exe_keep])
    exe_valid = (exe_bins >= 0) & (exe_bins < T)
    exe_bins = exe_bins[exe_valid]
    exe_size = msg.loc[exe_keep, "size"].to_numpy(dtype=np.float64)[exe_valid]
    exe_price = msg.loc[exe_keep, "price"].to_numpy(dtype=np.float64)[exe_valid] / 10000.0
    traded_value = np.zeros(T, dtype=np.float64)
    if len(exe_bins):
        np.add.at(traded_value, exe_bins, exe_size * exe_price)

    psi = np.column_stack(
        [
            V_exe,
            traded_value,
            deltaP_sub,
            Q_sub,
            deltaP_can,
            Q_can,
            deltaP_exe,
            Q_exe,
        ]
    )

    # Order-book imbalance, cumulative over levels.
    obi_num = np.cumsum(qb - qa, axis=1)
    obi_den = np.cumsum(qb + qa, axis=1) + EPS
    obi_share = obi_num / obi_den

    bid_dollar = pb * qb
    ask_dollar = pa * qa
    obi_d_num = np.cumsum(bid_dollar - ask_dollar, axis=1)
    obi_d_den = np.cumsum(bid_dollar + ask_dollar, axis=1) + EPS
    obi_dollar = obi_d_num / obi_d_den

    # Order-flow imbalance from binned carry-forward states.
    ofi_b = np.zeros((T, H), dtype=np.float64)
    ofi_a = np.zeros((T, H), dtype=np.float64)
    ofi_db = np.zeros((T, H), dtype=np.float64)
    ofi_da = np.zeros((T, H), dtype=np.float64)
    pair_valid = valid_level[1:] & valid_level[:-1]

    bid_gt = bid_p[1:] > bid_p[:-1]
    bid_eq = bid_p[1:] == bid_p[:-1]
    bid_lt = bid_p[1:] < bid_p[:-1]
    ask_lt = ask_p[1:] < ask_p[:-1]
    ask_eq = ask_p[1:] == ask_p[:-1]
    ask_gt = ask_p[1:] > ask_p[:-1]

    ofi_b[1:] = (
        bid_gt * bid_q[1:] + bid_eq * (bid_q[1:] - bid_q[:-1]) - bid_lt * bid_q[:-1]
    ) * pair_valid
    ofi_a[1:] = (
        ask_lt * ask_q[1:] + ask_eq * (ask_q[:-1] - ask_q[1:]) - ask_gt * ask_q[:-1]
    ) * pair_valid

    bd = bid_p * bid_q
    ad = ask_p * ask_q
    ofi_db[1:] = (
        bid_gt * bd[1:] + bid_eq * (bd[1:] - bd[:-1]) - bid_lt * bd[:-1]
    ) * pair_valid
    ofi_da[1:] = (
        ask_lt * ad[1:] + ask_eq * (ad[:-1] - ad[1:]) - ask_gt * ad[:-1]
    ) * pair_valid

    ofi_share = ofi_b + ofi_a
    ofi_dollar = ofi_db + ofi_da

    X = np.concatenate(
        [
            ofi_share,
            ofi_dollar,
            obi_share,
            obi_dollar,
            spread.reshape(-1, 1),
            weighted_mid.reshape(-1, 1),
            psi,
            event_mask.astype(np.float64).reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)

    summary = {
        "ticker": ticker,
        "zip_file": str(zip_path.relative_to(ROOT)),
        "message_member": msg_member,
        "orderbook_member": ob_member,
        "message_rows": int(len(msg)),
        "orderbook_rows": int(orderbook_rows_seen),
        "event_bins": int(event_mask.sum()),
        "carry_forward_bins": int(T - event_mask.sum()),
        "best_spread_min": float(np.min(spread)),
        "best_spread_median": float(np.median(spread)),
        "best_spread_max": float(np.max(spread)),
        "mid_min": float(np.min(mid)),
        "mid_max": float(np.max(mid)),
        "invalid_deep_level_cells": int((~valid_level).sum()),
        "nonzero_ofi_share_entries": int(np.count_nonzero(ofi_share)),
        "nonzero_message_flow_bins": int(np.count_nonzero(np.linalg.norm(psi, axis=1) > 0)),
    }
    return TickerBuildResult(
        ticker=ticker,
        X_raw=X,
        mid=mid.astype(np.float32),
        spread=spread.astype(np.float32),
        weighted_mid=weighted_mid.astype(np.float32),
        event_mask=event_mask,
        summary=summary,
    )


def build_targets(mid_all: np.ndarray, spread_all: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    y_logret: Dict[str, np.ndarray] = {}
    z_spread: Dict[str, np.ndarray] = {}
    log_mid = np.log(mid_all.astype(np.float64))
    for tau in TAUS:
        y = np.full((T, N), np.nan, dtype=np.float32)
        z = np.full((T, N), np.nan, dtype=np.float32)
        y[:-tau] = (log_mid[tau:] - log_mid[:-tau]).astype(np.float32)
        z[:-tau] = ((mid_all[tau:] - mid_all[:-tau]) / (spread_all[:-tau] + EPS)).astype(np.float32)
        y_logret[f"y_logret_tau_{tau}"] = y
        z_spread[f"z_spread_tau_{tau}"] = z
    return y_logret, z_spread


def fmt_obj(x: object) -> str:
    if isinstance(x, float):
        return f"{x:.8g}"
    if isinstance(x, (tuple, list)):
        return str(tuple(x))
    return str(x)


def write_reports(gate: Gate, summaries: List[Dict[str, object]], metadata: Dict[str, object]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "step02_feature_build_500ms.json"
    md_path = REPORT_DIR / "step02_feature_build_500ms.md"

    payload = {
        "overall_status": "PASS" if gate.passed else "FAIL",
        "metadata": metadata,
        "ticker_summaries": summaries,
        "checks": gate.rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Step 02 500ms Feature Build Benchmark")
    lines.append("")
    lines.append(f"Overall status: **{'PASS' if gate.passed else 'FAIL'}**")
    lines.append("")
    lines.append("## Output contract")
    lines.append(f"- Tickers: `{', '.join(TICKERS)}`")
    lines.append(f"- Time bins: `{T}`")
    lines.append(f"- Assets: `{N}`")
    lines.append(f"- LOB levels: `{H}`")
    lines.append(f"- Feature dimension: `{D}`")
    lines.append(f"- Feature tensor: `X_raw.shape = {tuple(metadata['X_raw_shape'])}`")
    lines.append(f"- Standardized tensor: `X_std.shape = {tuple(metadata['X_std_shape'])}`")
    lines.append(f"- Output NPZ: `{metadata['output_npz']}`")
    lines.append("")
    lines.append("## Ticker feature summaries")
    lines.append("| ticker | event bins | carry-forward bins | min spread | median spread | nonzero OFI entries | nonzero message-flow bins | invalid deep-level cells |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        lines.append(
            f"| {s['ticker']} | {s['event_bins']} | {s['carry_forward_bins']} | "
            f"{float(s['best_spread_min']):.6g} | {float(s['best_spread_median']):.6g} | "
            f"{s['nonzero_ofi_share_entries']} | {s['nonzero_message_flow_bins']} | {s['invalid_deep_level_cells']} |"
        )
    lines.append("")
    lines.append("## Standardization audit")
    lines.append(f"- Nonconstant standardized train features: `{metadata['nonconstant_train_features']}`")
    lines.append(f"- Max absolute train mean after standardization: `{metadata['max_abs_train_mean_nonconstant']:.6g}`")
    lines.append(f"- Max absolute train std error after standardization: `{metadata['max_abs_train_std_error_nonconstant']:.6g}`")
    lines.append(f"- Near-constant train feature count: `{metadata['near_constant_train_features']}`")
    lines.append("")
    lines.append("## Target audit")
    lines.append("| tau | valid samples per asset | train | validation | test | max abs log-return | max abs spread-normalized return |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for item in metadata["target_audit"]:
        lines.append(
            f"| {item['tau']} | {item['valid_samples_per_asset']} | {item['train_samples']} | "
            f"{item['val_samples']} | {item['test_samples']} | {item['max_abs_logret']:.6g} | {item['max_abs_spread_return']:.6g} |"
        )
    lines.append("")
    lines.append("## Benchmark checks")
    lines.append("| check | status | observed | expected |")
    lines.append("|---|---|---|---|")
    for row in gate.rows:
        lines.append(
            f"| {row['check']} | {row['status']} | `{fmt_obj(row['observed'])}` | `{fmt_obj(row['expected'])}` |"
        )
    lines.append("")
    if gate.passed:
        lines.append("Gate passed: `.pipeline_state/step02_feature_build.PASS` should exist.")
        lines.append("Next module should be `03_graph_build_and_baseline.sh`, but do not run it until this report is reviewed.")
    else:
        lines.append("Gate failed: inspect the failing checks above before continuing.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    gate = Gate()
    gate.add("step01_gate_exists", (STATE_DIR / "step01_raw_audit.PASS").exists(), ".pipeline_state/step01_raw_audit.PASS", "exists")
    gate.add("time_bins", T == 46800, T, 46800)
    gate.add("chronological_split", (TRAIN_END, VAL_END - TRAIN_END, TEST_END - VAL_END) == (32760, 7020, 7020), (TRAIN_END, VAL_END - TRAIN_END, TEST_END - VAL_END), (32760, 7020, 7020))
    gate.add("feature_dimension_contract", D == 51 and len(FEATURE_NAMES) == 51, len(FEATURE_NAMES), 51)

    results: List[TickerBuildResult] = []
    for ticker in TICKERS:
        result = build_ticker(ticker)
        results.append(result)
        gate.add(f"{ticker}_X_shape", result.X_raw.shape == (T, D), result.X_raw.shape, (T, D))
        gate.add(f"{ticker}_event_bins", int(result.event_mask.sum()) == EXPECTED_EVENT_BINS[ticker], int(result.event_mask.sum()), EXPECTED_EVENT_BINS[ticker])
        gate.add(f"{ticker}_spread_positive", bool(np.all(result.spread > 0)), float(np.min(result.spread)), "> 0")
        gate.add(f"{ticker}_mid_positive", bool(np.all(result.mid > 0)), float(np.min(result.mid)), "> 0")
        gate.add(f"{ticker}_X_finite", bool(np.isfinite(result.X_raw).all()), bool(np.isfinite(result.X_raw).all()), True)

    X_raw = np.stack([r.X_raw for r in results], axis=1).astype(np.float32)  # T x N x D
    mid_all = np.stack([r.mid for r in results], axis=1).astype(np.float32)
    spread_all = np.stack([r.spread for r in results], axis=1).astype(np.float32)
    weighted_mid_all = np.stack([r.weighted_mid for r in results], axis=1).astype(np.float32)
    event_mask_all = np.stack([r.event_mask for r in results], axis=1).astype(np.uint8)

    any_asset_event_bins = int((event_mask_all.sum(axis=1) > 0).sum())
    gate.add("X_raw_shape", X_raw.shape == (T, N, D), X_raw.shape, (T, N, D))
    gate.add("any_asset_event_bins", any_asset_event_bins == EXPECTED_ANY_ASSET_EVENT_BINS, any_asset_event_bins, EXPECTED_ANY_ASSET_EVENT_BINS)
    gate.add("X_raw_all_finite", bool(np.isfinite(X_raw).all()), bool(np.isfinite(X_raw).all()), True)

    # Standardize using training statistics only, asset by asset and feature by feature.
    train_X = X_raw[:TRAIN_END]
    mu = train_X.mean(axis=0, dtype=np.float64).astype(np.float32)  # N x D
    sigma = train_X.std(axis=0, dtype=np.float64).astype(np.float32)
    near_constant = sigma < 1e-10
    sigma_safe = np.where(near_constant, 1.0, sigma).astype(np.float32)
    X_std = ((X_raw - mu[None, :, :]) / sigma_safe[None, :, :]).astype(np.float32)

    std_train = X_std[:TRAIN_END]
    train_mean = std_train.mean(axis=0, dtype=np.float64)
    train_std = std_train.std(axis=0, dtype=np.float64)
    nonconstant = ~near_constant
    max_abs_mean = float(np.max(np.abs(train_mean[nonconstant]))) if np.any(nonconstant) else 0.0
    max_abs_std_error = float(np.max(np.abs(train_std[nonconstant] - 1.0))) if np.any(nonconstant) else 0.0

    gate.add("X_std_shape", X_std.shape == (T, N, D), X_std.shape, (T, N, D))
    gate.add("X_std_all_finite", bool(np.isfinite(X_std).all()), bool(np.isfinite(X_std).all()), True)
    gate.add("standardized_train_mean_close_to_zero", max_abs_mean < 1e-4, max_abs_mean, "< 1e-4")
    gate.add("standardized_train_std_close_to_one", max_abs_std_error < 1e-4, max_abs_std_error, "< 1e-4")

    y_logret, z_spread = build_targets(mid_all, spread_all)
    target_audit: List[Dict[str, object]] = []
    for tau in TAUS:
        y = y_logret[f"y_logret_tau_{tau}"]
        z = z_spread[f"z_spread_tau_{tau}"]
        valid = slice(0, T - tau)
        y_valid = y[valid]
        z_valid = z[valid]
        gate.add(f"target_tau_{tau}_logret_finite", bool(np.isfinite(y_valid).all()), bool(np.isfinite(y_valid).all()), True)
        gate.add(f"target_tau_{tau}_spread_return_finite", bool(np.isfinite(z_valid).all()), bool(np.isfinite(z_valid).all()), True)
        gate.add(f"target_tau_{tau}_tail_nan", bool(np.isnan(y[-tau:]).all() and np.isnan(z[-tau:]).all()), "tail NaN", "tail NaN")
        target_audit.append(
            {
                "tau": tau,
                "valid_samples_per_asset": int(T - tau),
                "train_samples": int(TRAIN_END - tau),
                "val_samples": int(VAL_END - TRAIN_END - tau),
                "test_samples": int(TEST_END - VAL_END - tau),
                "max_abs_logret": float(np.nanmax(np.abs(y))),
                "max_abs_spread_return": float(np.nanmax(np.abs(z))),
            }
        )

    out_npz = PROCESSED_DIR / "features_500ms_step02.npz"
    metadata_json = PROCESSED_DIR / "features_500ms_step02_metadata.json"
    ticker_summary_csv = PROCESSED_DIR / "features_500ms_step02_ticker_summary.csv"

    save_kwargs: Dict[str, object] = {
        "X_raw": X_raw,
        "X_std": X_std,
        "mid": mid_all,
        "spread": spread_all,
        "weighted_mid": weighted_mid_all,
        "event_mask": event_mask_all,
        "train_mu": mu,
        "train_sigma": sigma,
        "train_sigma_safe": sigma_safe,
        "tickers": np.array(TICKERS),
        "feature_names": np.array(FEATURE_NAMES),
        "taus": np.array(TAUS, dtype=np.int64),
    }
    save_kwargs.update(y_logret)
    save_kwargs.update(z_spread)
    print(f"[INFO] Saving compressed feature tensor to {out_npz}", flush=True)
    np.savez_compressed(out_npz, **save_kwargs)

    summaries = [r.summary for r in results]
    pd.DataFrame(summaries).to_csv(ticker_summary_csv, index=False)

    metadata: Dict[str, object] = {
        "root": str(ROOT),
        "output_npz": str(out_npz.relative_to(ROOT)),
        "metadata_json": str(metadata_json.relative_to(ROOT)),
        "ticker_summary_csv": str(ticker_summary_csv.relative_to(ROOT)),
        "tickers": TICKERS,
        "T": T,
        "N": N,
        "H": H,
        "D": D,
        "feature_names": FEATURE_NAMES,
        "X_raw_shape": list(X_raw.shape),
        "X_std_shape": list(X_std.shape),
        "mid_shape": list(mid_all.shape),
        "event_mask_shape": list(event_mask_all.shape),
        "time_window_seconds": [START_SEC, END_SEC],
        "bin_width_seconds": BIN_WIDTH,
        "splits": {
            "train": [0, TRAIN_END],
            "validation": [TRAIN_END, VAL_END],
            "test": [VAL_END, TEST_END],
        },
        "taus": TAUS,
        "target_audit": target_audit,
        "nonconstant_train_features": int(nonconstant.sum()),
        "near_constant_train_features": int(near_constant.sum()),
        "max_abs_train_mean_nonconstant": max_abs_mean,
        "max_abs_train_std_error_nonconstant": max_abs_std_error,
        "any_asset_event_bins": any_asset_event_bins,
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    gate.add("output_npz_exists", out_npz.exists() and out_npz.stat().st_size > 0, str(out_npz.relative_to(ROOT)), "nonempty file")
    gate.add("metadata_json_exists", metadata_json.exists() and metadata_json.stat().st_size > 0, str(metadata_json.relative_to(ROOT)), "nonempty file")
    gate.add("ticker_summary_csv_exists", ticker_summary_csv.exists() and ticker_summary_csv.stat().st_size > 0, str(ticker_summary_csv.relative_to(ROOT)), "nonempty file")

    write_reports(gate, summaries, metadata)

    pass_file = STATE_DIR / "step02_feature_build.PASS"
    if gate.passed:
        pass_file.write_text("PASS\n", encoding="utf-8")
        print("[PASS] Step 02 feature build benchmark passed.")
        print(f"[REPORT] {REPORT_DIR / 'step02_feature_build_500ms.md'}")
        return 0
    else:
        if pass_file.exists():
            pass_file.unlink()
        print("[FAIL] Step 02 feature build benchmark failed. See report:", REPORT_DIR / "step02_feature_build_500ms.md", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

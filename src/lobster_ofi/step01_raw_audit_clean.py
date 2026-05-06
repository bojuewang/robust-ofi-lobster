from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

TICKERS = ["AMZN", "AAPL", "GOOG", "INTC", "MSFT"]
LEVELS = 10
KAPPA0 = 34200.0
KAPPA1 = 57600.0
BIN_SECONDS = 0.5
N_BINS = 46800
PRICE_SCALE = 10000.0
EXPECTED_ROWS = {
    "AAPL": 400391,
    "AMZN": 269748,
    "GOOG": 147916,
    "INTC": 624040,
    "MSFT": 668765,
}
EXPECTED_EVENT_BINS = {
    "AAPL": 34800,
    "AMZN": 29801,
    "GOOG": 22678,
    "INTC": 36086,
    "MSFT": 37005,
}
EXPECTED_ANY_ASSET_EVENT_BINS = 46290

MSG_COLS = ["time", "event_type", "order_id", "size", "price", "direction"]


def find_zip(raw_dir: Path, ticker: str) -> Path:
    matches = sorted(raw_dir.glob(f"LOBSTER_SampleFile_{ticker}_*_10*.zip"))
    if not matches:
        matches = sorted(raw_dir.glob(f"*{ticker}*.zip"))
    if not matches:
        raise FileNotFoundError(f"Missing raw LOBSTER sample zip for {ticker} in {raw_dir}")
    return matches[0]


def find_members(zf: zipfile.ZipFile, ticker: str) -> Tuple[str, str]:
    names = zf.namelist()
    msg = [n for n in names if ticker in Path(n).name and "message" in Path(n).name.lower() and n.endswith(".csv")]
    ob = [n for n in names if ticker in Path(n).name and "orderbook" in Path(n).name.lower() and n.endswith(".csv")]
    if not msg or not ob:
        raise FileNotFoundError(f"Cannot find message/orderbook CSV for {ticker} inside {zf.filename}")
    return msg[0], ob[0]


def count_lines(zf: zipfile.ZipFile, member: str) -> int:
    with zf.open(member) as f:
        return sum(1 for _ in f)


def read_first_orderbook_sample(zf: zipfile.ZipFile, member: str, nrows: int = 500) -> pd.DataFrame:
    cols = []
    for h in range(1, LEVELS + 1):
        cols += [f"ask_price_{h}", f"ask_size_{h}", f"bid_price_{h}", f"bid_size_{h}"]
    with zf.open(member) as f:
        return pd.read_csv(f, header=None, names=cols, nrows=nrows)


def read_message(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with zf.open(member) as f:
        msg = pd.read_csv(
            f,
            header=None,
            names=MSG_COLS,
            usecols=[0, 1, 3, 4, 5],
            dtype={
                "time": "float64",
                "event_type": "int16",
                "order_id": "int64",
                "size": "float64",
                "price": "float64",
                "direction": "int8",
            },
        )
    return msg


def clean_message(msg: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = msg.copy()
    out["price"] = out["price"] / PRICE_SCALE
    out["bin_idx"] = np.floor((out["time"].to_numpy() - KAPPA0) / BIN_SECONDS).astype("int64")
    out["bin_id"] = out["bin_idx"] + 1
    out["is_regular_time"] = (out["time"] >= KAPPA0) & (out["time"] < KAPPA1)
    out["is_valid_event"] = out["event_type"].isin([1, 2, 3, 4, 5])
    out["event_family"] = np.select(
        [out["event_type"].eq(1), out["event_type"].isin([2, 3]), out["event_type"].isin([4, 5])],
        ["sub", "can", "exe"],
        default="other",
    )
    out["ticker"] = ticker
    keep = out["is_regular_time"] & (out["bin_idx"] >= 0) & (out["bin_idx"] < N_BINS)
    cols = ["ticker", "time", "bin_idx", "bin_id", "event_type", "event_family", "size", "price", "direction", "is_valid_event"]
    return out.loc[keep, cols].reset_index(drop=True)


def audit_orderbook_sample(sample: pd.DataFrame) -> Dict[str, Any]:
    # LOBSTER orderbook prices are multiplied by 10000. Check best-level sanity on a sample.
    ask1 = sample["ask_price_1"].astype(float) / PRICE_SCALE
    bid1 = sample["bid_price_1"].astype(float) / PRICE_SCALE
    askq1 = sample["ask_size_1"].astype(float)
    bidq1 = sample["bid_size_1"].astype(float)
    spread = ask1 - bid1
    return {
        "sample_rows": int(len(sample)),
        "sample_columns": int(sample.shape[1]),
        "ask1_min": float(ask1.min()),
        "bid1_min": float(bid1.min()),
        "spread_min": float(spread.min()),
        "spread_median": float(spread.median()),
        "positive_best_sizes": bool(((askq1 > 0) & (bidq1 > 0)).all()),
    }


def process_ticker(raw_dir: Path, interim_dir: Path, ticker: str) -> Dict[str, Any]:
    zip_path = find_zip(raw_dir, ticker)
    with zipfile.ZipFile(zip_path) as zf:
        msg_member, ob_member = find_members(zf, ticker)
        msg = read_message(zf, msg_member)
        message_rows = int(len(msg))
        # Step 01 deliberately does not stream-count the full orderbook file, because
        # the large level-10 files make that slow. Full orderbook row equality is verified
        # in Step 02 when the full book is read for feature construction. Here we verify
        # the member exists and has the expected 40 columns on a sample.
        orderbook_rows = None
        ob_sample = read_first_orderbook_sample(zf, ob_member)

    clean = clean_message(msg, ticker)

    # Store a compact cleaned 500ms event summary, not the full message table.
    # Full message/orderbook parsing is deferred to Step 02 feature construction.
    valid = clean[clean["is_valid_event"]].copy()
    event_bins = int(valid["bin_idx"].nunique())
    summary = pd.DataFrame({"bin_idx": np.arange(N_BINS, dtype=np.int64), "bin_id": np.arange(1, N_BINS + 1, dtype=np.int64)})
    summary["ticker"] = ticker
    counts_all = valid.groupby("bin_idx").size().reindex(range(N_BINS), fill_value=0).astype(int)
    summary["n_valid_events"] = counts_all.to_numpy()
    summary["event_mask"] = (summary["n_valid_events"] > 0).astype(int)
    for family in ["sub", "can", "exe"]:
        fam = valid[valid["event_family"] == family]
        summary[f"{family}_count"] = fam.groupby("bin_idx").size().reindex(range(N_BINS), fill_value=0).astype(int).to_numpy()
        summary[f"{family}_volume"] = fam.groupby("bin_idx")["size"].sum().reindex(range(N_BINS), fill_value=0.0).to_numpy()
    summary_path = interim_dir / f"event_summary_{ticker}.csv"
    summary.to_csv(summary_path, index=False)

    event_counts = {str(k): int(v) for k, v in msg["event_type"].value_counts().sort_index().items()}
    family_counts = {str(k): int(v) for k, v in clean["event_family"].value_counts().sort_index().items()}
    volume_by_family = {str(k): float(v) for k, v in clean.groupby("event_family")["size"].sum().sort_index().items()}
    orderbook_audit = audit_orderbook_sample(ob_sample)
    return {
        "ticker": ticker,
        "zip_path": str(zip_path),
        "message_member": msg_member,
        "orderbook_member": ob_member,
        "message_rows": int(message_rows),
        "orderbook_rows": orderbook_rows,
        "first_timestamp": float(msg["time"].iloc[0]),
        "last_timestamp": float(msg["time"].iloc[-1]),
        "clean_message_rows": int(len(clean)),
        "event_bins": event_bins,
        "carry_forward_bins": int(N_BINS - event_bins),
        "event_type_counts": event_counts,
        "event_family_counts": family_counts,
        "volume_by_family": volume_by_family,
        "orderbook_sample_audit": orderbook_audit,
        "event_summary_path": str(summary_path),
    }


def build_any_asset_mask(interim_dir: Path) -> int:
    mask = np.zeros(N_BINS, dtype=bool)
    for ticker in TICKERS:
        path = interim_dir / f"event_summary_{ticker}.csv"
        df = pd.read_csv(path, usecols=["bin_idx", "event_mask"])
        idx = df.loc[df["event_mask"] == 1, "bin_idx"].to_numpy(dtype="int64")
        mask[idx] = True
    return int(mask.sum())


def add_check(checks: List[Dict[str, Any]], name: str, ok: bool, observed: Any, expected: Any, severity: str = "FAIL") -> None:
    checks.append({"name": name, "status": "PASS" if ok else severity, "observed": observed, "expected": expected})


def benchmark(stats: List[Dict[str, Any]], any_asset_event_bins: int) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    sdict = {s["ticker"]: s for s in stats}
    for ticker in TICKERS:
        s = sdict[ticker]
        add_check(checks, f"{ticker}_zip_found", Path(s["zip_path"]).exists(), s["zip_path"], "exists")
        add_check(checks, f"{ticker}_orderbook_member_exists", isinstance(s["orderbook_member"], str) and s["orderbook_member"].endswith(".csv"), s["orderbook_member"], "orderbook csv exists")
        add_check(checks, f"{ticker}_expected_message_row_count", s["message_rows"] == EXPECTED_ROWS[ticker], s["message_rows"], EXPECTED_ROWS[ticker])
        add_check(checks, f"{ticker}_timestamp_start", KAPPA0 <= s["first_timestamp"] < KAPPA0 + 1.0, s["first_timestamp"], f"[{KAPPA0}, {KAPPA0+1})")
        add_check(checks, f"{ticker}_timestamp_end", KAPPA1 - 1.0 < s["last_timestamp"] < KAPPA1, s["last_timestamp"], f"({KAPPA1-1}, {KAPPA1})")
        add_check(checks, f"{ticker}_event_bins", s["event_bins"] == EXPECTED_EVENT_BINS[ticker], s["event_bins"], EXPECTED_EVENT_BINS[ticker])
        add_check(checks, f"{ticker}_orderbook_has_40_columns", s["orderbook_sample_audit"]["sample_columns"] == 40, s["orderbook_sample_audit"]["sample_columns"], 40)
        add_check(checks, f"{ticker}_sample_spread_positive", s["orderbook_sample_audit"]["spread_min"] > 0, s["orderbook_sample_audit"]["spread_min"], "> 0")
        add_check(checks, f"{ticker}_sample_best_sizes_positive", bool(s["orderbook_sample_audit"]["positive_best_sizes"]), s["orderbook_sample_audit"]["positive_best_sizes"], True)
    add_check(checks, "any_asset_event_bins", any_asset_event_bins == EXPECTED_ANY_ASSET_EVENT_BINS, any_asset_event_bins, EXPECTED_ANY_ASSET_EVENT_BINS)
    overall = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    return {"overall_status": overall, "checks": checks}


def write_reports(stats: List[Dict[str, Any]], any_asset_event_bins: int, bench: Dict[str, Any], reports_dir: Path) -> None:
    payload = {
        "config": {
            "tickers": TICKERS,
            "levels": LEVELS,
            "kappa0": KAPPA0,
            "kappa1": KAPPA1,
            "bin_seconds": BIN_SECONDS,
            "n_bins": N_BINS,
            "train_val_test": {"train": 32760, "val": 7020, "test": 7020},
        },
        "ticker_stats": stats,
        "any_asset_event_bins": any_asset_event_bins,
        "benchmark": bench,
    }
    json_path = reports_dir / "step01_raw_audit_clean.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Step 01 Raw LOBSTER Audit + Cleaning Benchmark")
    lines.append("")
    lines.append(f"Overall status: **{bench['overall_status']}**")
    lines.append("")
    lines.append("## Time-grid contract")
    lines.append(f"- Continuous trading window: `[34200, 57600)` seconds after midnight")
    lines.append(f"- Bin width: `0.5` seconds")
    lines.append(f"- Number of bins: `{N_BINS}`")
    lines.append(f"- Chronological split: `32760 / 7020 / 7020`")
    lines.append(f"- Any-asset event bins: `{any_asset_event_bins}`")
    lines.append("")
    lines.append("## Ticker audit")
    lines.append("| ticker | msg rows | orderbook file | first ts | last ts | event bins | carry-forward bins | sample min spread |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in stats:
        lines.append(
            f"| {s['ticker']} | {s['message_rows']} | exists | "
            f"{s['first_timestamp']:.6f} | {s['last_timestamp']:.6f} | "
            f"{s['event_bins']} | {s['carry_forward_bins']} | {s['orderbook_sample_audit']['spread_min']:.6g} |"
        )
    lines.append("")
    lines.append("## Benchmark checks")
    lines.append("| check | status | observed | expected |")
    lines.append("|---|---|---|---|")
    for c in bench["checks"]:
        lines.append(f"| {c['name']} | {c['status']} | `{c['observed']}` | `{c['expected']}` |")
    lines.append("")
    if bench["overall_status"] == "PASS":
        lines.append("Gate passed: `.pipeline_state/step01_raw_audit.PASS` should exist.")
        lines.append("Next module should be `02_feature_build_500ms.sh`, but do not run it until this report is reviewed.")
    else:
        lines.append("Gate failed. Do not proceed. Paste this report back for debugging and parameter adjustment.")
    (reports_dir / "step01_raw_audit_clean.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/benchmarks"))
    parser.add_argument("--ticker", choices=TICKERS, default=None, help="Process one ticker and write its stats JSON")
    parser.add_argument("--combine-only", action="store_true", help="Combine per-ticker stats and run the benchmark gate")
    args = parser.parse_args()
    args.interim_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    Path(".pipeline_state").mkdir(exist_ok=True)

    if args.ticker is not None:
        print(f"[INFO] Step 01 auditing {args.ticker}", flush=True)
        stat = process_ticker(args.raw_dir, args.interim_dir, args.ticker)
        (args.reports_dir / f"step01_stats_{args.ticker}.json").write_text(json.dumps(stat, indent=2), encoding="utf-8")
        print(f"[INFO] Finished {args.ticker}", flush=True)
        return 0

    if args.combine_only:
        stats: List[Dict[str, Any]] = []
        for ticker in TICKERS:
            stat_path = args.reports_dir / f"step01_stats_{ticker}.json"
            if not stat_path.exists():
                raise FileNotFoundError(f"Missing per-ticker stats: {stat_path}")
            stats.append(json.loads(stat_path.read_text(encoding="utf-8")))
        any_asset_event_bins = build_any_asset_mask(args.interim_dir)
        bench = benchmark(stats, any_asset_event_bins)
        write_reports(stats, any_asset_event_bins, bench, args.reports_dir)

        pass_file = Path(".pipeline_state/step01_raw_audit.PASS")
        if bench["overall_status"] == "PASS":
            pass_file.write_text("PASS\n", encoding="utf-8")
            print("[PASS] Step 01 raw audit + cleaning benchmark passed.")
            print("[INFO] Paste reports/benchmarks/step01_raw_audit_clean.md back into ChatGPT.")
            return 0
        if pass_file.exists():
            pass_file.unlink()
        print("[FAIL] Step 01 raw audit + cleaning benchmark failed.", file=sys.stderr)
        print("[INFO] Paste reports/benchmarks/step01_raw_audit_clean.md back into ChatGPT.", file=sys.stderr)
        return 1

    # Default: run per-ticker processing in this process. The shell script uses separate
    # processes to keep memory stable, but this fallback is useful for debugging.
    stats: List[Dict[str, Any]] = []
    for ticker in TICKERS:
        print(f"[INFO] Step 01 auditing {ticker}", flush=True)
        stats.append(process_ticker(args.raw_dir, args.interim_dir, ticker))
        gc.collect()
    any_asset_event_bins = build_any_asset_mask(args.interim_dir)
    bench = benchmark(stats, any_asset_event_bins)
    write_reports(stats, any_asset_event_bins, bench, args.reports_dir)
    return 0 if bench["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Aggregate baseline summary.csv files and produce aggregated tables (no pretrained).
This version normalizes the key columns (feature_set, metric, loss_type, neg_strategy)
so that missing/empty values become the canonical 'n/a'.
Compatible with older pandas versions.
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

KEYS = ["feature_set", "metric", "loss_type", "neg_strategy"]

def normalize_key_value(v):
    if pd.isna(v):
        return "n/a"
    s = str(v).strip()
    if s == "":
        return "n/a"
    s_low = s.lower()
    if s_low in {"nan", "none", "na", "n/a", "null"}:
        return "n/a"
    return s

def read_and_normalize(path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  - Could not read {path}: {e}")
        return None
    # Ensure key columns exist
    for k in KEYS:
        if k not in df.columns:
            df[k] = "n/a"
    # Normalize key columns
    for k in KEYS:
        df[k] = df[k].apply(normalize_key_value)
    df["__source_file"] = str(path)
    return df

def concat_group(files):
    dfs = []
    for p in files:
        d = read_and_normalize(p)
        if d is None:
            continue
        dfs.append(d)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True, sort=False)

def aggregate_df_from_concat(df_concat):
    if df_concat is None:
        return None

    # Numeric candidate columns: everything except keys and __source_file
    numeric_cols = [c for c in df_concat.columns if c not in KEYS + ["__source_file"]]

    # Coerce numeric-like columns
    for c in numeric_cols:
        df_concat[c] = pd.to_numeric(df_concat[c], errors="coerce")

    grouped = df_concat.groupby(KEYS)

    # counts (number of runs)
    count_series = grouped.size()
    count_df = count_series.reset_index()
    count_df.rename(columns={0: "n_runs"}, inplace=True)

    # means and stds
    mean_df = grouped[numeric_cols].mean().reset_index()
    std_df = grouped[numeric_cols].std(ddof=0).reset_index().fillna(0.0)

    # rename mean/std columns
    for c in numeric_cols:
        if c in mean_df.columns:
            mean_df.rename(columns={c: f"{c}_mean"}, inplace=True)
        if c in std_df.columns:
            std_df.rename(columns={c: f"{c}_std"}, inplace=True)

    out = count_df.merge(mean_df, on=KEYS, how="left").merge(std_df, on=KEYS, how="left")

    # reorder: keys, n_runs, rest
    rest = [c for c in out.columns if c not in KEYS + ["n_runs"]]
    out = out[KEYS + ["n_runs"] + rest]
    return out

def find_all_summary_files(root: Path):
    return list(root.rglob("summary.csv"))

def save_table_image(df, out_png, title=None, fontsize=9):
    if df is None or df.shape[0] == 0:
        print(f"  - No rows to plot for {out_png}, skipping image.")
        return
    ncols = df.shape[1]
    nrows = df.shape[0]
    fig_w = max(10, ncols * 1.8)
    fig_h = max(2.5, nrows * 0.35 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=14)
    table = ax.table(cellText=df.values.tolist(), colLabels=df.columns.tolist(),
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.05)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  - Saved table image: {out_png}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/workspace/runs/reidentification_results")
    p.add_argument("--kw-baseline", default="baseline", help="Substring to identify baseline runs")
    p.add_argument("--out-dir", default="/workspace/runs/reidentification_results")
    p.add_argument("--no-keyword-filter", action="store_true",
                   help="If set, aggregate all summary.csv files as baseline (ignore keyword)")
    args = p.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f"Root not found: {root}", file=sys.stderr)
        sys.exit(1)

    all_summary = find_all_summary_files(root)
    print(f"Found {len(all_summary)} summary.csv files under {root}")
    if len(all_summary) == 0:
        print("No summary.csv files found. Exiting.")
        return

    if args.no_keyword_filter:
        base_files = all_summary
        print(f"Aggregating ALL {len(base_files)} files as baseline (no keyword filter).")
    else:
        kwb = args.kw_baseline.lower()
        base_files = [p for p in all_summary if kwb in str(p).lower()]
        print(f"Baseline candidate files: {len(base_files)} (keyword: '{args.kw_baseline}')")
        for f in base_files:
            print("  -", f)
        if len(base_files) == 0:
            print("No files classified as baseline by keyword. Use --no-keyword-filter or adjust --kw-baseline.")
            return

    # Read and normalize baseline files
    base_concat = concat_group(base_files)
    base_agg = aggregate_df_from_concat(base_concat)

    if base_agg is not None:
        base_out = out_dir / "results_baseline.csv"
        base_agg.to_csv(base_out, index=False)
        print(f"Saved aggregated baseline: {base_out}")
    else:
        print("No baseline aggregation produced.")
        return

    # Produce a compact PNG view for a few key metrics if present
    preferred_metrics = [
        "mAP_fair_mean", "mAP_fair_std",
        "success_rate_objects_%_mean", "success_rate_objects_%_std"
    ]
    disp_cols = [c for c in KEYS + ["n_runs"] + preferred_metrics if c in base_agg.columns]
    df_disp = base_agg[disp_cols].copy()

    # Format mean±std into single columns for display
    def merge_mean_std(df, base_name, label, is_percent=False):
        mean_col = f"{base_name}_mean"
        std_col = f"{base_name}_std"
        if mean_col in df.columns and std_col in df.columns:
            def fmt(m, s):
                if pd.isna(m):
                    return ""
                if is_percent:
                    return f"{float(m):0.2f} ± {float(0.0 if pd.isna(s) else s):0.2f}"
                return f"{float(m):0.4f} ± {float(0.0 if pd.isna(s) else s):0.4f}"
            df[label] = [fmt(m, s) for m, s in zip(df[mean_col], df[std_col])]
            df.drop([mean_col, std_col], axis=1, inplace=True)

    merge_mean_std(df_disp, "mAP_fair", "mAP_fair (mean±std)", is_percent=False)
    merge_mean_std(df_disp, "success_rate_objects_%", "success_rate_objects_% (mean±std %)", is_percent=True)

    save_table_image(df_disp, out_dir / "results_baseline.png", title="Baseline Aggregation")

    print("All done. Outputs written to:", out_dir)

if __name__ == "__main__":
    main()

# python
import os, glob
import pandas as pd
import numpy as np

config_output_dir = "/workspace/runs/reidentification_results/evaluation_results_mAP_baseline_run_1/best-seg_baseline/dinov2_conf_0.4"
summary_csv = os.path.join(config_output_dir, "summary.csv")

def to_pct(x):
    try:
        if pd.isna(x): return np.nan
        x = float(x)
        return np.nan if np.isnan(x) else round(x * 100.0, 2)
    except Exception:
        return np.nan

def read_cmc_csv(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    def pick(k):
        m = df[df["rank"] == k]
        return np.nan if m.empty else round(float(m["cmc"].iloc[0]) * 100.0, 2)
    return pick(1), pick(5), pick(10)

def cmc_from_results(results_csv):
    if not results_csv or not os.path.exists(results_csv):
        return None
    df = pd.read_csv(results_csv)
    # Coerce rank to numeric, drop invalid
    df["gt_rank"] = pd.to_numeric(df.get("gt_rank"), errors="coerce")
    df = df.dropna(subset=["gt_object_id", "gt_rank"])
    if df.empty:
        return None
    best = df.groupby("gt_object_id")["gt_rank"].min()
    def pct_le(k): return round(float((best <= k).mean()) * 100.0, 2)
    return pct_le(1), pct_le(5), pct_le(10)

def find_one(patterns):
    for p in patterns:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    return None

# Load summary if available
df_sum = pd.read_csv(summary_csv) if os.path.exists(summary_csv) else None
if df_sum is not None:
    for c in ["feature_set", "metric", "loss_type"]:
        if c in df_sum.columns:
            df_sum[c] = df_sum[c].astype(str).str.strip().str.lower()

# 1) Baseline: feature + cosine (ignore loss_type)
cmc_base = None
if df_sum is not None:
    sel = df_sum[(df_sum["feature_set"] == "feature") &
                 (df_sum["metric"].str.contains("cosine", na=False))]
    if not sel.empty:
        r = sel.iloc[0]
        c1, c5, c10 = to_pct(r.get("cmc_rank1")), to_pct(r.get("cmc_rank5")), to_pct(r.get("cmc_rank10"))
        if not any(pd.isna(v) for v in (c1, c5, c10)):
            cmc_base = (c1, c5, c10)

if cmc_base is None:
    csv_b = find_one([
        os.path.join(config_output_dir, "feature_cosine", "cmc_feature_cosine_cmc.csv"),
        os.path.join(config_output_dir, "feature_cosine", "cmc_*cosine*_cmc.csv"),
        os.path.join(config_output_dir, "feature_cosine", "cmc_feature_cosine.csv"),
    ])
    cmc_base = read_cmc_csv(csv_b)

if cmc_base is None:
    cmc_base = cmc_from_results(os.path.join(config_output_dir, "feature_cosine", "results.csv"))

# 2) Schedule: concat_feature + attention + triplet_schedule
cmc_sch = None
if df_sum is not None:
    sel = df_sum[(df_sum["feature_set"] == "concat_feature") &
                 (df_sum["metric"].str.contains("attention", na=False)) &
                 (df_sum["loss_type"].str.contains("triplet_schedule", na=False))]
    if not sel.empty:
        r = sel.iloc[0]
        c1, c5, c10 = to_pct(r.get("cmc_rank1")), to_pct(r.get("cmc_rank5")), to_pct(r.get("cmc_rank10"))
        if not any(pd.isna(v) for v in (c1, c5, c10)):
            cmc_sch = (c1, c5, c10)

if cmc_sch is None:
    csv_s = find_one([
        os.path.join(config_output_dir, "concat_feature_attention_triplet_schedule", "cmc_concat_feature_attention_triplet_schedule_cmc.csv"),
        os.path.join(config_output_dir, "concat_feature_attention_triplet_schedule", "cmc_*schedule*_cmc.csv"),
        os.path.join(config_output_dir, "concat_feature_attention_triplet_schedule", "cmc_concat_feature_attention_triplet_schedule.csv"),
    ])
    cmc_sch = read_cmc_csv(csv_s)

if cmc_sch is None:
    cmc_sch = cmc_from_results(os.path.join(config_output_dir, "concat_feature_attention_triplet_schedule", "results.csv"))

# Print final two rows
import pandas as pd
rows = [
    {"Setting": "Baseline (Cosine)", "Feature": "Visual-only",
     "CMC@1 (%)": cmc_base[0] if cmc_base else np.nan,
     "CMC@5 (%)": cmc_base[1] if cmc_base else np.nan,
     "CMC@10 (%)": cmc_base[2] if cmc_base else np.nan},
    {"Setting": "Attention (Triplet Schedule)", "Feature": "Fused",
     "CMC@1 (%)": cmc_sch[0] if cmc_sch else np.nan,
     "CMC@5 (%)": cmc_sch[1] if cmc_sch else np.nan,
     "CMC@10 (%)": cmc_sch[2] if cmc_sch else np.nan},
]
out = pd.DataFrame(rows)
print(out)
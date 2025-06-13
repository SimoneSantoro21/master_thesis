# paired_test_with_normality_and_effect_size.py

import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, shapiro

# 1) Define your patient indices here:
INDICES = ['003', '013', '016', '026', '075', '091', '092']

# 2) Build the list of patient directories:
PATIENT_DIRS = [
    os.path.join('/Volumes/SEAGATE BAS/DTI_data/base_dataset', f"Dbs_{idx}")
    for idx in INDICES
]

def load_stats(dirs, stat_type="MD"):
    """
    Reads each <stat_type>_stats.csv in the given dirs,
    returns a DataFrame with columns:
      ROI, mean_gr_th, std_gr_th, mean_inference, std_inference, Patient.
    """
    rows = []
    for d in dirs:
        f = os.path.join(d, f"{stat_type}_stats.csv")
        pid = os.path.basename(d)
        if os.path.exists(f):
            df = pd.read_csv(f)
            df = df.rename(columns={df.columns[0]: "ROI"})
            needed = ["mean_gr_th", "std_gr_th", "mean_inference", "std_inference"]
            if all(col in df.columns for col in needed):
                sub = df[["ROI", "mean_gr_th", "std_gr_th", "mean_inference", "std_inference"]].copy()
                sub["Patient"] = pid
                rows.append(sub)
            else:
                print(f"  ⚠️ {f} missing one of {needed}")
        else:
            print(f"  ⚠️ {f} not found")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# 3) Load both MD and FA stats:
md = load_stats(PATIENT_DIRS, stat_type="MD")
fa = load_stats(PATIENT_DIRS, stat_type="FA")

def summarize_and_test(df, roi):
    """
    For a given ROI in df, compute:
      - mean & std of mean_gr_th (a) and mean_inference (b)
      - Shapiro–Wilk p-value for a and for b
      - paired t-test p-value on a vs. b
      - Cohen's d if p < 0.05
    Returns:
      (a_mean, a_std,
       b_mean, b_std,
       shapiro_p_a, shapiro_p_b,
       ttest_p, cohens_d)
    """
    sub = df[df["ROI"] == roi]
    if len(sub) < 2:
        return (np.nan,)*8

    a = sub["mean_gr_th"].values
    b = sub["mean_inference"].values

    # 1) Means & stds
    a_mean, a_std = a.mean(), a.std(ddof=1)
    b_mean, b_std = b.mean(), b.std(ddof=1)

    # 2) Shapiro–Wilk p-values
    sh_p_a = shapiro(a).pvalue if len(a) >= 3 else np.nan
    sh_p_b = shapiro(b).pvalue if len(b) >= 3 else np.nan

    # 3) Paired t-test
    t_p = ttest_rel(a, b).pvalue

    # 4) Cohen's d (only if significant)
    if t_p < 0.05:
        cohens_d = (np.mean(a) - np.mean(b)) / np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2.0)
    else:
        cohens_d = np.nan

    return (a_mean, a_std, b_mean, b_std, sh_p_a, sh_p_b, t_p, cohens_d)

# 4) Prepare summary rows per ROI:
summary_rows = []
rois = sorted(set(md["ROI"]) | set(fa["ROI"]))
for roi in rois:
    row = {"ROI": roi}

    # MD summary
    (
        row["MD_gr_th_mean_avg"],
        row["MD_gr_th_mean_std"],
        row["MD_inference_mean_avg"],
        row["MD_inference_mean_std"],
        row["MD_shapiro_p"],
        row["MDinf_shapiro_p"],
        row["MD_mean_pvalue"],
        row["MD_cohens_d"]
    ) = summarize_and_test(md, roi)

    # FA summary
    (
        row["FA_gr_th_mean_avg"],
        row["FA_gr_th_mean_std"],
        row["FA_inference_mean_avg"],
        row["FA_inference_mean_std"],
        row["FA_shapiro_p"],
        row["FAinf_shapiro_p"],
        row["FA_mean_pvalue"],
        row["FA_cohens_d"]
    ) = summarize_and_test(fa, roi)

    summary_rows.append(row)

# 5) Build DataFrame, print & save:
summary = pd.DataFrame(summary_rows, columns=[
    "ROI",
    # MD
    "MD_gr_th_mean_avg", "MD_gr_th_mean_std",
    "MD_inference_mean_avg", "MD_inference_mean_std",
    "MD_shapiro_p", "MDinf_shapiro_p",
    "MD_mean_pvalue", "MD_cohens_d",
    # FA
    "FA_gr_th_mean_avg", "FA_gr_th_mean_std",
    "FA_inference_mean_avg", "FA_inference_mean_std",
    "FA_shapiro_p", "FAinf_shapiro_p",
    "FA_mean_pvalue", "FA_cohens_d",
])

print(summary.to_string(index=False))
summary.to_csv("single_shell_results.csv", index=False)
print("\nSaved to single_shell_results.csv")


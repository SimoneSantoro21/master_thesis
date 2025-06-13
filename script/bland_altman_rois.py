#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ───────────────────────────────────────────────────────────────────────────────
# LaTeX-style serif (STIX) + bumped sizes
mpl.rcParams.update({
    "font.family":           "serif",
    "font.serif":            ["STIXGeneral"],
    "mathtext.fontset":      "stix",
    "mathtext.rm":           "serif",
    "mathtext.it":           "serif:italic",
    "mathtext.bf":           "serif:bold",
    "font.size":             20,
    "axes.titlesize":        22,
    "axes.labelsize":        18,
    "xtick.labelsize":       16,
    "ytick.labelsize":       16,
    "legend.fontsize":       20,
    "legend.title_fontsize": 22,
})
# ───────────────────────────────────────────────────────────────────────────────

INDICES  = ['003', '013', '016', '026','075', '091', '092']
BASE_DIR = '/Volumes/SEAGATE BAS/DTI_data/base_dataset'


def load_roi_means(metric):
    records = []
    for idx in INDICES:
        f = os.path.join(BASE_DIR, f"Dbs_{idx}", f"{metric}_stats.csv")
        if not os.path.exists(f):
            print(f"Warning: {f} not found.")
            continue
        df = pd.read_csv(f)
        df = df.rename(columns={df.columns[0]: "ROI"})
        df = df[["ROI", "mean_gr_th", "mean_inference"]].copy()
        df.columns = ["ROI", "original", "synthetic"]
        df["Patient"] = idx
        records.append(df)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

def plot_bland_altman_grid(df, metric):
    rois     = sorted(df["ROI"].unique())
    patients = sorted(df["Patient"].unique())
    cmap     = cm.get_cmap("tab10", len(patients))

    fig = plt.figure(figsize=(18, 10))

    # layout params
    ncols, nrows = 3, 2
    left, right, top = 0.05, 0.05, 0.05
    bottom         = 0.20     # leave room for legend
    wspace         = 0.06     # horizontal gap
    hspace         = 0.15     # increased vertical gap

    # compute cell size
    width  = (1 - left - right - (ncols-1)*wspace) / ncols
    height = (1 - top - bottom - (nrows-1)*hspace) / nrows

    # column positions
    col_lefts = [left + i*(width + wspace) for i in range(ncols)]
    # row bottoms: index 0 is top row
    row_bottoms = [bottom + height + hspace, bottom]

    # for centering 2nd-row panels
    group_w    = 2*width + wspace
    start2     = (1 - group_w)/2
    second_cols = [start2, start2 + width + wspace]

    for i, roi in enumerate(rois):
        if i < 3:
            row, col = 0, i
            l = col_lefts[col]; b = row_bottoms[row]
        else:
            row, col = 1, i-3
            l = second_cols[col]; b = row_bottoms[row]

        ax = fig.add_axes([l, b, width, height])
        sub = df[df["ROI"] == roi]

        # scatter per patient
        for j, pid in enumerate(patients):
            psub = sub[sub["Patient"] == pid]
            if psub.empty: continue
            x = (psub["original"] + psub["synthetic"]) / 2
            y = psub["original"] - psub["synthetic"]
            ax.scatter(
                x, y,
                color=cmap(j),
                s=80,
                label=pid if i == 0 else "",
                edgecolor='k',
                linewidth=0.5
            )

        # bias & LoA
        d = sub["original"] - sub["synthetic"]
        bias = d.mean()
        loa  = 1.96 * d.std(ddof=1)
        ax.axhline(bias,      color='black', linestyle='--')
        ax.axhline(bias+loa,  color='gray',  linestyle=':')
        ax.axhline(bias-loa,  color='gray',  linestyle=':')

        ax.set_title(f"ROI = {roi}")
        ax.set_xlabel("Mean of original & synthetic")
        ax.set_ylabel(f"Diff ({metric})")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # single patient legend
    proxies = [
        Line2D([0],[0],
               marker='o', color=cmap(i),
               linestyle='None', markersize=8,
               label=pid)
        for i, pid in enumerate(patients)
    ]
    fig.legend(
        handles=proxies,
        title="Patient",
        loc='lower center',
        ncol=min(len(proxies), 7),
        frameon=False
    )

    # apply tight layout with our bottom margin
    plt.tight_layout(rect=[0, bottom, 1, 1])
    out_dir = "BA_plots"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"BA_{metric}_grid.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved combined figure → {out_file}")

if __name__ == "__main__":
    md_df = load_roi_means("MD")
    fa_df = load_roi_means("FA")
    plot_bland_altman_grid(md_df, "MD")
    plot_bland_altman_grid(fa_df, "FA")
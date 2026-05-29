"""Step 7 - all-vs-all operational-parameter / ASV-abundance correlation heatmap
(correlations_heatmap_reduced.png).

Ported from the correlations_heatmap cell of
../codiffusion_bioreactor/data_processing.py. The reference only wired in six hand
chosen metrics; here we load EVERY operational parameter produced by step 3 so the
figure is a true all-parameters-vs-all-ASVs correlation. Benjamini-Hochberg FDR is
applied per parameter; rows (ASVs) with no surviving correlation are dropped
(reduced=True).
"""
import re
import sys
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from matplotlib import colors
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (REPO, MODEL_INPUTS, MEASUREMENTS, CORR_DIR, load_json,
                     ORGANISMS_TO_HIGHLIGHT)

PREFIX = "ASV_correlations_all_relevant_samples_"


def slugify(s):
    s = re.sub(r"[\s/]+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9%_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "param"


def prettify(name):
    return (name.replace("H2", "$H_2$").replace("CO2", "$CO_2$").replace("CH4", "$CH_4$"))


def main():
    # map slug -> original parameter name (clean column labels)
    params = load_json(MEASUREMENTS / "operational_params.json")
    slug_to_param = {slugify(p): p for p in params}

    correlations, pvals = defaultdict(dict), defaultdict(dict)
    for path in sorted(glob.glob(str(CORR_DIR / f"{PREFIX}*.json"))):
        slug = Path(path).stem[len(PREFIX):]
        param = slug_to_param.get(slug, slug.replace("_", " "))
        for ID, v in load_json(path).items():
            correlations[param][ID] = v["correlation"]
            pvals[param][ID] = v["p_value"]
    print(f"loaded {len(correlations)} operational parameters")

    # Benjamini-Hochberg FDR per parameter; drop non-significant entries
    new_corr = {p: dict(d) for p, d in correlations.items()}
    new_pval = {p: dict(d) for p, d in pvals.items()}
    for p, d in pvals.items():
        reject = multipletests(list(d.values()), alpha=0.05, method="fdr_bh")[0]
        for i, k in enumerate(list(d.keys())):
            if not reject[i]:
                new_corr[p].pop(k)
                new_pval[p].pop(k)

    df = DataFrame(new_corr).fillna(0)
    # drop every all-blank ASV row AND every all-blank operational-parameter column
    # (all-zero lines hold no surviving correlation; removing them affects no nonzero cell)
    df = df.loc[(df != 0).any(axis=1), (df != 0).any(axis=0)]
    if df.empty:
        print("No FDR-significant correlations survived; nothing to plot.")
        return
    print(f"reduced matrix: {df.shape[0]} ASVs x {df.shape[1]} operational parameters "
          f"(blank rows/columns removed)")

    id_levels = load_json(MODEL_INPUTS / "iterativeID_levels.json")
    id_level_short = {k.split(".")[0]: v for k, v in id_levels.items()}

    min_max = (round(df.min().min(), 1), round(df.max().max(), 1))
    # adaptive canvas: keep cells legible for however many params/ASVs survive
    fig_w = max(20, df.shape[1] * 1.15 + 8)
    fig_h = max(20, df.shape[0] * 0.7 + 8)
    cm = sns.clustermap(df, cmap="coolwarm_r",
                        norm=colors.TwoSlopeNorm(vmin=min(min_max[0], -0.01), vcenter=0, vmax=max(min_max[1], 0.01)),
                        col_cluster=False, clip_on=True, figsize=(fig_w, fig_h),
                        cbar_kws={"label": "Correlation"})

    labelsize = 50
    cm.ax_heatmap.set_xlabel("Operational metric", fontsize=70, labelpad=40)
    cm.ax_heatmap.set_ylabel("ASV", fontsize=70, labelpad=40)
    cm.ax_col_dendrogram.set_visible(False)
    cm.ax_row_dendrogram.set_visible(False)
    cm.ax_heatmap.yaxis.tick_left()
    cm.ax_heatmap.yaxis.set_label_position("left")
    hm = cm.ax_heatmap.get_position()
    rd = cm.ax_row_dendrogram.get_position()
    cm.ax_heatmap.set_position([rd.x0, hm.y0, hm.x1 - rd.x0, hm.height])

    # colourbar
    cbar = cm.ax_cbar
    ticks = [round(t, 1) for t in cbar.get_yticks()[::2]]
    if ticks:
        ticks[0], ticks[-1] = min_max[0], min_max[1]
        cbar.set_yticks(ticks)
        cbar.set_yticklabels(ticks, fontsize=40)
    cbar.set_xlabel("Spearman $\\rho$", fontsize=60, labelpad=30)
    cbar.set_ylabel("")
    hp = cm.ax_heatmap.get_position()
    cm.ax_cbar.set_position([hp.x1 + 0.02, hp.y0 + hp.height * 0.6, 0.1, hp.height / 7])

    # label styling: bold methanogens, italic genera, prettified metric names
    ylabels = cm.ax_heatmap.get_yticklabels()
    for label in ylabels:
        t = label.get_text()
        if any(x in t for x in ORGANISMS_TO_HIGHLIGHT):
            label.set_fontsize(labelsize * 1.2)
            label.set_fontweight("bold")
        if id_level_short.get(t.split(".")[0]) == "Genus":
            label.set_fontstyle("italic")
        label.set_rotation(0)
    cm.ax_heatmap.set_yticklabels(ylabels, fontsize=labelsize)
    cm.ax_heatmap.set_xticklabels([prettify(t.get_text()) for t in cm.ax_heatmap.get_xticklabels()],
                                  fontsize=int(labelsize * 0.8), rotation=80, ha="right", rotation_mode="anchor")

    # annotate each significant cell with its rho
    drow = cm.dendrogram_row.reordered_ind
    data2d = cm.data2d
    for i in range(data2d.shape[0]):
        for j in range(data2d.shape[1]):
            v = data2d.iloc[i, j]
            if v == 0:
                continue
            color = "black" if abs(v) < 0.7 else "white"
            cm.ax_heatmap.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center", color=color, fontsize=28)

    out = REPO / "correlations_heatmap_reduced.png"
    cm.figure.savefig(out, bbox_inches="tight", dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

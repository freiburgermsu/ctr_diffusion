"""Step 5 - top-10 ASV relative-abundance heatmap
(abundance_heatmaps/top_10_asvs_(%_abundance).png).

Ported from the _create_heatmap cell of ../codiffusion_bioreactor/data_processing.py.
For each timepoint the 10 most abundant ASVs are taken; their union forms the rows.
Rows cluster by taxonomy (taxonomy_linkage), colours mark phylum, and a Shannon
diversity trace is overlaid on top. Unlike the correlation figures this uses the
FULL time course (every sequenced timepoint), matching the reference heatmap.
"""
import sys
from json import load
from math import log
from pathlib import Path

import numpy as np
from numpy import log10, inf, nan, nanmean, where, isnan
import pandas as pd
from pandas import DataFrame, Series
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from scipy.cluster import hierarchy
import sigfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (REPO, MODEL_INPUTS, HEATMAP_DIR, TAXO_LEVELS, load_json,
                     load_sample_days, ORGANISMS_TO_HIGHLIGHT, is_archaea, taxonomy_linkage)

TOP_NUM = 10
GENUS = "Genus"


def shannon_index(abundances):
    return -sum(a * log(a) for a in abundances if a > 0)


def build_heatmap(df, taxonomies, title, inlayed_data, color_map, genera_color_map, id_levels):
    new_cmap = LinearSegmentedColormap.from_list("NewMap", [(0.0, "aliceblue"), (0.25, "lightblue"), (1.0, "navy")])
    new_cmap.set_bad("aliceblue")
    vmin, vmax = df.min().min(), df.max().max()
    vcenter = log10(0.1)
    vcenter = min(max(vcenter, vmin + 1e-6), vmax - 1e-6)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    DEFAULT_COLOR = "lightgray"

    def lookup(idx):
        if idx in color_map:
            return color_map[idx]
        if idx in genera_color_map:
            return genera_color_map[idx]
        return DEFAULT_COLOR

    row_colors = Series({idx: lookup(idx) for idx in df.index}, name="Phylum")
    cm = sns.clustermap(df, row_colors=row_colors, cmap=new_cmap, norm=norm, clip_on=True,
                        col_cluster=False, row_cluster=True, figsize=(40, 20),
                        row_linkage=taxonomy_linkage(taxonomies), dendrogram_ratio=(0.2, 0.15))
    for tick in cm.ax_row_colors.get_xticklabels():
        tick.set_fontsize(22)

    cbar = cm.ax_cbar
    log_ticks = np.array([vmin, log10(0.01), vcenter, log10(0.2), vmax])
    cbar.set_yticks(log_ticks)
    original_ticks = 10 ** log_ticks
    original_labels = [f"{x*100:.1e}".replace("e-0", "E-") if x * 100 < 0.1 else round(x * 100)
                       for x in original_ticks]
    original_labels[0] = "0"
    cbar.set_yticklabels(original_labels, fontsize=22)
    cbar.set_yticks(log_ticks)
    cbar.set_xlabel("Rel. Abundance %", fontsize=18, labelpad=10)

    if inlayed_data:
        top_ax = cm.ax_col_dendrogram
        top_ax.clear()
        top_ax.plot(list(inlayed_data.keys()), list(inlayed_data.values()), color="black", marker="o", linewidth=3)
        top_ax.set_ylabel("Shannon Diversity", fontsize=20)
        top_ax.yaxis.tick_right()
        top_ax.yaxis.set_label_position("right")
        top_ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        top_ax.set_xticks([])
        top_ax.tick_params(axis="y", labelsize=16)

    for i in range(cm.data2d.shape[0]):
        for j in range(cm.data2d.shape[1]):
            cell = cm.data2d.iloc[i, j]
            if isnan(cell):
                continue
            value = 10 ** cell * 100
            if value < 0.1:
                continue
            color = "black" if value < 10 else "white"
            str_val = str(sigfig.round(value, 2)) if value >= 1 else str(round(value, 1))
            if str_val.endswith(".0"):
                str_val = str_val[:-2]
            cm.ax_heatmap.text(j + 0.5, i + 0.5, str_val, ha="center", va="center", color=color, fontsize=20)

    cm.figure.subplots_adjust(bottom=0.15, top=0.95)
    cm.ax_heatmap.set_yticklabels(cm.ax_heatmap.get_yticklabels(), fontsize=24, rotation=0)
    # x-axis day labels flush to the left edge of each column (day 0 flush with the box)
    _xlabels = [t.get_text() for t in cm.ax_heatmap.get_xticklabels()]
    cm.ax_heatmap.set_xticks(np.arange(len(_xlabels)))
    cm.ax_heatmap.set_xticklabels(_xlabels, fontsize=24, ha="left")
    cm.ax_heatmap.set_xlabel("Days after inoculation", fontsize=32, labelpad=20)
    cm.ax_row_dendrogram.xaxis.set_visible(False)
    cm.ax_row_dendrogram.text(0.65, -0.02, "Taxonomical tree", fontsize=24, ha="center",
                              transform=cm.ax_row_dendrogram.transAxes)
    cm.ax_heatmap.yaxis.tick_left()
    cm.ax_heatmap.yaxis.set_label_position("left")

    label_right, heatmap_right = 0.22, 0.68
    dendro_left, dendro_right = 0.72, 0.85
    hm_pos = cm.ax_heatmap.get_position()
    dend_pos = cm.ax_row_dendrogram.get_position()
    cm.ax_cbar.set_position([0.09, 0.86, 0.06, 0.14])
    cm.ax_heatmap.set_position([label_right, hm_pos.y0, heatmap_right - label_right, hm_pos.height])
    cm.ax_row_dendrogram.set_position([dendro_left, dend_pos.y0, dendro_right - dendro_left, dend_pos.height])
    cm.ax_row_dendrogram.invert_xaxis()
    gap = 0.03
    col_dend_pos = cm.ax_col_dendrogram.get_position()
    cm.ax_col_dendrogram.set_position([label_right, col_dend_pos.y0 + gap, heatmap_right - label_right, col_dend_pos.height])

    for label in cm.ax_heatmap.get_yticklabels():
        text = label.get_text()
        if any(x in text for x in ORGANISMS_TO_HIGHLIGHT):
            label.set_fontweight("bold")
        if id_levels.get(text) == "Genus":
            label.set_fontstyle("italic")

    hm_pos = cm.ax_heatmap.get_position()
    fig_w = cm.figure.get_figwidth()
    strip_w = 0.015
    cm.ax_row_colors.set_position([hm_pos.x1 + 0.005, hm_pos.y0, strip_w, hm_pos.height])
    cm.ax_heatmap.tick_params(axis="y", pad=strip_w * fig_w * 72 + 12)

    # phylum legend (Archaea / Bacteria grouped)
    phylum_color = {}
    for idx in df.index:
        parts = str(taxonomies.get(idx, "")).split("|")
        if len(parts) < 2:
            continue
        phylum = parts[1]
        if phylum in ("None", "", "Unknown", "nan"):
            continue
        c = row_colors.get(idx)
        if c is not None:
            phylum_color.setdefault(phylum, c)
    archaea = sorted(p for p in phylum_color if is_archaea(p))
    bacteria = sorted(p for p in phylum_color if not is_archaea(p))
    handles = []
    if archaea:
        handles.append(Patch(color="none", label=r"$\bf{Archaea}$"))
        handles += [Patch(facecolor=phylum_color[p], label=p) for p in archaea]
    if bacteria:
        handles.append(Patch(color="none", label=r"$\bf{Bacteria}$"))
        handles += [Patch(facecolor=phylum_color[p], label=p) for p in bacteria]
    cm.figure.legend(handles=handles, title="Phylum", title_fontsize=20, fontsize=16, loc="upper right",
                     bbox_to_anchor=(0.07, 1.0), frameon=True, borderaxespad=0.5, handlelength=1.5, handletextpad=0.6)
    for spine in cm.ax_heatmap.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1)

    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    stem = title.lower().replace(" ", "_")
    cm.figure.savefig(HEATMAP_DIR / f"{stem}.png", bbox_inches="tight", dpi=800)
    cm.figure.savefig(HEATMAP_DIR / f"{stem}.svg", bbox_inches="tight")
    print(f"wrote {HEATMAP_DIR / (stem + '.png')}  ({df.shape[0]} ASVs x {df.shape[1]} timepoints)")


def main():
    # full time course, but exclude pre-inoculation samples (day < 0, the Stickney
    # seed inoculum c_A at day -16) so the "Days after inoculation" axis starts at 0.
    sample_days = {s: d for s, d in load_sample_days().items() if d >= 0}
    color_map = load_json(REPO / "iterativeID_color_map.json")
    genera_color_map = {i.split(".")[0]: v for i, v in color_map.items()}
    id_levels = load_json(MODEL_INPUTS / "iterativeID_levels.json")

    total = pd.read_csv(MODEL_INPUTS / "total.csv").set_index("seq")
    total["rel_ab"] = total["rel_ab"] / 100.0
    iterativeIDs = load_json(MODEL_INPUTS / "iterativeIDs.json")
    abundances = load_json(MODEL_INPUTS / "abundances.json")

    # Shannon diversity per timepoint, ordered to match the heatmap columns
    shannon_by_day = {}
    for sample, day in sample_days.items():
        if sample in abundances:
            shannon_by_day[day] = shannon_index(list(abundances[sample].values()))

    # aggregate relative abundance by (day, iterativeID)
    per_day, taxonomies = {}, {}
    for seq, row in total.iterrows():
        day = sample_days.get(row["sample"])
        if day is None:
            continue
        uid = iterativeIDs.get(seq)
        taxonomies.setdefault(uid, "|".join(str(row[l]) for l in TAXO_LEVELS
                                            if TAXO_LEVELS.index(l) <= TAXO_LEVELS.index(GENUS)))
        per_day.setdefault(day, {}).setdefault(uid, 0.0)
        per_day[day][uid] += row["rel_ab"]

    nonzero_per_day = {day: dict(sorted({k: v for k, v in d.items() if v > 0}.items(),
                                        key=lambda kv: kv[1], reverse=True))
                       for day, d in per_day.items()}

    top_union = set()
    for d in nonzero_per_day.values():
        top_union.update(list(d.keys())[:TOP_NUM])

    top_per_day = {day: {org: log10(v) for org, v in d.items() if org in top_union}
                   for day, d in nonzero_per_day.items()}
    taxonomies = {org: t for org, t in taxonomies.items() if org in top_union}

    # columns in strict chronological order (numeric day sort)
    ordered_days = sorted(top_per_day.keys())
    print(f"x-axis day order: {ordered_days}")
    df = DataFrame(top_per_day)[ordered_days].astype(float).replace([inf, -inf], nan)
    taxonomy_series = Series({idx: taxonomies.get(idx, f"Unknown|{idx}") for idx in df.index})

    inlayed = {str(d): shannon_by_day[d] for d in df.columns if d in shannon_by_day}

    build_heatmap(df, taxonomy_series, f"Top {TOP_NUM} ASVs (% abundance)", inlayed,
                  color_map, genera_color_map, id_levels)


if __name__ == "__main__":
    main()

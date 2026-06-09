"""Step 6 - ASV-vs-ASV abundance correlation triangle
(abundance_heatmaps/abundance_correlatons_one_triangle.png).

Ported from the abundance-correlation cell of
../codiffusion_bioreactor/data_processing.py. Spearman-correlates the relative
abundances of the co-occurrence network's organisms across the post-startup
samples, clusters them, shows only the lower triangle, marks p < 0.05 cells with a
green star, and boxes the highlighted methanogens.
"""
import sys
import colorsys
from pathlib import Path

import numpy as np
from numpy import nan, ones, triu, ones_like
import pandas as pd
from pandas import DataFrame, Series
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (REPO, MODEL_INPUTS, HEATMAP_DIR, load_json, load_sample_days,
                     ORGANISMS_TO_HIGHLIGHT, is_archaea, CORR_MIN_DAY,
                     proteo_label, group_color_map)

LEVEL_3 = "Genus"
DEFAULT_COLOR = "lightgray"


def corr_pvalues(df):
    n = df.shape[1]
    pvals = DataFrame(ones((n, n)), index=df.columns, columns=df.columns)
    for i in range(n):
        for j in range(i + 1, n):
            _, p = spearmanr(df.iloc[:, i], df.iloc[:, j])
            pvals.iloc[i, j] = pvals.iloc[j, i] = p
    return pvals


def make(root=False):
    sample_days = load_sample_days(CORR_MIN_DAY)
    iterativeIDs = load_json(MODEL_INPUTS / "iterativeIDs.json")
    iterativeID_color_map = load_json(REPO / "iterativeID_color_map.json")
    genera_color_map = {i.split(".")[0]: v for i, v in iterativeID_color_map.items()}
    id_levels = load_json(MODEL_INPUTS / "iterativeID_levels.json")

    connected = list(load_json(REPO / "significantly_connected_organisms.json"))
    connected.append("Methanobacteriaceae.1")

    abundances = load_json(MODEL_INPUTS / "abundances.json")
    df = DataFrame(abundances).T
    df = df.drop(df.index.difference(sample_days.keys()))
    df.columns = [iterativeIDs.get(c, c) for c in df.columns]
    df = df.drop([c for c in df.columns if c not in connected], axis=1)
    if root:
        # collapse ASVs to their root name (drop the numeric suffix), summing abundances
        df.columns = [c.split(".")[0] for c in df.columns]
        df = df.T.groupby(level=0).sum().T
    df = df.drop([c for c in df.columns if df[c].max() < 0.0005], axis=1)
    print(f"abundance-correlation matrix ({'root' if root else 'ASV'}): "
          f"{df.shape[1]} {'root names' if root else 'ASVs'} x {df.shape[0]} samples")

    corr = df.corr("spearman")
    pval = corr_pvalues(df)

    # taxonomy strings (Kingdom|..|Genus) keyed by the grouping key (iterativeID or root name)
    tax_by_id = {iterativeIDs.get(k, k): v for k, v in load_json(MODEL_INPUTS / "taxonomy.json").items()}
    if root:
        tax_by_id = {iid.split(".")[0]: t for iid, t in tax_by_id.items()}
    taxonomies = {}
    for col in df.columns:
        content = tax_by_id.get(col)
        if isinstance(content, dict):
            taxonomies[col] = "|".join(v for k, v in content.items() if k != "Species" and v is not None)
        else:
            taxonomies[col] = f"Unknown|{col}"

    # display group per ASV: phylum, but Proteobacteria split into its class (Alpha, Gamma),
    # each a distinct shade of the Proteobacteria base colour
    phylum_color_map = load_json(REPO / "Phylum_color_map.json")

    def asv_group(idx):
        parts = str(taxonomies.get(idx, "")).split("|")
        ph = parts[1] if len(parts) > 1 else ""
        if ph in ("", "None", "Unknown", "nan"):
            return None
        return proteo_label(ph, parts[2] if len(parts) > 2 else None)

    proteo_classes = {taxonomies[i].split("|")[2] for i in df.columns
                      if len(taxonomies[i].split("|")) >= 3 and taxonomies[i].split("|")[1] == "Proteobacteria"}
    gcolors = group_color_map(phylum_color_map, proteo_classes)

    def lookup(idx):
        g = asv_group(idx)
        if g and g in gcolors:
            return gcolors[g]
        if idx in iterativeID_color_map:
            return iterativeID_color_map[idx]
        if idx in genera_color_map:
            return genera_color_map[idx]
        return DEFAULT_COLOR

    row_colors = Series({idx: lookup(idx) for idx in corr.index}, name="Phylum")
    col_colors = Series({idx: lookup(idx) for idx in corr.columns}, name="Phylum")

    cm = sns.clustermap(corr, row_colors=row_colors, col_colors=col_colors, cbar_pos=None,
                        cmap="coolwarm_r", center=0, figsize=(78, 91), dendrogram_ratio=(0.1, 0.2))
    cm.figure.subplots_adjust(bottom=0.15, top=0.95)
    cm.ax_row_dendrogram.set_visible(False)
    cm.ax_col_dendrogram.set_visible(False)
    cm.ax_heatmap.yaxis.set_ticks_position("left")
    cm.ax_heatmap.yaxis.set_label_position("left")

    hm_pos = cm.ax_heatmap.get_position()
    fig_w, fig_h = cm.figure.get_figwidth(), cm.figure.get_figheight()
    strip_w = strip_h = 0.015
    cm.ax_row_colors.set_position([hm_pos.x0 - strip_w, hm_pos.y0, strip_w, hm_pos.height])
    cm.ax_col_colors.set_position([hm_pos.x0, hm_pos.y0 - strip_h, hm_pos.width, strip_h])
    cm.ax_heatmap.tick_params(axis="y", pad=strip_w * fig_w * 72 + 15)
    cm.ax_heatmap.tick_params(axis="x", pad=strip_h * fig_h * 72 + 15)

    labelsize = 40
    cm.ax_heatmap.set_yticklabels(cm.ax_heatmap.get_yticklabels(), fontsize=labelsize, rotation=0)
    cm.ax_heatmap.set_xticklabels(cm.ax_heatmap.get_xticklabels(), fontsize=labelsize, rotation=60,
                                  ha="right", rotation_mode="anchor")

    id_level_short = {k.split(".")[0]: v for k, v in id_levels.items()}
    orgs = set()
    for label in cm.ax_heatmap.get_yticklabels():
        t = label.get_text()
        if any(x in t for x in ORGANISMS_TO_HIGHLIGHT):
            orgs.add(t)
            label.set_fontsize(labelsize * 1.2)
            label.set_fontweight("bold")
        if id_levels.get(t) == "Genus":
            label.set_fontstyle("italic")
    for label in cm.ax_heatmap.get_xticklabels():
        t = label.get_text()
        if any(x in t for x in ORGANISMS_TO_HIGHLIGHT):
            orgs.add(t)
            label.set_fontsize(labelsize * 1.2)
            label.set_fontweight("bold")
        if id_level_short.get(t.split(".")[0]) == "Genus":
            label.set_fontstyle("italic")

    drow = cm.dendrogram_row.reordered_ind
    dcol = cm.dendrogram_col.reordered_ind

    # keep only the lower triangle
    df_reordered = corr.iloc[drow, dcol]
    mask = triu(ones_like(df_reordered, dtype=bool), k=1)
    mesh = cm.ax_heatmap.collections[0]
    arr = mesh.get_array().reshape(df_reordered.shape).astype(float)
    arr[mask] = nan
    mesh.set_array(arr.ravel())

    # box the highlighted methanogens (row band + column band, lower-triangle form)
    drow_list, dcol_list = list(drow), list(dcol)
    for org in orgs:
        if org not in corr.index or org not in corr.columns:
            continue
        org_ix = corr.index.get_loc(org)
        if org_ix in drow_list:
            row_pos = drow_list.index(org_ix)
            cm.ax_heatmap.add_patch(patches.Rectangle((0, row_pos), row_pos + 1, 1, linewidth=6,
                                                      edgecolor="black", facecolor="none", clip_on=False))
        col_ix = corr.columns.get_loc(org)
        if col_ix in dcol_list:
            col_pos = dcol_list.index(col_ix)
            cm.ax_heatmap.add_patch(patches.Rectangle((col_pos, len(corr.index)), 1, -(len(corr.index) - col_pos),
                                                      linewidth=6, edgecolor="black", facecolor="none", clip_on=False))

    # significance stars on the lower triangle
    pvals_reordered = pval.iloc[drow, dcol]
    for i in range(pvals_reordered.shape[0]):
        for j in range(pvals_reordered.shape[1]):
            if pvals_reordered.iloc[i, j] < 0.05 and mask[i, j] == 0:
                cm.ax_heatmap.text(j + 0.5, i + 0.5, "*", color="lightgreen", fontsize=60,
                                   fontweight="bold", ha="center", va="center")

    cm.ax_heatmap.set_xlabel("Member ASVs", fontsize=50)
    cm.ax_heatmap.set_ylabel("Member ASVs", fontsize=50)

    # legend by display group (Archaea / Bacteria; Proteobacteria shown as its classes)
    phylum_color, shown = {}, set()
    for idx in corr.index:
        g = asv_group(idx)
        if g is None:
            continue
        phylum_color.setdefault(g, row_colors.get(idx))
        shown.add(g)
    archaea = sorted(p for p in shown if is_archaea(p))
    bacteria = sorted(p for p in shown if not is_archaea(p))

    def header_patch(title):
        return patches.Patch(color="none", label=rf"$\bf{{{title}}}$")

    handles = []
    if archaea:
        handles = [header_patch("Archaea")] + [patches.Patch(color=phylum_color.get(p, DEFAULT_COLOR), label=p) for p in archaea]
    if bacteria:
        handles += [header_patch("Bacteria")] + [patches.Patch(color=phylum_color.get(p, DEFAULT_COLOR), label=p) for p in bacteria]
    fig_width = cm.figure.get_figwidth()
    cm.ax_heatmap.legend(handles=handles, title=LEVEL_3, title_fontsize=8 * (fig_width / 10),
                         loc="lower left", bbox_to_anchor=(0.5, 0.6), fontsize=7 * (fig_width / 10), frameon=True)

    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    out = HEATMAP_DIR / f"abundance_correlatons_one_triangle{'_root' if root else ''}.png"
    cm.figure.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    make(root=False)   # ASV level
    make(root=True)    # collapsed to root names

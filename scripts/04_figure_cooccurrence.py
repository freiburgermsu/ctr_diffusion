"""Step 4 - co-occurrence network (cooccurrence_network_p_value_FDR.png).

Ported from the co-occurrence cell of ../codiffusion_bioreactor/data_processing.py.
Edges connect ASV pairs whose abundances are Spearman-correlated across samples and
survive Benjamini-Hochberg FDR (q < 0.05); node size = mean relative abundance,
node colour = phylum.

Also writes artefacts consumed by later steps:
    Phylum_color_map.json
    iterativeID_color_map.json
    significantly_connected_organisms.json   (graph node list)
"""
import sys
from json import dump
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import networkx as nx
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (REPO, MODEL_INPUTS, load_json, load_sample_days, is_archaea,
                     CORR_MIN_DAY)

LEVEL = "Phylum"


def main():
    sample_days = load_sample_days(CORR_MIN_DAY)
    print(f"co-occurrence sample window (day >= {CORR_MIN_DAY}): "
          f"{len(sample_days)} samples -> {list(sample_days)}")
    iterativeIDs = load_json(MODEL_INPUTS / "iterativeIDs.json")
    taxonomy = load_json(MODEL_INPUTS / "taxonomy.json")

    iterativeID_taxonomy = {iterativeIDs.get(s, s): taxonomy.get(s, {}) for s in iterativeIDs}
    iterativeID_level = {i: (c.get(LEVEL) if isinstance(c, dict) else None)
                         for i, c in iterativeID_taxonomy.items()}

    archaea_phyla = sorted({v for v in iterativeID_level.values() if v and is_archaea(v)})
    bacteria_phyla = sorted({v for v in iterativeID_level.values() if v and not is_archaea(v)})
    n_a, n_b = len(archaea_phyla), len(bacteria_phyla)
    taxa_color_map = {}
    for i, p in enumerate(archaea_phyla):
        taxa_color_map[p] = plt.cm.turbo(i / max(n_a, 1) * 0.15)
    for i, p in enumerate(bacteria_phyla):
        taxa_color_map[p] = plt.cm.turbo(0.2 + i / max(n_b, 1) * 0.8)

    # serialise colours (RGBA tuples -> lists) for downstream steps
    dump({k: list(v) for k, v in taxa_color_map.items()},
         open(REPO / f"{LEVEL}_color_map.json", "w"))
    iterativeID_color_map = {i: list(taxa_color_map[p]) for i, p in iterativeID_level.items() if p}
    dump(iterativeID_color_map, open(REPO / "iterativeID_color_map.json", "w"))

    # abundance matrix: samples x ASV (canonical sample set)
    abundances = load_json(MODEL_INPUTS / "abundances.json")
    df = pd.DataFrame(abundances).T
    df = df.drop(df.index.difference(sample_days.keys()))
    df = df.loc[:, (df.fillna(0) > 0).sum() >= 3]
    df.columns = [iterativeIDs.get(c, c) for c in df.columns]

    rel = df.div(df.sum(axis=1), axis=0)
    mean_rel_abund = rel.mean(axis=0)
    presence = (df > 0).astype(int)

    cooccurrence = defaultdict(int)
    for sample in presence.itertuples(index=False):
        present = [col for col, val in zip(presence.columns, sample) if val]
        for pair in combinations(sorted(present), 2):
            cooccurrence[pair] += 1

    pair_data = []
    for (a, b), count in cooccurrence.items():
        rho, p = spearmanr(df[a], df[b])
        if np.isnan(rho):
            continue
        pair_data.append((a, b, rho, p, count))

    pvals = np.array([t[3] for t in pair_data])
    reject, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

    G = nx.Graph()
    for (a, b, rho, p, count), q, keep in zip(pair_data, qvals, reject):
        if keep:
            G.add_edge(a, b, weight=abs(rho), rho=rho, pvalue=p, qvalue=q, cooccurrence=count)
    print(f"tests run: {len(pair_data)} | FDR-significant pairs: {int(reject.sum())}")
    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    pos = nx.spring_layout(G, seed=42, iterations=100, k=0.3)
    edges = G.edges(data=True)
    rho_values = [d["rho"] for _, _, d in edges]
    edge_widths = [3 * d["weight"] for _, _, d in edges]
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cmap = plt.get_cmap("coolwarm_r")
    edge_colors = [cmap(norm(r)) for r in rho_values]

    width, height = 40, 30
    fig, ax = plt.subplots(figsize=(width, height))
    scale = 5000 * (width / 10) * 2
    compressor = np.sqrt
    node_sizes = [scale * compressor(mean_rel_abund.get(n, 0)) for n in G.nodes()]
    node_colors = [taxa_color_map.get(iterativeID_level.get(n, "Unknown"), "lightgray") for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.85, ax=ax)

    significantly_connected = list(pos.keys())
    dump(significantly_connected, open(REPO / "significantly_connected_organisms.json", "w"))

    LABEL_MIN_ABUND = 0.002
    for n, (x, y) in pos.items():
        abund = mean_rel_abund.get(n, 0)
        if abund < LABEL_MIN_ABUND:
            continue
        fs = 6 + 14 * compressor(abund) / compressor(mean_rel_abund.max()) * (width / 10)
        fs = max(5, min(fs, 48))
        ax.text(x, y, str(n), fontsize=fs, color="black", fontweight="bold", ha="center", va="center")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Spearman $\\rho$", fontsize=10 * (width / 10))
    cbar.ax.tick_params(labelsize=8 * (width / 10), length=8, width=2)

    # mean-abundance size legend
    legend_entries = [(0.002, "0.2%"), (0.02, "2%"), (0.2, "20%")]
    sizes_pt2 = [scale * compressor(a) for a, _ in legend_entries]
    diameters_pt = [2 * np.sqrt(s / np.pi) for s in sizes_pt2]
    offsets_pt = [0.0]
    for i in range(1, len(legend_entries)):
        offsets_pt.append(offsets_pt[-1] + (diameters_pt[i - 1] + diameters_pt[i]) / 2 + 8)
    fig_h_pts = fig.get_figheight() * 72
    top_y, circle_x, label_x = 0.9, 0.85, 0.88
    ax.text(circle_x, top_y + 0.025, "Mean rel. abundance", transform=fig.transFigure,
            va="bottom", fontweight="bold", fontsize=6 * (width / 10), clip_on=False)
    for (abund, label), s, off in zip(legend_entries, sizes_pt2, offsets_pt):
        y = top_y - off / fig_h_pts
        ax.scatter([circle_x], [y], s=s, color="slategray", alpha=0.9, transform=fig.transFigure, clip_on=False)
        ax.text(label_x, y, label, transform=fig.transFigure, va="center", fontsize=5 * (width / 10), clip_on=False)

    # phylum legend (uses the same phylum colour map)
    def header_patch(title):
        return mpatches.Patch(color="none", label=f"$\\bf{{{title}}}$")
    archaea_patches = [mpatches.Patch(color=taxa_color_map[p], label=p) for p in archaea_phyla if p in taxa_color_map]
    bacteria_patches = [mpatches.Patch(color=taxa_color_map[p], label=p) for p in bacteria_phyla if p in taxa_color_map]
    handles = [header_patch("Archaea")] + archaea_patches + [header_patch("Bacteria")] + bacteria_patches
    ax.legend(handles=handles, title="Taxonomic " + LEVEL, title_fontsize=8 * (width / 10),
              loc="lower left", bbox_to_anchor=(-0.24, 0.1), fontsize=7 * (width / 10), frameon=True)
    ax.axis("off")
    plt.tight_layout()
    out = REPO / "cooccurrence_network_p_value_FDR.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

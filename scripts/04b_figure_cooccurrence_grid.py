"""Step 4b - co-occurrence network, GRID layout with Louvain modularity
(cooccurrence_network_p_value_FDR_grid_layout.png).

Emulates ../codiffusion_bioreactor's cooccurrence_network_p_value_FDR_grid_layout.png:
Louvain community detection on the positive-rho subgraph, modules laid out on a
rotated grid with convex-hull shading and a modularity summary box.

Parameters tuned for this dataset (see scripts/README.md):
    RESOLUTION = 1.0       Louvain resolution
    MIN_MODULE_SIZE = 5    a "multi-organismal module" must have >= 5 ASVs
This yields 8 substantial modules (sizes 32, 24, 14, 10, 6, 5, 5, 5); the many
incidental co-occurring pairs/triplets and negative-only isolates are placed on the
periphery rather than shaded as modules.
"""
import sys
import math
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
import matplotlib.patheffects as path_effects
import networkx as nx
from scipy.stats import spearmanr
from scipy.spatial import ConvexHull
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, MODEL_INPUTS, load_json, load_sample_days, is_archaea, CORR_MIN_DAY

LEVEL = "Phylum"
RESOLUTION = 1.0       # Louvain resolution -> 8 modules of size >= MIN_MODULE_SIZE
MIN_MODULE_SIZE = 5    # ASVs per "multi-organismal module"


def louvain(graph, resolution):
    try:
        return list(nx.community.louvain_communities(graph, weight="weight", resolution=resolution, seed=42))
    except AttributeError:
        return list(nx.algorithms.community.greedy_modularity_communities(graph, weight="weight"))


def modularity(graph, comms):
    try:
        return nx.community.modularity(graph, comms, weight="weight")
    except AttributeError:
        return nx.algorithms.community.modularity(graph, comms, weight="weight")


def main():
    sample_days = load_sample_days(CORR_MIN_DAY)
    iterativeIDs = load_json(MODEL_INPUTS / "iterativeIDs.json")
    taxonomy = load_json(MODEL_INPUTS / "taxonomy.json")

    iterativeID_level = {iterativeIDs.get(s, s): (taxonomy.get(s) or {}).get(LEVEL) for s in iterativeIDs}
    archaea_phyla = sorted({v for v in iterativeID_level.values() if v and is_archaea(v)})
    bacteria_phyla = sorted({v for v in iterativeID_level.values() if v and not is_archaea(v)})
    taxa_color_map = {}
    for i, p in enumerate(archaea_phyla):
        taxa_color_map[p] = plt.cm.turbo(i / max(len(archaea_phyla), 1) * 0.15)
    for i, p in enumerate(bacteria_phyla):
        taxa_color_map[p] = plt.cm.turbo(0.2 + i / max(len(bacteria_phyla), 1) * 0.8)

    # --- build FDR co-occurrence graph (day >= CORR_MIN_DAY), same as step 4 ---
    df = pd.DataFrame(load_json(MODEL_INPUTS / "abundances.json")).T
    df = df.drop(df.index.difference(sample_days.keys()))
    df = df.loc[:, (df.fillna(0) > 0).sum() >= 3]
    df.columns = [iterativeIDs.get(c, c) for c in df.columns]
    mean_rel_abund = df.div(df.sum(axis=1), axis=0).mean(axis=0)

    presence = (df > 0).astype(int)
    cooc = defaultdict(int)
    for sample in presence.itertuples(index=False):
        present = [c for c, v in zip(presence.columns, sample) if v]
        for pair in combinations(sorted(present), 2):
            cooc[pair] += 1
    pair_data = []
    for (a, b), count in cooc.items():
        rho, p = spearmanr(df[a], df[b])
        if not np.isnan(rho):
            pair_data.append((a, b, rho, p, count))
    reject = multipletests(np.array([t[3] for t in pair_data]), alpha=0.05, method="fdr_bh")[0]
    G = nx.Graph()
    for (a, b, rho, p, count), keep in zip(pair_data, reject):
        if keep:
            G.add_edge(a, b, weight=abs(rho), rho=rho, pvalue=p, cooccurrence=count)
    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # --- Louvain on the positive-rho subgraph ---
    pos_sub = G.edge_subgraph([(u, v) for u, v, d in G.edges(data=True) if d["rho"] > 0]).copy()
    comms = louvain(pos_sub, RESOLUTION)
    Q = modularity(pos_sub, comms) if comms else 0.0

    modules = sorted((c for c in comms if len(c) >= MIN_MODULE_SIZE), key=len, reverse=True)
    minor = [c for c in comms if len(c) < MIN_MODULE_SIZE]
    placed = {n for c in modules for n in c}
    periphery = [n for n in G.nodes() if n not in placed]
    print(f"Louvain @res={RESOLUTION}: Q={Q:.3f}; {len(modules)} modules (size>={MIN_MODULE_SIZE}) "
          f"sizes={[len(c) for c in modules]}; {len(periphery)} peripheral nodes")

    # module-membership export
    export = {}
    for i, c in enumerate(modules, 1):
        export[f"module_{i}"] = {"size": len(c), "type": "louvain", "members": sorted(c)}
    for i, c in enumerate(minor, 1):
        export[f"minor_{i}"] = {"size": len(c), "type": "minor", "members": sorted(c)}
    dump(export, open(REPO / "network_module_membership.json", "w"), indent=2)

    # --- per-module spring layout, then rotated grid placement ---
    np.random.seed(42)
    SCALE_FACTOR = 35.0
    INTRA_AMP = 5.4 * 3
    module_layouts = {}
    for mi, comm in enumerate(modules):
        sub = G.subgraph(comm).copy()
        module_scale = len(comm) * SCALE_FACTOR
        if sub.number_of_edges() == 0:
            module_layouts[mi] = {n: np.random.normal(0, module_scale * 0.3, 2) for n in comm}
        else:
            module_layouts[mi] = nx.spring_layout(sub, seed=42, iterations=5000, k=180.0,
                                                  scale=module_scale, weight=None)
        for n in module_layouts[mi]:
            module_layouts[mi][n] = module_layouts[mi][n] * INTRA_AMP

    max_module_radius = max((max(np.linalg.norm(p) for p in ml.values()) for ml in module_layouts.values()),
                            default=1.0)
    grid_n = int(math.ceil(math.sqrt(len(modules))))
    cell_size = 2 * max_module_radius * (1.0 / 1.05)   # modules ~5% larger than their grid cells
    v_stretch, right_v_extra = 1.6, 1.4
    center_left_shift = -0.85 * cell_size

    pos = {}
    sorted_mis = sorted(module_layouts.keys(), key=lambda mi: -len(modules[mi]))
    for i, mi in enumerate(sorted_mis):
        pre_row, pre_col = i // grid_n, i % grid_n
        pre_col = {0: 1, 1: 0}.get(pre_col, pre_col)
        pre_x = (pre_col - (grid_n - 1) / 2.0) * cell_size
        pre_y = (pre_row - (grid_n - 1) / 2.0) * cell_size
        cx, cy = -pre_y, pre_x * v_stretch
        if abs(cx) < 0.5 * cell_size:
            cx += center_left_shift
        elif cx > 0.5 * cell_size:
            cy *= right_v_extra
        center = np.array([cx, cy])
        for n, local in module_layouts[mi].items():
            pos[n] = center + np.array([-local[1], local[0]])

    if periphery:
        xext = cell_size + max_module_radius
        yext = (v_stretch * right_v_extra) * cell_size + max_module_radius
        ring = max(xext, yext) + max_module_radius * 0.2
        for i, n in enumerate(periphery):
            theta = 2 * np.pi * (i + 0.5) / len(periphery)
            pos[n] = np.array([np.cos(theta), np.sin(theta)]) * ring

    # --- render ---
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cmap = plt.get_cmap("coolwarm_r")
    edges = G.edges(data=True)
    edge_widths = [3 * d["weight"] for _, _, d in edges]
    edge_colors = [cmap(norm(d["rho"])) for _, _, d in edges]

    width, height = 40, 30
    fig, ax = plt.subplots(figsize=(width, height))
    scale = 5000 * (width / 10) * 2
    compressor = np.sqrt
    node_sizes = [scale * compressor(mean_rel_abund.get(n, 0)) for n in G.nodes()]
    node_colors = [taxa_color_map.get(iterativeID_level.get(n, "Unknown"), "lightgray") for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.85, ax=ax)

    LABEL_MIN_ABUND = 0.002
    for n, (x, y) in pos.items():
        abund = mean_rel_abund.get(n, 0)
        if abund < LABEL_MIN_ABUND:
            continue
        fs = max(5, min(6 + 14 * compressor(abund) / compressor(mean_rel_abund.max()) * (width / 10), 48))
        txt = ax.text(x, y, str(n), fontsize=fs, color="black", fontweight="bold", ha="center", va="center",
                      zorder=20 if n == "Methanobacteriaceae.1" else 5)
        txt.set_path_effects([path_effects.Stroke(linewidth=max(1.5, fs / 6), foreground="white"),
                              path_effects.Normal()])

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(r"Spearman $\rho$", fontsize=10 * (width / 10))
    cbar.ax.tick_params(labelsize=8 * (width / 10), length=8, width=2)

    legend_entries = [(0.002, "0.2%"), (0.02, "2%"), (0.20, "20%")]
    sizes_pt2 = [scale * compressor(a) for a, _ in legend_entries]
    diameters_pt = [2 * np.sqrt(s / np.pi) for s in sizes_pt2]
    offsets_pt = [0.0]
    for i in range(1, len(legend_entries)):
        offsets_pt.append(offsets_pt[-1] + (diameters_pt[i - 1] + diameters_pt[i]) / 2 + 8)
    fig_h_pts = fig.get_figheight() * 72
    ax.text(0.85, 0.925, "Mean rel. abundance", transform=fig.transFigure, va="bottom",
            fontweight="bold", fontsize=6 * (width / 10), clip_on=False)
    for (abund, label), s, off in zip(legend_entries, sizes_pt2, offsets_pt):
        y = 0.9 - off / fig_h_pts
        ax.scatter([0.85], [y], s=s, color="slategray", alpha=0.9, transform=fig.transFigure, clip_on=False)
        ax.text(0.88, y, label, transform=fig.transFigure, va="center", fontsize=5 * (width / 10), clip_on=False)

    def header_patch(title):
        return mpatches.Patch(color="none", label=rf"$\bf{{{title}}}$")
    archaea_patches = [mpatches.Patch(color=taxa_color_map[p], label=p) for p in archaea_phyla if p in taxa_color_map]
    bacteria_patches = [mpatches.Patch(color=taxa_color_map[p], label=p) for p in bacteria_phyla if p in taxa_color_map]
    ax.legend(handles=[header_patch("Archaea")] + archaea_patches + [header_patch("Bacteria")] + bacteria_patches,
              title="Taxonomic " + LEVEL, title_fontsize=8 * (width / 10), loc="lower left",
              bbox_to_anchor=(-0.24, 0.1), fontsize=7 * (width / 10), frameon=True)
    ax.axis("off")
    plt.tight_layout()

    # --- convex-hull shading of the modules (SAT-shrink to remove overlaps) ---
    palette = plt.cm.tab10(np.linspace(0, 1, 10))

    def polys_intersect(p1, p2):
        for poly, other in ((p1, p2), (p2, p1)):
            for i in range(len(poly)):
                edge = poly[(i + 1) % len(poly)] - poly[i]
                normal = np.array([-edge[1], edge[0]])
                if (poly @ normal).max() < (other @ normal).min() or (other @ normal).max() < (poly @ normal).min():
                    return False
        return True

    hull_data = {}
    for mi, comm in enumerate(modules):
        pts = np.array([pos[n] for n in comm if n in pos])
        centroid = pts.mean(axis=0)
        if len(pts) >= 3:
            verts = pts[ConvexHull(pts).vertices]
        else:
            r = max(0.05, np.linalg.norm(pts - centroid, axis=1).max())
            theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
            verts = centroid + r * np.column_stack([np.cos(theta), np.sin(theta)])
        hull_data[mi] = [centroid, verts, 1.30]

    for _ in range(120):
        overlap = False
        ids = list(hull_data)
        for ia in range(len(ids)):
            ca, va, sa = hull_data[ids[ia]]
            poly_a = ca + (va - ca) * sa
            for ib in range(ia + 1, len(ids)):
                cb, vb, sb = hull_data[ids[ib]]
                poly_b = cb + (vb - cb) * sb
                if polys_intersect(poly_a, poly_b):
                    overlap = True
                    for k in (ids[ia], ids[ib]):
                        if hull_data[k][2] > 0.75:
                            hull_data[k][2] = max(0.75, hull_data[k][2] * 0.95)
        if not overlap:
            break

    for mi, (ca, verts, sv) in hull_data.items():
        ax.add_patch(mpatches.Polygon(ca + (verts - ca) * sv, closed=True, facecolor=palette[mi % 10],
                                      edgecolor=palette[mi % 10], linewidth=2, alpha=0.22, zorder=0))

    summary = (r"$\bf{Modularity}$" + "\n" + f"Q = {Q:.3f}" + "\n"
               + f"modules = {len(modules)} (size >= {MIN_MODULE_SIZE})" + "\n"
               + f"sizes: {[len(c) for c in modules]}" + "\n"
               + f"peripheral nodes = {len(periphery)}")
    ax.text(-0.10, 0.7, summary, transform=ax.transAxes, fontsize=8 * (width / 10), ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.85, linewidth=1.5))

    out = REPO / "cooccurrence_network_p_value_FDR_grid_layout.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"wrote {REPO / 'network_module_membership.json'}")


if __name__ == "__main__":
    main()

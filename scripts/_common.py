"""Shared constants and helpers for the ctr_diffusion figure pipeline.

This pipeline reproduces, for the ctr_diffusion dataset, the figures that exist
in ../codiffusion_bioreactor:
    - cooccurrence_network_p_value_FDR.png
    - abundance_heatmaps/top_10_asvs_(%_abundance).png
    - abundance_heatmaps/abundance_correlatons_one_triangle.png
    - correlations_heatmap_reduced.png   (all operational params x ASV rel-abundance)

All inputs come from this repo only:
    - Ctrdif qiime.zip        -> relative abundance + taxonomy (extracted to qiime_raw/)
    - ctr_dif operations TEST.xlsx -> operational parameters ('Summary' sheet)

The canonical microbiome sample set is ONE sample per unique sampling day.
Where a day has both a suspended and a biofilm sample (e.g. c_Q and c_Qb), the
suspended sample (shortest name) is kept so abundances are never double counted.
"""
from pathlib import Path
from json import load
import numpy as np
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent
QIIME = REPO / "qiime_raw" / "Ctrdif"
MODEL_INPUTS = REPO / "model_inputs"
MEASUREMENTS = MODEL_INPUTS / "measurements"
CORR_DIR = REPO / "modeling_files" / "correlations"
HEATMAP_DIR = REPO / "abundance_heatmaps"

# raw qiime files (extracted from Ctrdif qiime.zip)
WIDE_REL = QIIME / "table_rel_ctrdif_output_050426.csv"   # seq x sample, percentages
LONG_TAX = QIIME / "table1_ctrdif_output_050426.csv"      # 1 row / seq, full taxonomy
META_XLSX = QIIME / "16s_metadata_ctrdif_050426.xlsx"     # sample -> Day, Description
META_SHEET = "16s_metadata_2026"

OPERATIONS_XLSX = REPO / "ctr_dif operations TEST.xlsx"
SUMMARY_SHEET = "Summary"

TAXO_LEVELS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]

# Post-startup cutoff (days) for the correlation-based figures (co-occurrence,
# abundance correlation, operational-parameter correlation). Excludes the ~150-day
# startup/acclimation so figure density matches the reference. The top-10 ASV
# heatmap deliberately ignores this and shows the full time course.
CORR_MIN_DAY = 150

ORGANISMS_TO_HIGHLIGHT = ["Methanobacterium", "Methanosarcina", "Methanobacteriaceae"]
ARCHAEA_MARKERS = ("archaeo", "euryarchaeota", "crenarchaeota", "thaumarchaeota",
                   "candidatus thermoplasmatota", "halobacterota", "methanobacteriota",
                   "micrarchaeota", "nanoarchaeota")


def ensure_dirs():
    for d in (MODEL_INPUTS, MEASUREMENTS, CORR_DIR, HEATMAP_DIR):
        d.mkdir(parents=True, exist_ok=True)


def is_archaea(phylum):
    p = str(phylum).lower()
    return any(m in p for m in ARCHAEA_MARKERS)


def load_json(path):
    with open(path) as fh:
        return load(fh)


def load_sample_days(min_day=None):
    """{sample: day} for the canonical one-sample-per-day set, ordered by day.

    Pass ``min_day`` (e.g. CORR_MIN_DAY) to restrict to the post-startup window.
    """
    d = load_json(MODEL_INPUTS / "sample_days.json")
    if min_day is not None:
        d = {s: day for s, day in d.items() if day >= min_day}
    return dict(sorted(d.items(), key=lambda kv: kv[1]))


def taxonomy_linkage(taxonomy_series):
    """Build a linkage matrix that EXACTLY follows the taxonomy hierarchy.

    Ported verbatim from ../codiffusion_bioreactor/data_processing.py so the
    top-10 ASV heatmap clusters rows by taxonomy rather than by abundance.
    """
    parsed = (taxonomy_series.fillna("unknown").astype(str)
              .str.split("[;|,]", regex=True)
              .apply(lambda lst: [x.strip() for x in lst or [] if x.strip()]))
    ranks_n = max((len(p) for p in parsed))
    if ranks_n == 0:
        ranks_n = 1
    parsed = parsed.apply(lambda lst: (lst + [""] * ranks_n)[:ranks_n])
    n = len(parsed)
    if n <= 1:
        return np.empty((0, 4), dtype=float)
    idx_to_pos = {idx: pos for pos, idx in enumerate(parsed.index)}
    linkage_rows = []
    next_cluster_id = n

    def build_subtree(indices, depth):
        nonlocal next_cluster_id
        if len(indices) == 1:
            return (idx_to_pos[indices[0]], 1)
        if depth >= ranks_n:
            merge_distance = 0.5
            cluster_id, total_count = (idx_to_pos[indices[0]], 1)
            for idx in indices[1:]:
                new_id = next_cluster_id
                next_cluster_id += 1
                linkage_rows.append([cluster_id, idx_to_pos[idx], merge_distance, total_count + 1])
                cluster_id = new_id
                total_count += 1
            return (cluster_id, total_count)
        groups = defaultdict(list)
        for idx in indices:
            rank_value = parsed.loc[idx][depth]
            if rank_value == "":
                rank_value = f"__unclassified_{idx}"
            groups[rank_value].append(idx)
        subclusters = []
        for rank_value, group_indices in groups.items():
            sub_id, sub_count = build_subtree(group_indices, depth + 1)
            subclusters.append((sub_id, sub_count))
        if len(subclusters) == 1:
            return subclusters[0]
        merge_distance = float(ranks_n - depth)
        cluster_id, total_count = subclusters[0]
        for sub_id, sub_count in subclusters[1:]:
            new_id = next_cluster_id
            next_cluster_id += 1
            linkage_rows.append([cluster_id, sub_id, merge_distance, total_count + sub_count])
            cluster_id = new_id
            total_count += sub_count
        return (cluster_id, total_count)

    build_subtree(list(parsed.index), depth=0)
    Z = np.array(linkage_rows, dtype=float)
    if len(Z) > 0:
        for i in range(1, len(Z)):
            if Z[i, 2] < Z[i - 1, 2]:
                Z[i, 2] = Z[i - 1, 2]
    return Z

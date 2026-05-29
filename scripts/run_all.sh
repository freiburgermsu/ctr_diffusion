#!/usr/bin/env bash
# Regenerate every figure from the raw ctr_diffusion data, end to end.
# Steps must run in order: step 4 (co-occurrence) writes the colour maps and the
# significantly-connected-organism list that steps 5 and 6 consume.
set -euo pipefail

PY="$HOME/Documents/py_venv/bin/python"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
cd "$REPO"

# extract the qiime archive if it has not been unpacked yet
if [ ! -d "qiime_raw/Ctrdif" ]; then
  echo "[0/7] extracting Ctrdif qiime.zip"
  unzip -o -q "Ctrdif qiime.zip" -d qiime_raw
fi

echo "[1/7] build canonical inputs";              "$PY" "$HERE/01_build_inputs.py"
echo "[2/7] build operational parameters";        "$PY" "$HERE/02_build_operational.py"
echo "[3/7] correlate operational params x ASVs"; "$PY" "$HERE/03_correlate_all_params.py"
echo "[4/8] co-occurrence network (spring)";       "$PY" "$HERE/04_figure_cooccurrence.py"
echo "[4b/8] co-occurrence network (grid+louvain)";"$PY" "$HERE/04b_figure_cooccurrence_grid.py"
echo "[5/8] top-10 ASV heatmap";                   "$PY" "$HERE/05_figure_top10_heatmap.py"
echo "[6/8] abundance correlation triangle";       "$PY" "$HERE/06_figure_abundance_correlation.py"
echo "[7/8] operational correlation heatmap";      "$PY" "$HERE/07_figure_operational_correlation.py"

echo "done. figures:"
echo "  cooccurrence_network_p_value_FDR.png"
echo "  cooccurrence_network_p_value_FDR_grid_layout.png"
echo "  abundance_heatmaps/top_10_asvs_(%_abundance).png"
echo "  abundance_heatmaps/abundance_correlatons_one_triangle.png"
echo "  correlations_heatmap_reduced.png"

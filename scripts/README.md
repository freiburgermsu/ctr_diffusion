# ctr_diffusion figure pipeline

Reproduces, for the **ctr_diffusion** dataset, four figures that exist in
`../codiffusion_bioreactor`, using only data in this repo:

| figure | output path |
|---|---|
| Co-occurrence network (spring layout) | `cooccurrence_network_p_value_FDR.png` |
| Co-occurrence network (grid layout + Louvain modules) | `cooccurrence_network_p_value_FDR_grid_layout.png` |
| Top-10 ASV abundance heatmap | `abundance_heatmaps/top_10_asvs_(%_abundance).png` |
| ASV-vs-ASV abundance correlation triangle | `abundance_heatmaps/abundance_correlatons_one_triangle.png` |
| All operational params vs ASV abundances | `correlations_heatmap_reduced.png` |

## Inputs (this repo only)

- `Ctrdif qiime.zip` &rarr; relative abundance + taxonomy (extracted to `qiime_raw/`)
  - `table_rel_ctrdif_output_050426.csv` — wide relative abundance (% per sample)
  - `table1_ctrdif_output_050426.csv` — per-ASV taxonomy
  - `16s_metadata_ctrdif_050426.xlsx` — sample &rarr; Day
- `ctr_dif operations TEST.xlsx` &rarr; operational parameters (`Summary` sheet)

## Run

```bash
bash scripts/run_all.sh        # uses ~/Documents/py_venv/bin/python
```

Steps are numbered and must run in order — step 4 writes `iterativeID_color_map.json`,
`Phylum_color_map.json`, and `significantly_connected_organisms.json`, which steps 5
and 6 consume.

## Key choices

- **Canonical sample set** — one microbiome sample per sampling day. Where a day has
  both a suspended and a biofilm sample (e.g. `c_Q` / `c_Qb`), the suspended sample is
  kept so relative abundances are never double counted (27 timepoints, day −16…349).
- **Correlation window** — the three correlation-based figures (co-occurrence,
  abundance triangle, operational correlation) use only **day ≥ 150** (14 samples),
  excluding the ~150-day startup/acclimation. This makes network density comparable to
  the reference (≈152 vs 118 nodes). The cutoff is `CORR_MIN_DAY` in `_common.py`.
  The **top-10 heatmap uses the full time course** but **excludes pre-inoculation
  samples** (day < 0, the Stickney seed inoculum `c_A` at day −16) so the
  "Days after inoculation" axis starts at day 0.
- **Grid-layout co-occurrence** (`04b`) — Louvain community detection on the
  positive-ρ subgraph (`RESOLUTION = 1.0`, seed 42); communities with `MIN_MODULE_SIZE
  = 5` ASVs are laid out on a rotated grid with convex-hull shading. These parameters
  give **8 multi-organismal modules** (sizes 32, 24, 14, 10, 6, 5, 5, 5; Q = 0.642);
  smaller co-occurring pairs/triplets and negative-only isolates sit on the periphery.
  Membership is written to `network_module_membership.json`.
- **All-vs-all operational correlation** — unlike the reference (which wired in six
  hand-picked metrics) every numeric `Summary` parameter with ≥5 paired samples is
  correlated against every ASV present in ≥5 samples (Spearman), with Benjamini-Hochberg
  FDR per parameter; all-blank ASV rows **and** all-blank parameter columns are dropped.

## Intermediate files (written under this repo)

```
model_inputs/  sample_days.json, abundances.json, taxonomy.json, iterativeIDs.json,
               iterativeID_levels.json, total.csv,
               measurements/{Summary_interpolated.csv, summary_samples.json, operational_params.json}
modeling_files/correlations/  ASV_correlations_all_relevant_samples_<param>.json  (one per parameter)
iterativeID_color_map.json, Phylum_color_map.json, significantly_connected_organisms.json
```

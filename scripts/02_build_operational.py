"""Step 2 - build per-sample operational parameters from the operations xlsx.

The reference repo keyed operational data by a 'Biomass Sample ID' column; the
ctr_diffusion 'Summary' sheet has no such column, so we link operational data to
microbiome samples by DAY: each canonical sample's day is matched to the nearest
'Days of Operation' row, after linearly interpolating internal gaps in every
numeric column (mirrors the reference's Summary interpolation).

Reads:
    ctr_dif operations TEST.xlsx  ('Summary' sheet, real header on row 2)
    model_inputs/sample_days.json

Writes:
    model_inputs/measurements/Summary_interpolated.csv
    model_inputs/measurements/summary_samples.json   {sample: {param: value}}
    model_inputs/measurements/operational_params.json [param, ...]  (numeric params)
"""
import sys
from json import dump
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (ensure_dirs, MEASUREMENTS, OPERATIONS_XLSX, SUMMARY_SHEET,
                     load_sample_days)

DAY_COL = "Days of Operation"
# columns that are not real operational parameters
DROP_COLS = {"Start Date", "Unnamed: 2"}


def main():
    ensure_dirs()
    sample_days = load_sample_days()

    summary = pd.read_excel(OPERATIONS_XLSX, sheet_name=SUMMARY_SHEET, header=1, engine="openpyxl")
    # drop the stray datetime-named column and known non-parameter columns
    drop = [c for c in summary.columns if c in DROP_COLS or not isinstance(c, str)]
    summary = summary.drop(columns=drop)
    summary = summary[summary[DAY_COL].notna()].copy()
    summary = summary.sort_values(DAY_COL).reset_index(drop=True)

    # numeric parameter columns (exclude the day axis itself and all-empty columns)
    param_cols = [c for c in summary.columns
                  if c != DAY_COL and pd.api.types.is_numeric_dtype(summary[c])
                  and summary[c].notna().any()]

    # interpolate internal gaps so any day can be sampled (reference behaviour)
    for c in param_cols:
        s = summary[c]
        if s.isna().any() and not s.isna().all():
            summary[c] = s.interpolate(method="linear", limit_direction="forward", limit_area="inside")

    summary.to_csv(MEASUREMENTS / "Summary_interpolated.csv", index=False)
    print(f"Summary: {summary.shape[0]} day-rows, {len(param_cols)} numeric parameters")

    doo = summary[DAY_COL].to_numpy(dtype=float)

    summary_samples = {}
    matched = []
    for sample, day in sample_days.items():
        if day < 0:
            continue  # inoculum predates operation -> no operational data
        i = int(np.abs(doo - day).argmin())
        row = summary.iloc[i]
        vals = {}
        for c in param_cols:
            v = row[c]
            if pd.notna(v):
                vals[c] = float(v)
        summary_samples[sample] = vals
        matched.append((sample, day, round(float(doo[i]), 2), len(vals)))

    dump(summary_samples, open(MEASUREMENTS / "summary_samples.json", "w"), indent=2)
    dump(param_cols, open(MEASUREMENTS / "operational_params.json", "w"), indent=2)

    print(f"wrote summary_samples.json ({len(summary_samples)} samples with operational data)")
    print(f"{'sample':8s} {'day':>5s} {'nearest DoO':>12s} {'n_params':>9s}")
    for s, d, doo_match, n in matched:
        print(f"{s:8s} {d:5d} {doo_match:12.2f} {n:9d}")


if __name__ == "__main__":
    main()

"""Step 1 - build canonical model_inputs/* from the ctr_diffusion qiime data.

Adapted from ../codiffusion_bioreactor/_rebuild_inputs_050526.py.

Reads (this repo only):
    qiime_raw/Ctrdif/table_rel_ctrdif_output_050426.csv   wide rel-abundance %, seq x sample
    qiime_raw/Ctrdif/table1_ctrdif_output_050426.csv       1 row / seq, full taxonomy
    qiime_raw/Ctrdif/16s_metadata_ctrdif_050426.xlsx       sample -> Day, Description

Writes:
    model_inputs/sample_days.json          {sample: day}  (one sample per day, suspended preferred)
    model_inputs/abundances.json           {sample: {seq: rel_ab_fraction}}   (ALL samples)
    model_inputs/taxonomy.json             {seq: {Kingdom..Species}}
    model_inputs/iterativeIDs.json         {seq: "<lowest_named_taxon>.<N>"}
    model_inputs/iterativeID_levels.json   {"<name>.<N>": "<rank>"}
    model_inputs/total.csv                 long: seq, sample, rel_ab(%), taxonomy, meta
"""
import sys
from json import dump
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import (ensure_dirs, MODEL_INPUTS, WIDE_REL, LONG_TAX, META_XLSX,
                     META_SHEET, TAXO_LEVELS)


def choose_one_sample_per_day(sample_to_day):
    """One sample per unique day; prefer the shortest name (suspended over biofilm)."""
    by_day = defaultdict(list)
    for s, d in sample_to_day.items():
        by_day[d].append(s)
    chosen = {}
    for day, samples in by_day.items():
        pick = sorted(samples, key=lambda s: (len(s), s))[0]
        chosen[pick] = int(day)
    return dict(sorted(chosen.items(), key=lambda kv: kv[1]))


def main():
    ensure_dirs()

    # --- 1. metadata: sample -> Day / Description ---
    meta = pd.read_excel(META_XLSX, sheet_name=META_SHEET)
    meta["sample"] = meta["sample"].astype(str)
    sample_to_day = dict(zip(meta["sample"], meta["Day"].astype(int)))
    sample_to_desc = dict(zip(meta["sample"], meta["Description"]))
    print(f"metadata: {len(meta)} samples; days {sorted(set(sample_to_day.values()))}")

    sample_days = choose_one_sample_per_day(sample_to_day)
    dump(sample_days, open(MODEL_INPUTS / "sample_days.json", "w"), indent=2)
    print(f"canonical sample set ({len(sample_days)} of {len(sample_to_day)}): {sample_days}")

    # --- 2. wide relative abundance (percentages -> fractions) ---
    wide = pd.read_csv(WIDE_REL).set_index("seq")
    wide.columns = [str(c) for c in wide.columns]
    sample_cols = list(wide.columns)
    max_val = float(np.nanmax(wide.values))
    is_percentage = max_val > 1.5
    print(f"wide: {wide.shape[0]} seqs x {wide.shape[1]} samples; max={max_val:.3f} "
          f"({'PERCENT -> /100' if is_percentage else 'FRACTION'})")
    frac = wide / 100.0 if is_percentage else wide

    # --- 3. taxonomy (1 row / seq from the long table) ---
    long_df = pd.read_csv(LONG_TAX)
    tax_per_seq = long_df.drop_duplicates(subset=["seq"]).set_index("seq")[TAXO_LEVELS]
    missing = set(wide.index) - set(tax_per_seq.index)
    if missing:
        print(f"WARNING: {len(missing)} seqs lack taxonomy; padding with None")
        pad = pd.DataFrame({c: [None] * len(missing) for c in TAXO_LEVELS}, index=list(missing))
        tax_per_seq = pd.concat([tax_per_seq, pad])
    tax_per_seq = tax_per_seq.where(tax_per_seq.notna() & (tax_per_seq != "NA"), None)

    taxonomy_dict = {seq: {l: row[l] for l in TAXO_LEVELS} for seq, row in tax_per_seq.iterrows()}
    dump(taxonomy_dict, open(MODEL_INPUTS / "taxonomy.json", "w"))
    print(f"wrote taxonomy.json ({len(taxonomy_dict)} seqs)")

    # --- 4. iterativeIDs: lowest named taxon + unique numeric suffix ---
    counters = defaultdict(int)
    iterativeIDs, iterativeID_levels = {}, {}
    for seq, row in tax_per_seq.iterrows():
        chosen_name, chosen_level = None, "Unknown"
        for lvl in reversed(TAXO_LEVELS):
            val = row[lvl]
            if val is None or (isinstance(val, float) and np.isnan(val)) or val in ("NA", ""):
                continue
            chosen_name, chosen_level = str(val), lvl
            break
        if chosen_name is None:
            chosen_name, chosen_level = "Unknown", "Unknown"
        counters[chosen_name] += 1
        uid = f"{chosen_name}.{counters[chosen_name]}"
        iterativeIDs[seq] = uid
        iterativeID_levels[uid] = chosen_level
    dump(iterativeIDs, open(MODEL_INPUTS / "iterativeIDs.json", "w"))
    dump(iterativeID_levels, open(MODEL_INPUTS / "iterativeID_levels.json", "w"))
    print(f"wrote iterativeIDs.json + iterativeID_levels.json ({len(iterativeIDs)} entries)")

    # --- 5. abundances.json (all samples; fractions) ---
    abundances = {s: {seq: float(v) for seq, v in frac[s].items() if pd.notna(v)} for s in sample_cols}
    dump(abundances, open(MODEL_INPUTS / "abundances.json", "w"))
    print(f"wrote abundances.json ({len(abundances)} samples)")

    # --- 6. total.csv (long; rel_ab kept in PERCENT like the reference) ---
    rows = []
    for seq, tax_row in tax_per_seq.iterrows():
        for sample in sample_cols:
            rel_ab = float(frac.at[seq, sample]) if seq in frac.index else 0.0
            rows.append({
                "seq": seq, "sample": sample, "rel_ab": rel_ab * 100.0,
                **{l: tax_row[l] for l in TAXO_LEVELS},
                "date": "", "media": "",
                "timepoint": sample_to_day.get(sample, np.nan),
                "notes": sample_to_desc.get(sample, ""),
            })
    cols = ["seq", "sample", "rel_ab"] + TAXO_LEVELS + ["date", "media", "timepoint", "notes"]
    total_df = pd.DataFrame(rows)[cols]
    total_df.to_csv(MODEL_INPUTS / "total.csv", index=False)
    print(f"wrote total.csv ({len(total_df)} rows = {wide.shape[0]} seqs x {len(sample_cols)} samples)")


if __name__ == "__main__":
    main()

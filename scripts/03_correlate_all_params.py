"""Step 3 - Spearman-correlate every numeric operational parameter against every
ASV's relative abundance.  Adapted from
../codiffusion_bioreactor/_correlate_all_params_050526.py.

Here abundance samples and operational samples share the same canonical names, so
they are joined directly by sample (the reference had to join by day).

Reads:
    model_inputs/abundances.json
    model_inputs/iterativeIDs.json
    model_inputs/measurements/summary_samples.json

Writes (one per parameter; raw correlations, FDR applied later by step 7):
    modeling_files/correlations/ASV_correlations_all_relevant_samples_<slug>.json
        {iterativeID: {"correlation": rho, "p_value": p}}
"""
import re
import sys
import glob
import os
from json import dump
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (ensure_dirs, MODEL_INPUTS, MEASUREMENTS, CORR_DIR, load_json,
                     load_sample_days, CORR_MIN_DAY)

MIN_PAIRS = 5        # need >= 5 paired samples for a correlation
ASV_MIN_SAMPLES = 5  # ASV must be present (>0) in >= 5 samples


def slugify(s):
    s = re.sub(r"[\s/]+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9%_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "param"


def main():
    ensure_dirs()
    abundances = load_json(MODEL_INPUTS / "abundances.json")
    iterativeIDs = load_json(MODEL_INPUTS / "iterativeIDs.json")
    summary_samples = load_json(MEASUREMENTS / "summary_samples.json")

    # restrict to the post-startup window (day >= CORR_MIN_DAY)
    window = set(load_sample_days(CORR_MIN_DAY))
    summary_samples = {s: c for s, c in summary_samples.items() if s in window}
    op_samples = list(summary_samples.keys())
    print(f"operational samples (day >= {CORR_MIN_DAY}): {len(op_samples)} -> {op_samples}")

    # seq -> {sample: rel_ab} restricted to operational samples
    seq_to_abund = {}
    for sample in op_samples:
        for seq, val in abundances.get(sample, {}).items():
            seq_to_abund.setdefault(seq, {})[sample] = val
    seq_to_abund = {seq: d for seq, d in seq_to_abund.items()
                    if sum(1 for v in d.values() if v and v > 0) >= ASV_MIN_SAMPLES}
    print(f"ASVs present (>0) in >= {ASV_MIN_SAMPLES} samples: {len(seq_to_abund)}")

    # numeric parameters with enough values
    all_params = sorted({p for c in summary_samples.values() for p in c})
    numeric_params = []
    for p in all_params:
        finite = [c[p] for c in summary_samples.values()
                  if p in c and isinstance(c[p], (int, float)) and not (isinstance(c[p], float) and np.isnan(c[p]))]
        if len(finite) >= MIN_PAIRS:
            numeric_params.append(p)
    print(f"parameters with >= {MIN_PAIRS} values: {len(numeric_params)} / {len(all_params)}")

    # clear stale outputs
    for f in glob.glob(str(CORR_DIR / "ASV_correlations_all_relevant_samples_*.json")):
        os.remove(f)

    metrics_summary = []
    for param in numeric_params:
        metric_by_sample = {s: float(c[param]) for s, c in summary_samples.items()
                            if param in c and not (isinstance(c[param], float) and np.isnan(c[param]))}
        if len(metric_by_sample) < MIN_PAIRS:
            continue
        metric_series = pd.Series(metric_by_sample)

        correlations = {}
        const_skipped = 0
        for seq, sample_abund in seq_to_abund.items():
            abund_series = pd.Series(sample_abund)
            a, b = abund_series.align(metric_series, join="inner")
            if len(a) < MIN_PAIRS:
                continue
            if a.nunique() <= 1 or b.nunique() <= 1:
                const_skipped += 1
                continue
            rho, p = spearmanr(a, b)
            if np.isnan(rho):
                continue
            correlations[iterativeIDs.get(seq, seq)] = {"correlation": float(rho), "p_value": float(p)}

        if not correlations:
            continue
        correlations = dict(sorted(correlations.items(), key=lambda kv: -abs(kv[1]["correlation"])))
        dump(correlations, open(CORR_DIR / f"ASV_correlations_all_relevant_samples_{slugify(param)}.json", "w"), indent=2)
        metrics_summary.append((param, len(correlations), len(metric_by_sample), const_skipped))

    print(f"\nwrote {len(metrics_summary)} correlation files to {CORR_DIR}")
    print(f"{'parameter':55s} {'n_ASVs':>7s} {'n_samples':>9s} {'const_skip':>10s}")
    print("-" * 86)
    for param, n_corr, n_samp, n_const in sorted(metrics_summary, key=lambda x: -x[1]):
        print(f"{param[:55]:55s} {n_corr:7d} {n_samp:9d} {n_const:10d}")


if __name__ == "__main__":
    main()

# Pilots — H -> WW submission

`./run.sh <dataset_dir> <output_dir>` writes `<output_dir>/RESULT.json` with:

- `mu` — the signal strength relative to the Standard Model (stat/syst/total).
- `mH` — the Higgs mass in GeV (optional but reported).
- `significance_obs` — local significance of the observed excess in sigma.
- `discovery` — `true` when the local significance is at least 5 sigma.

## How it works

`fit.py` performs a binned Poisson likelihood fit with three parameters — mass
`m`, signal strength `mu`, background normalisation `b` — using only numpy +
scipy on templates on the **(MiniHEP-head classifier score, m_T, m_ll)** plane:

1. **Analysis head** (`minihep.py` + `model.json`, `"head_feat"`): an MLP
   (10 → 32 → 16 → 1, ReLU) on the 10 engineered features (lepton pT, MET,
   m_ll, m_T, pT_ll, Δφ, Δη, jets, flavour), trained offline on the pool
   (weights as text; no torch at run time; validated AUC ≈ 0.92). The
   transformer-embedding head (MiniHEP backbone + MLP) is shipped in the same
   file for the slower alternative modes.
2. **Exact per-mass templates**: because the mass-morphing rule scales lepton
   pT and MET by `m / 125`, every head feature scales *exactly linearly* with
   `m` (m_T, m_ll, pT_ll, lepton/MET pT scale by `m/125`; angles, jets and
   flavour are invariant). The signal template at every mass of the grid
   (2.5 GeV steps on an extended [95, 155] GeV range) is therefore an exact
   rescoring of the full pool — full statistics, ~6 s per dataset. Background
   at its nominal normalisation.
3. **Fit**: profile scan over the mass grid + Nelder-Mead refinement of
   `(m, mu, b)`; uncertainties from the numerical Hessian (μ); the mass
   uncertainty from the profile likelihood (where ΔNLL = 1/2, 0.5 GeV steps);
   significance from the likelihood ratio against `mu = 0`. The quoted stat
   errors carry a small coverage calibration (μ ×0.85, mH ×1.5) measured on
   validation pseudo-datasets, so the pull widths are ≈ 1 (honest error bars).
4. **Fallback**: `CLF_MODE=ref` (or a missing fast head) runs the same fit on
   the plain `(m_T, m_ll)` plane.

On the 21-seed local suite this scores: μ slope 1.00 / pull width 0.93,
mH slope 0.92 / width 0.86, no false discoveries, discovery power 19/19,
level 2 on both observables (vs the plain fit: slope 1.05 / width 1.07,
power 17/19).

## Classifier choice: tested by the measurement, not by AUC (10/16)

The head was re-trained in several event-weighted variants
(`minihep_student/train_fast_head_weighted.py`, torch-only) and each was scored
on the same 21-seed suite (13 seeds with expected significance ≥ 5) through the
full fit — the training loss of the shipped head *does not* use the MC event
weights, and the background weights vary by ~4 orders of magnitude
(0.0003 → 5.2), so this had to be checked:

| head | training | μ slope | μ pull width | μ typ. miss | σ_μ | mH slope | mH width | σ_mH |
|---|---|---|---|---|---|---|---|---|
| **v0 (shipped)** | 50/50 balanced, unweighted | **1.01** | 0.92 | 0.095 | 0.099 | **0.94** | **1.02** | **2.76** |
| v2 | weighted mixture sample | 1.01 | 0.89 | 0.097 | 0.107 | 0.85 | 0.87 | 3.73 |
| v3 | full, weighted BCE | 1.05 | 0.76 | **0.084** | 0.105 | 0.80 | 1.34 | 3.38 |
| v3r10 | v3 + adaptive score bins  | 1.03 | 0.86 | 0.084 | 0.094 | 0.90 | 1.00 | 4.72 |
| v4 | mass-parametrised head (features + m/125 input), signal morphed over [100,150] | 0.95 | 2.04 | 0.204 | 0.100 | 0.03 | 6.06 | 19.9 |

Findings:

* **Event weighting improves the μ direction** (mean pull −0.46 → −0.16, typical
  miss 0.095 → 0.084) but the *mass* gets worse (σ_mH 2.8 → 3.4–4.7, slope
  0.94 → 0.80–0.90): the weighted head's score is a proper posterior of the
  true mixture — compressed at low score, and the fixed score binning tuned for
  the old head wastes it (adaptive signal-quantile bins recover μ, not the mass).
* **The mass-parametrised head does not work** as shipped: with k = m/125 as an
  extra input it is degenerate in the fit (mH slope 0.03; the energy is in a
  "ruler" the fit cannot use without a per-mass-hypothesis re-binning of the
  data, which was also tried — still degenerate). This is the GUIDE's
  "research topic" — abandoned.
* **Decision: keep the shipped head (v0).** It remains the best for the
  combined μ + m_H measurement, and the weighted variants do not beat it once
  the mass is included; the run-time pipeline is unchanged.
* `fit.py` now reads an optional `submission/fit.cfg` (JSON: `"score_edges"` or
  `"score_nbins"` for adaptive signal-quantile score bins, and
  `"score_data_per_m"` for the per-hypothesis data binning) — absent by
  default, the behaviour is exactly as before (checked with `sbh check`).

The pipeline is deterministic (no randomness), reads only `<dataset_dir>` and
its own directory, and needs nothing beyond Python 3 with numpy and scipy.
The weights were trained offline (`minihep_student/train_fast_head.py` for the
fast head, `minihep_student/train_head.py` for the transformer head); the
sealed evaluation receives the same pool, so no training happens at run time.

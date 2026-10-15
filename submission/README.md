# Pilots — H -> WW submission

`./run.sh <dataset_dir> <output_dir>` writes `<output_dir>/RESULT.json` with:

- `mu` — the signal strength relative to the Standard Model (stat/syst/total).
- `mH` — the Higgs mass in GeV (optional but reported).
- `significance_obs` — local significance of the observed excess in sigma.
- `discovery` — `true` when the local significance is at least 5 sigma.

## How it works

`fit.py` performs a binned Poisson likelihood fit with three parameters — mass
`m`, signal strength `mu`, background normalisation `b` — using only numpy +
scipy:

1. **Templates** on the `(m_T, m_ll)` plane come from the pool: background at
   its nominal normalisation; signal at `mu = 1` for every mass of the grid
   `[100, 150] GeV`, built by morphing the 125 GeV sample with the dataset's
   own rule (lepton pT and missing ET scaled by `m / 125`). Templates at masses
   between grid points are linear interpolations.
2. **Fit**: profile scan over the mass grid + Nelder-Mead refinement of
   `(m, mu, b)`; uncertainties from the numerical Hessian; the mass uncertainty
   from the profile likelihood (where ΔNLL = 1/2); significance from the
   likelihood ratio against `mu = 0`.

The pipeline is deterministic (no randomness), reads only `<dataset_dir>` and
its own directory, and needs nothing beyond Python 3 with numpy and scipy.

## MiniHEP analysis head (optional, experimental)

`minihep.py` + `model.json` ship a signal/background classifier trained on the
MiniHEP foundation model (masked four-vector transformer, d_model 32 — forward
pass reimplemented in numpy, weights shipped as text; trained offline by
`minihep_student/train_head.py`). Each event is tokenised into a fixed 5-slot
sequence (lepton 0, lepton 1, jet 0, jet 1, MET proxy), embedded by the
transformer, and an MLP head maps the embedding + engineered features
(dilepton mass/transverse mass/pT, Δφ, Δη, jets, flavour) to a
signal-vs-background probability.

Two variants are switchable at run time (env vars, both deterministic):

- `CLF_MODE=clf` — template fit on the `(classifier score, m_T)` plane,
  `CLF_MASS_SCORE=rescore` re-scores the signal per mass (default: score once
  at the nominal mass).
- `CLF_MODE=cut` — `(m_T, m_ll)` fit on events selected by score, threshold
  `CLF_CUT` (default 0.6).

The classifier head separates signal from background well (test AUC ≈ 0.9),
but on this toy the classifier variants were measured to score *worse* than
the plain `(m_T, m_ll)` fit under the leaderboard metrics (worse mass
accuracy, overconfident errors), so the plain fit is the default. The head is
kept in the submission so it can be re-enabled and tuned further.

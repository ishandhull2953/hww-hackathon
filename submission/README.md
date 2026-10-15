# Pilots — H -> WW submission

`./run.sh <dataset_dir> <output_dir>` writes `<output_dir>/RESULT.json` with:

- `mu` — the signal strength relative to the Standard Model (stat/syst/total).
- `mH` — the Higgs mass in GeV (optional but reported).
- `significance_obs` — local significance of the observed excess in sigma.
- `discovery` — `true` when the local significance is at least 5 sigma.

## How it works

`fit.py` builds two-dimensional templates on the (transverse mass, dilepton mass)
plane, exclusively with numpy + scipy:

1. The background template comes straight from `mc/pool.npz` at its nominal
   normalisation (weights sum to the expected events).
2. The signal template comes from the pool's signal events, morphed to each
   mass of the grid `[100, 150] GeV` via the dataset's own rule (lepton pT and
   missing ET scaled by `m / 125`).
3. A binned Poisson likelihood with three parameters — mass `m`, signal
   strength `mu`, background normalisation `b` — is maximised with a profile
   scan over the mass grid and a local refinement (Nelder-Mead).
4. Uncertainties come from the numerical Hessian; the mass uncertainty is
   taken from the profile likelihood (where ΔNLL = 1/2).
5. The significance is the likelihood-ratio test against `mu = 0`.

The pipeline is deterministic (no randomness), reads only `<dataset_dir>` and
its own directory, and needs nothing beyond Python 3 with numpy and scipy.

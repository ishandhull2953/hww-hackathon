#!/usr/bin/env python3
"""The Pilots H -> WW pipeline: a three-dimensional template fit for the Higgs
mass and the signal strength.

    python3 fit.py <dataset_dir> <output_dir>      (run.sh calls this)

The default analysis (mode "score3") uses the MiniHEP analysis head to build
templates on the (classifier score, m_T, m_ll) plane:

* every event is scored once by the fast head on the 10 engineered features
  (minihep.py / model.json "head_feat": an MLP, weights as text);
* because the mass-morphing rule scales lepton pT and MET by m / 125, every
  head feature scales exactly linearly with m (m_T, m_ll, pT_ll and the lepton
  and MET pT scale by k; the angles, jet and flavour features are invariant),
  so the signal template at *every* mass is an exact rescoring of the pool
  events -- full statistics, no approximation, ~10 s per dataset;
* fit (m_H, mu, b) with a Poisson likelihood, signal templates linearly
  interpolated in mass; errors from the numerical Hessian (mu) and the
  profile likelihood (m_H); significance from the likelihood ratio vs mu = 0.

The quoted stat errors carry a coverage calibration (mu_err_scale / mh_err_scale)
measured on validation pseudo-datasets, so the pull widths are ~1 (honest).

Fallbacks: CLF_MODE=ref uses the plain (m_T, m_ll) templates; CLF_MODE=clf|cuts
are the earlier transformer-head experiments (now switchable only if the
transformer head is present).  If model.json has no fast head, the fit falls
back to the plain (m_T, m_ll) templates.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hww as H             # noqa: E402
import minihep as M         # noqa: E402

MASS_GRID = np.arange(95.0, 155.1, 2.5)     # 2.5 GeV steps on an extended grid
M_LO, M_HI = float(MASS_GRID[0]), float(MASS_GRID[-1])
MT_EDGES = np.linspace(0.0, 250.0, 36)      # 35 bins (a finer grid sharpens m_T)
MLL_EDGES = np.linspace(0.0, 150.0, 15)     # 14 bins (a finer grid sharpens m_ll)
SCORE_EDGES = np.array([0.0, 0.35, 0.55, 0.7, 0.8, 0.86, 0.92, 1.0])
MU_ESCALE = float(os.environ.get("MU_ESCALE", "0.85"))    # coverage calibration (pseudo-data validation)
MH_ESCALE = float(os.environ.get("MH_ESCALE", "1.5"))
# head feature scaling with the mass-morphing factor k = m / 125:
#   [lep_pt_0, lep_pt_1, met_et, dphi, mll, ptll, mt, jet_n, deta, same_flavour]
FEAT_MASS_SCALE = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                           dtype=np.float32)


def score3_templates(model, pool, data):
    """(score, m_T, m_ll) data histogram and signal templates per mass (exact
    linear rescoring) plus the background template."""
    import hww as H2
    sig = H2.select(pool, pool["label"] == 1)
    bkg = H2.select(pool, pool["label"] == 0)
    kin_s = H2.kinematics(sig)
    feats0 = M.features(sig)                       # at the nominal event
    mt0, mll0 = kin_s["mt"], kin_s["mll"]

    feats_d = M.features(data); kin_d = H2.kinematics(data)
    n = np.histogramdd([model.head_feat.score(feats_d), kin_d["mt"], kin_d["mll"]],
                       [SCORE_EDGES, MT_EDGES, MLL_EDGES])[0]
    feats_b = M.features(bkg); kin_b = H2.kinematics(bkg)
    B = np.histogramdd([model.head_feat.score(feats_b), kin_b["mt"], kin_b["mll"]],
                       [SCORE_EDGES, MT_EDGES, MLL_EDGES], weights=bkg["weight"])[0]

    S = {}
    for m in MASS_GRID:
        k = float(m) / H.M_REF
        f = feats0 * (1.0 + (k - 1.0) * FEAT_MASS_SCALE)
        S[float(m)] = np.histogramdd([model.head_feat.score(f), mt0 * k, mll0 * k],
                                     [SCORE_EDGES, MT_EDGES, MLL_EDGES],
                                     weights=sig["weight"])[0]
    return S, B, n


def template2d(ev, weights=None):
    k = H.kinematics(ev)
    return np.histogram2d(k["mt"], k["mll"], [MT_EDGES, MLL_EDGES], weights=weights)[0]


def fit_scan(nll, grid):
    def profile(m):
        r = minimize(lambda p: nll(m, p[0], p[1]), [1.0, 1.0], method="L-BFGS-B",
                     bounds=[(-2.0, 20.0), (0.2, 5.0)])
        return r.fun, r.x

    scan = [(profile(m), m) for m in grid]
    (f0, (mu0, b0)), m0 = min(scan, key=lambda t: t[0][0])
    r = minimize(lambda p: nll(np.clip(p[0], M_LO, M_HI), p[1], p[2]), [m0, mu0, b0],
                 method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 4000})
    m_hat = float(np.clip(r.x[0], M_LO, M_HI)); mu_hat = float(r.x[1]); b_hat = float(r.x[2])
    return m_hat, mu_hat, b_hat, nll(m_hat, mu_hat, b_hat)


def hessian_errors(nll, x, h):
    hess = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ei, ej = np.eye(3)[i] * h[i], np.eye(3)[j] * h[j]
            hess[i, j] = (nll(*(x + ei + ej)) - nll(*(x + ei - ej))
                          - nll(*(x - ei + ej)) + nll(*(x - ei - ej))) / (4 * h[i] * h[j])
    try:
        cov = np.linalg.inv(hess)
        err = np.sqrt(np.diag(cov))
        if not np.all(np.isfinite(err)) or np.any(np.diag(cov) <= 0):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        err = np.array([0.5 * (M_HI - M_LO), 1.0, 0.1])
        return err, "Hessian not positive definite; errors set to prior widths"
    return err, ""


def main(ds: Path, out: Path):
    data = H.load(ds / "data" / "events.npz")
    pool = H.load(ds / "mc" / "pool.npz")
    sig = H.select(pool, pool["label"] == 1)
    bkg = H.select(pool, pool["label"] == 0)

    mode = os.environ.get("CLF_MODE", "score3")
    model_path = HERE / "model.json"
    model = M.load_model(model_path) if model_path.is_file() else None

    if mode == "score3" and model is not None and model.head_feat is not None:
        S, B, n = score3_templates(model, pool, data)
        note = ("three-dimensional (MiniHEP-head score, m_T, m_ll) template fit; "
                "signal score re-derived exactly at every template mass "
                f"(score bins {SCORE_EDGES.tolist()})")
    else:
        n = template2d(data)
        S = {float(m): template2d(H.morph(sig, m), sig["weight"]) for m in MASS_GRID}
        B = template2d(bkg, bkg["weight"])
        mode = "ref"
        note = "two-dimensional (m_T, m_ll) template fit"

    grid = sorted(S)

    def nll(m, mu, b):
        lam = np.clip(mu * H.interpolate(S, m) + b * B, 1e-9, None)
        return float(np.sum(lam - n * np.log(lam)))

    m_hat, mu_hat, b_hat, f_min = fit_scan(nll, grid)
    err, ehint = hessian_errors(nll, np.array([m_hat, mu_hat, b_hat]),
                                np.array([0.5, 0.02, 0.01]))
    mu_err = float(max(err[1], 1e-3)) * MU_ESCALE
    if ehint:
        note += "; " + ehint

    def profile0(m):
        r = minimize(lambda p: nll(m, p[0], p[1]), [1.0, 1.0], method="L-BFGS-B",
                     bounds=[(-2.0, 20.0), (0.2, 5.0)])
        return r.fun, r.x

    ms = np.arange(M_LO, M_HI + 0.01, 0.5)
    prof = np.array([profile0(m)[0] for m in ms])
    prof = prof - min(prof.min(), f_min)
    center = ms[int(np.argmin(np.abs(ms - m_hat)))]
    lo, hi = M_LO, M_HI
    left = [(m, v) for m, v in zip(ms, prof) if m < center]
    for (m1, v1), (m2, v2) in zip(left[:-1], left[1:]):
        if v1 >= 0.5 > v2:
            lo = m1 + (v1 - 0.5) / (v1 - v2) * (m2 - m1)
    right = [(m, v) for m, v in zip(ms, prof) if m > center]
    for (m1, v1), (m2, v2) in zip(right[:-1], right[1:]):
        if v1 < 0.5 <= v2:
            hi = m1 + (0.5 - v1) / (v2 - v1) * (m2 - m1)
            break
    m_err = float(np.clip(0.5 * (hi - lo), 0.5, 0.5 * (M_HI - M_LO))) * MH_ESCALE
    note += f"; mass interval [{lo:.1f}, {hi:.1f}] from the profile likelihood; " \
            f"errors coverage-scaled ({MU_ESCALE:.2f}, {MH_ESCALE:.2f})"

    r0 = minimize(lambda p: nll(m_hat, 0.0, p[0]), [b_hat], method="L-BFGS-B", bounds=[(0.2, 5.0)])
    q0 = max(0.0, 2.0 * (r0.fun - f_min)) if mu_hat > 0 else 0.0
    z = float(np.sqrt(q0))

    result = {
        "pois": [{"name": "mH", "value": m_hat, "stat": m_err, "syst": 0.0, "total": m_err},
                 {"name": "mu", "value": mu_hat, "stat": mu_err, "syst": 0.0, "total": mu_err}],
        "significance_obs": z,
        "discovery": bool(z >= 5.0),
        "note": note + f"; background normalisation {b_hat:.3f} ± {float(err[2]):.3f}",
        "model": mode,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1))
    print(f"mH = {m_hat:.2f} ± {m_err:.2f}  mu = {mu_hat:.3f} ± {mu_err:.3f}  b = {b_hat:.3f}  Z = {z:.2f}  [{mode}]")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

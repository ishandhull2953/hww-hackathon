#!/usr/bin/env python3
"""The reference pipeline: a two-dimensional template fit for the Higgs mass and the signal strength.

    python3 fit.py <dataset_dir> <output_dir>      (run.sh calls this)

Templates on the (m_T, m_ll) plane come from the pool: the background at its nominal normalisation and the signal
at mu = 1 for every mass of the grid, built by morphing the 125 GeV sample (hww.py, a copy of the kit's). The fit
has three parameters, m_H, mu and the background normalisation b, and a Poisson likelihood bin by bin; the signal
template at a mass between grid points is the linear interpolation of its neighbours. Errors come from the
Hessian at the minimum, the significance from the likelihood ratio against mu = 0.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hww as H  # noqa: E402

M_LO, M_HI = float(H.MASS_GRID[0]), float(H.MASS_GRID[-1])


def main(ds: Path, out: Path):
    data = H.load(ds / "data" / "events.npz")
    pool = H.load(ds / "mc" / "pool.npz")
    n = H.template2d(data)
    S, B = H.templates(pool)
    grid = sorted(S)

    def nll(m, mu, b):
        lam = np.clip(mu * H.interpolate(S, m) + b * B, 1e-9, None)
        return float(np.sum(lam - n * np.log(lam)))

    def profile(m):
        """mu and b at fixed m."""
        r = minimize(lambda p: nll(m, p[0], p[1]), [1.0, 1.0], method="L-BFGS-B", bounds=[(-2.0, 20.0), (0.2, 5.0)])
        return r.fun, r.x

    scan = [(profile(m), m) for m in grid]
    (f0, (mu0, b0)), m0 = min(scan, key=lambda t: t[0][0])
    r = minimize(lambda p: nll(np.clip(p[0], M_LO, M_HI), p[1], p[2]), [m0, mu0, b0], method="Nelder-Mead",
                 options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 4000})
    m_hat, mu_hat, b_hat = float(np.clip(r.x[0], M_LO, M_HI)), float(r.x[1]), float(r.x[2])
    f_min = nll(m_hat, mu_hat, b_hat)

    # errors: numerical Hessian of the NLL (half a unit of NLL = one sigma)
    x = np.array([m_hat, mu_hat, b_hat]); h = np.array([0.5, 0.02, 0.01])
    hess = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ei, ej = np.eye(3)[i] * h[i], np.eye(3)[j] * h[j]
            hess[i, j] = (nll(*(x + ei + ej)) - nll(*(x + ei - ej)) - nll(*(x - ei + ej)) + nll(*(x - ei - ej))) / (4 * h[i] * h[j])
    note = "two-dimensional (m_T, m_ll) template fit, mass morphed from the 125 GeV sample"
    try:
        cov = np.linalg.inv(hess)
        err = np.sqrt(np.diag(cov))
        if not np.all(np.isfinite(err)) or np.any(np.diag(cov) <= 0):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        err = np.array([0.5 * (M_HI - M_LO), 1.0, 0.1])
        note += "; the Hessian was not positive definite, errors set to the prior width"
    mu_err = float(max(err[1], 1e-3))

    # the mass error from the profile likelihood: where NLL - NLL_min crosses 1/2 on each side of the minimum.
    # The Hessian misjudges it when the likelihood is flat or lopsided, which it is for a weak signal.
    ms = np.arange(M_LO, M_HI + 0.01, 1.0)
    prof = np.array([profile(m)[0] for m in ms])
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
    m_err = float(np.clip(0.5 * (hi - lo), 0.5, 0.5 * (M_HI - M_LO)))

    note += f"; mass interval [{lo:.1f}, {hi:.1f}] from the profile likelihood"

    # significance: the likelihood ratio against no signal (the mass is undefined there; b refits)
    r0 = minimize(lambda p: nll(m_hat, 0.0, p[0]), [b_hat], method="L-BFGS-B", bounds=[(0.2, 5.0)])
    q0 = max(0.0, 2.0 * (r0.fun - f_min)) if mu_hat > 0 else 0.0
    z = float(np.sqrt(q0))

    result = {
        "pois": [{"name": "mH", "value": m_hat, "stat": m_err, "syst": 0.0, "total": m_err},
                 {"name": "mu", "value": mu_hat, "stat": mu_err, "syst": 0.0, "total": mu_err}],
        "significance_obs": z,
        "discovery": bool(z >= 5.0),
        "note": note + f"; background normalisation {b_hat:.3f} ± {float(err[2]):.3f}",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1))
    print(f"mH = {m_hat:.2f} ± {m_err:.2f}  mu = {mu_hat:.3f} ± {mu_err:.3f}  b = {b_hat:.3f}  Z = {z:.2f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

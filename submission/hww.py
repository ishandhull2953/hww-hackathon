"""The H -> WW toy, shared by the generator, the examples and the plots: the pool, the kinematics, the mass
morphing and the templates.

Events are dicts of equal-length numpy arrays (the columns of DATASET.md). `kinematics` adds the dilepton
mass, the transverse mass, the dilepton pT and the opening angle. A signal of mass m is the 125 GeV sample with
every lepton pT and the missing ET scaled by m / 125 (`morph`): that is the rule the planted datasets were made
with, so templates built the same way are exact in this toy.
"""
import json
from pathlib import Path

import numpy as np

M_REF = 125.0                       # the simulated Higgs mass
LEPTON_COLS = ["lep_pt_0", "lep_eta_0", "lep_phi_0", "lep_pt_1", "lep_eta_1", "lep_phi_1",
               "lep_charge_0", "lep_charge_1", "lep_type_0", "lep_type_1"]
MET_COLS = ["met_et", "met_phi"]
JET_COLS = ["jet_n", "jet_pt_0", "jet_eta_0", "jet_phi_0", "jet_pt_1", "jet_eta_1", "jet_phi_1"]
EVENT_COLS = LEPTON_COLS + MET_COLS + JET_COLS
SCALED = ["lep_pt_0", "lep_pt_1", "met_et"]   # what the morphing scales

# the reference binning of the two-dimensional templates (m_T x m_ll); overflow is dropped
MT_EDGES = np.linspace(0.0, 250.0, 26)
MLL_EDGES = np.linspace(0.0, 150.0, 11)
MASS_GRID = np.arange(100.0, 150.1, 2.5)   # 2.5 GeV steps: with 5 the interpolated templates are visibly broader than the truth


def load(path):
    """An .npz of named columns as a dict of arrays."""
    with np.load(Path(path)) as f:
        return {k: f[k] for k in f.files}


def select(ev, mask):
    return {k: v[mask] for k, v in ev.items()}


def kinematics(ev):
    """m_ll, m_T, pT_ll and |delta phi_ll| from the two leptons (massless) and the missing ET."""
    pt0, eta0, phi0 = ev["lep_pt_0"].astype(float), ev["lep_eta_0"].astype(float), ev["lep_phi_0"].astype(float)
    pt1, eta1, phi1 = ev["lep_pt_1"].astype(float), ev["lep_eta_1"].astype(float), ev["lep_phi_1"].astype(float)
    met, mphi = ev["met_et"].astype(float), ev["met_phi"].astype(float)
    px = pt0 * np.cos(phi0) + pt1 * np.cos(phi1)
    py = pt0 * np.sin(phi0) + pt1 * np.sin(phi1)
    pz = pt0 * np.sinh(eta0) + pt1 * np.sinh(eta1)
    e = pt0 * np.cosh(eta0) + pt1 * np.cosh(eta1)
    mll = np.sqrt(np.clip(e * e - px * px - py * py - pz * pz, 0.0, None))
    ptll = np.hypot(px, py)
    et_ll = np.hypot(ptll, mll)
    mt = np.sqrt(np.clip((et_ll + met) ** 2 - (px + met * np.cos(mphi)) ** 2 - (py + met * np.sin(mphi)) ** 2, 0.0, None))
    dphi = np.abs(np.mod(phi1 - phi0 + 3 * np.pi, 2 * np.pi) - np.pi)
    return {"mll": mll, "mt": mt, "ptll": ptll, "dphi": dphi}


def morph(ev, m):
    """The events of a Higgs of mass m, from the simulated ones: lepton pT and missing ET scaled by m / 125."""
    k = float(m) / M_REF
    out = dict(ev)
    for c in SCALED:
        out[c] = (ev[c].astype(float) * k).astype(ev[c].dtype)
    return out


def template2d(ev, weights=None):
    """The (m_T, m_ll) histogram of the events on the reference binning."""
    k = kinematics(ev)
    h, _, _ = np.histogram2d(k["mt"], k["mll"], [MT_EDGES, MLL_EDGES], weights=weights)
    return h


def templates(pool, masses=MASS_GRID):
    """Signal templates at mu = 1 for every mass of the grid, and the nominal background template, from the
    pool (weights sum to the expected events at the dataset's luminosity)."""
    sig = select(pool, pool["label"] == 1)
    bkg = select(pool, pool["label"] == 0)
    S = {float(m): template2d(morph(sig, m), sig["weight"]) for m in masses}
    B = template2d(bkg, bkg["weight"])
    return S, B


def interpolate(S, m):
    """The signal template at any mass inside the grid, linearly between its two neighbours."""
    grid = np.array(sorted(S))
    if len(grid) == 1:
        return S[grid[0]]
    m = float(np.clip(m, grid[0], grid[-1]))
    j = int(np.clip(np.searchsorted(grid, m, side="right") - 1, 0, len(grid) - 2))
    lo, hi = grid[j], grid[j + 1]
    f = (m - lo) / (hi - lo)
    return (1 - f) * S[lo] + f * S[hi]


def asimov_significance(s, b):
    """The expected discovery significance of signal s on background b, bin by bin."""
    s, b = np.asarray(s, float), np.asarray(b, float)
    ok = (b > 0) & (s > 0)
    return float(np.sqrt(2.0 * np.sum((s[ok] + b[ok]) * np.log1p(s[ok] / b[ok]) - s[ok])))


def read_meta(dataset_dir):
    return json.loads((Path(dataset_dir) / "mc" / "pool_meta.json").read_text())

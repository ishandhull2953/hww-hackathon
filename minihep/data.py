"""NumPy loader for the ATLAS Open Data WW events (dataWW_d1.root).

Tokenizes each event into a fixed 5-slot sequence:

    slot 0: lepton 0
    slot 1: lepton 1
    slot 2: jet 0 (or padding)
    slot 3: jet 1 (or padding)
    slot 4: MET proxy

Each token's features are a 4-vector (px, py, pz, E) in GeV.  Leptons and
jets are built from (pt, eta, phi) with E in MeV from the file (divided by
1000); missing jets are zero vectors (the file uses -7.0 fillers, which are
never used).  The MET token is the transverse proxy
(MET*cos(phi_met), MET*sin(phi_met), 0, MET), NOT a neutrino-pair four-vector.

Token types (see TOKEN_NAMES): e-, e+, mu-, mu+, jet, met.  This dataset has
no jet flavour/type branch, so every jet gets the same TOK_JET code.  The
file stores at most two leptons and two jets; jets beyond slot 1 are not
available even though jet_n can be larger.

Labels and weights are returned separately from the model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import uproot

# ----------------------------------------------------------------------------
# Token scheme
# ----------------------------------------------------------------------------
TOK_EM = 0  # e-
TOK_EP = 1  # e+
TOK_MUM = 2  # mu-
TOK_MUP = 3  # mu+
TOK_JET = 4  # any jet (no flavour/type branch in this file)
TOK_MET = 5  # missing transverse momentum

TOKEN_NAMES = {
    TOK_EM: "e-",
    TOK_EP: "e+",
    TOK_MUM: "mu-",
    TOK_MUP: "mu+",
    TOK_JET: "jet",
    TOK_MET: "met",
}

N_SLOTS = 5
SLOT_LEP0, SLOT_LEP1, SLOT_JET0, SLOT_JET1, SLOT_MET = range(N_SLOTS)
N_FEATURES = 4  # (px, py, pz, E) in GeV

_MEV_TO_GEV = 1.0 / 1000.0  # file stores leptons/jets E in MeV
_PDG_ELECTRON = 11
_PDG_MUON = 13

_BRANCHES = (
    "index",
    "eventNumber",
    "runNumber",
    "channelNumber",
    "label",
    "mcWeight",
    "met_et",
    "met_phi",
    "lep_n",
    "lep_pt_0",
    "lep_pt_1",
    "lep_eta_0",
    "lep_eta_1",
    "lep_phi_0",
    "lep_phi_1",
    "lep_E_0",
    "lep_E_1",
    "lep_charge_0",
    "lep_charge_1",
    "lep_type_0",
    "lep_type_1",
    "jet_n",
    "jet_pt_0",
    "jet_pt_1",
    "jet_eta_0",
    "jet_eta_1",
    "jet_phi_0",
    "jet_phi_1",
    "jet_E_0",
    "jet_E_1",
)


@dataclass
class EventBatch:
    """One tokenized, sampled batch of events.

    Attributes
    ----------
    fourvec : (N, 5, 4) float32  token (px, py, pz, E) in GeV; zero-padded
    token_types : (N, 5) int64   codes from TOKEN_NAMES
    valid : (N, 5) bool          False for padded (absent) tokens
    event_index / event_number / run_number / channel_number : (N,) int64
        event identity; event_index is the row in the original file
    label : (N,) int64           ground truth (1 = signal), NOT a model input
    mc_weight : (N,) float64     MC event weight, NOT a model input
    """

    fourvec: np.ndarray
    token_types: np.ndarray
    valid: np.ndarray
    event_index: np.ndarray
    event_number: np.ndarray
    run_number: np.ndarray
    channel_number: np.ndarray
    label: np.ndarray
    mc_weight: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.label)
        expected = (n, N_SLOTS, N_FEATURES)
        if self.fourvec.shape != expected:
            raise ValueError(f"fourvec must be {expected}, got {self.fourvec.shape}")
        for name in ("token_types", "valid"):
            if getattr(self, name).shape != (n, N_SLOTS):
                raise ValueError(f"{name} must be {(n, N_SLOTS)}")
        if self.valid.dtype != np.bool_:
            raise ValueError("valid must be bool")

    @property
    def n_events(self) -> int:
        return len(self.label)


def repo_root() -> Path:
    """Directory that contains dataWW_d1.root (the repo root)."""
    return Path(__file__).resolve().parents[1]


def lepton_token_type(pdg_flavor: np.ndarray, charge: np.ndarray) -> np.ndarray:
    """Map (PDG flavour, charge) to a token code: e-0, e+1, mu-2, mu+3."""
    flavor = np.asarray(pdg_flavor)
    chg = np.asarray(charge)
    if not np.all(np.isin(flavor, (_PDG_ELECTRON, _PDG_MUON))):
        raise ValueError("unexpected lepton PDG flavour in file")
    if not np.all(np.abs(chg) == 1):
        raise ValueError("unexpected lepton charge in file")
    return np.where(flavor == _PDG_MUON, TOK_MUM, TOK_EM) + (chg > 0).astype(np.int64)


def _vec4_pt_eta_phi_E(
    pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, e_gev: np.ndarray
) -> np.ndarray:
    """(px, py, pz, E) components given (pt, eta, phi) and E in GeV."""
    pt = np.asarray(pt, dtype=np.float64)
    return np.stack(
        (
            pt * np.cos(phi),
            pt * np.sin(phi),
            pt * np.sinh(eta),
            np.asarray(e_gev, dtype=np.float64),
        ),
        axis=-1,
    )


def load_events(
    path: Path | None = None,
    max_events: int = 6000,
    seed: int = 42,
) -> EventBatch:
    """Read dataWW_d1.root, keep events with lep_n == 2, sample at most
    max_events of them with a fixed RNG seed, and tokenize.

    The subset is deterministic for a given (file, max_events, seed).  No
    mcWeight filtering is applied here; weights are returned as-is.
    """
    if path is None:
        path = repo_root() / "dataWW_d1.root"
    tree = uproot.open(path)["tree_event"]
    arr = tree.arrays(_BRANCHES, library="np")

    keep = np.asarray(arr["lep_n"]) == 2
    selected = np.flatnonzero(keep)
    rng = np.random.default_rng(seed)
    take = rng.permutation(selected.size)[: min(max_events, selected.size)]
    idx = selected[take]

    n = idx.size
    fourvec = np.zeros((n, N_SLOTS, N_FEATURES), dtype=np.float32)
    token_types = np.full((n, N_SLOTS), TOK_MET, dtype=np.int64)
    valid = np.zeros((n, N_SLOTS), dtype=np.bool_)

    # --- leptons (always present: we filtered lep_n == 2) --------------------
    for slot, k in ((SLOT_LEP0, 0), (SLOT_LEP1, 1)):
        e_gev = arr[f"lep_E_{k}"][idx] * _MEV_TO_GEV
        fourvec[:, slot] = _vec4_pt_eta_phi_E(
            arr[f"lep_pt_{k}"][idx],
            arr[f"lep_eta_{k}"][idx],
            arr[f"lep_phi_{k}"][idx],
            e_gev,
        )
        token_types[:, slot] = lepton_token_type(
            arr[f"lep_type_{k}"][idx], arr[f"lep_charge_{k}"][idx]
        )
    valid[:, SLOT_LEP0] = True
    valid[:, SLOT_LEP1] = True

    # --- jets: absent slots are zero vectors; ignore -7.0 fillers ------------
    for slot, k in ((SLOT_JET0, 0), (SLOT_JET1, 1)):
        present = np.asarray(arr[f"jet_pt_{k}"][idx]) != -7.0
        e_gev = arr[f"jet_E_{k}"][idx] * _MEV_TO_GEV
        vec = _vec4_pt_eta_phi_E(
            arr[f"jet_pt_{k}"][idx],
            arr[f"jet_eta_{k}"][idx],
            arr[f"jet_phi_{k}"][idx],
            e_gev,
        )
        fourvec[:, slot] = np.where(present[:, None], vec, 0.0)
        token_types[:, slot] = TOK_JET
        valid[:, slot] = present

    # --- MET transverse proxy ------------------------------------------------
    met_et = np.asarray(arr["met_et"][idx], dtype=np.float64)
    met_phi = np.asarray(arr["met_phi"][idx], dtype=np.float64)
    fourvec[:, SLOT_MET] = np.stack(
        (met_et * np.cos(met_phi), met_et * np.sin(met_phi), np.zeros(n), met_et),
        axis=-1,
    )
    valid[:, SLOT_MET] = True

    return EventBatch(
        fourvec=fourvec,
        token_types=token_types,
        valid=valid,
        event_index=np.asarray(arr["index"][idx]),
        event_number=np.asarray(arr["eventNumber"][idx]),
        run_number=np.asarray(arr["runNumber"][idx]),
        channel_number=np.asarray(arr["channelNumber"][idx]),
        label=np.asarray(arr["label"][idx]),
        mc_weight=np.asarray(arr["mcWeight"][idx]),
    )


def format_event(ev: EventBatch, i: int) -> str:
    """Human-readable dump of event i for debugging/inspection."""
    lines = [
        f"event #{i}: file_row={ev.event_index[i]} "
        f"run={ev.run_number[i]} eventNumber={ev.event_number[i]} "
        f"channel={ev.channel_number[i]} label={ev.label[i]} "
        f"mcWeight={ev.mc_weight[i]:.5g}"
    ]
    for slot in range(N_SLOTS):
        name = TOKEN_NAMES[int(ev.token_types[i, slot])]
        mark = " " if ev.valid[i, slot] else " (padding)"
        p = ev.fourvec[i, slot]
        lines.append(
            f"  slot {slot} {name:>4s}{mark}: "
            f"px={p[0]:9.3f} py={p[1]:9.3f} pz={p[2]:9.3f} E={p[3]:9.3f} GeV"
        )
    return "\n".join(lines)

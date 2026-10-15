"""MiniHEP data loader: ATLAS Open Data WW events -> fixed token sequences.

Reads ``dataWW_d1.root`` (tree ``tree_event``) with uproot and tokenizes each
event with exactly two leptons into a fixed 5-slot sequence::

    slot 0: lepton 0
    slot 1: lepton 1
    slot 2: jet 0 (or padding)
    slot 3: jet 1 (or padding)
    slot 4: MET transverse proxy

Each token carries a four-vector ``(px, py, pz, E)`` in GeV.  Token types
follow the seven-ID scheme used by the MiniHEP model::

    0 PAD, 1 e-, 2 e+, 3 mu-, 4 mu+, 5 jet, 6 MET

This is a self-supervised teaching exercise.  The ``label``, event identity and
MC weights are returned separately from the model inputs and are never used to
represent a token.

Only numpy and uproot are required here; no ROOT, JAX, awkward or GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import uproot

# ----------------------------------------------------------------------------
# Token scheme (seven type IDs; 0 is padding)
# ----------------------------------------------------------------------------
TOK_PAD = 0
TOK_EM = 1    # e-
TOK_EP = 2    # e+
TOK_MUM = 3   # mu-
TOK_MUP = 4   # mu+
TOK_JET = 5   # jet (this file has no jet flavour/type branch)
TOK_MET = 6   # missing transverse momentum proxy

TOKEN_NAMES = {
    TOK_PAD: "PAD",
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
N_TOKEN_TYPES = 7

_MEV_TO_GEV = 1.0 / 1000.0  # file stores lepton/jet E in MeV
_PDG_ELECTRON = 11
_PDG_MUON = 13
_JET_ABSENT_FILLER = -7.0  # never used: absent jets become zero padding

# Every branch the loader needs, in a single read.
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
    """One tokenized (sub)set of events.

    Attributes
    ----------
    fourvec : (N, 5, 4) float32   token (px, py, pz, E) in GeV; zero padded
    token_types : (N, 5) int64    codes from TOK_* / TOKEN_NAMES
    valid : (N, 5) bool           False for padded (absent) tokens
    event_index/event_number/run_number/channel_number : (N,) int64
        event identity; event_index is the row in the original file
    label : (N,) int64            ground truth (1 = signal), NOT a model input
    mc_weight : (N,) float64      MC weight, NOT a model input
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
        if self.fourvec.shape != (n, N_SLOTS, N_FEATURES):
            raise ValueError(
                f"fourvec must be {(n, N_SLOTS, N_FEATURES)}, got {self.fourvec.shape}"
            )
        for name in ("token_types", "valid"):
            arr = getattr(self, name)
            if arr.shape != (n, N_SLOTS):
                raise ValueError(f"{name} must be {(n, N_SLOTS)}, got {arr.shape}")
        if self.valid.dtype != np.bool_:
            raise ValueError("valid must be a bool array")
        if self.fourvec.dtype != np.float32:
            raise ValueError("fourvec must be float32")

    @property
    def n_events(self) -> int:
        return len(self.label)


def repo_root() -> Path:
    """The directory that contains dataWW_d1.root (parents[1] of this file)."""
    return Path(__file__).resolve().parents[1]


def lepton_token_type(pdg_flavor: np.ndarray, charge: np.ndarray) -> np.ndarray:
    """Map (PDG flavour, charge) to a token code: e-1, e+2, mu-3, mu+4."""
    flavor = np.asarray(pdg_flavor)
    chg = np.asarray(charge)
    if not np.all(np.isin(flavor, (_PDG_ELECTRON, _PDG_MUON))):
        raise ValueError("unexpected lepton PDG flavour in file")
    if not np.all(np.abs(chg) == 1):
        raise ValueError("unexpected lepton charge in file")
    base = np.where(flavor == _PDG_MUON, TOK_MUM, TOK_EM).astype(np.int64)
    return base + (chg > 0).astype(np.int64)


def _vec4(pt, eta, phi, e_gev) -> np.ndarray:
    """(px, py, pz, E) given (pt, eta, phi) and E already in GeV."""
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


def load_events(path: Path | str | None = None, max_events: int = 6000,
                seed: int = 42) -> EventBatch:
    """Load and tokenize at most ``max_events`` events with ``lep_n == 2``.

    Deterministic for a given (file, max_events, seed).  Absent jets become
    zero padding (the file's -7 fillers are never used).  The MET token is the
    transverse proxy (MET cos(phi), MET sin(phi), 0, MET), not a reconstructed
    neutrino-pair four-vector.
    """
    if path is None:
        path = repo_root() / "dataWW_d1.root"
    tree = uproot.open(str(path))["tree_event"]
    arr = tree.arrays(_BRANCHES, library="np")

    keep = np.asarray(arr["lep_n"]) == 2
    selected = np.flatnonzero(keep)
    rng = np.random.default_rng(seed)
    take = rng.permutation(selected.size)[: min(max_events, selected.size)]
    idx = selected[take]
    n = idx.size

    fourvec = np.zeros((n, N_SLOTS, N_FEATURES), dtype=np.float32)
    token_types = np.full((n, N_SLOTS), TOK_PAD, dtype=np.int64)
    valid = np.zeros((n, N_SLOTS), dtype=np.bool_)

    # --- leptons (always present: filtered lep_n == 2) ------------------------
    for slot, k in ((SLOT_LEP0, 0), (SLOT_LEP1, 1)):
        e_gev = np.asarray(arr[f"lep_E_{k}"][idx]) * _MEV_TO_GEV
        fourvec[:, slot] = _vec4(
            arr[f"lep_pt_{k}"][idx], arr[f"lep_eta_{k}"][idx],
            arr[f"lep_phi_{k}"][idx], e_gev,
        ).astype(np.float32)
        token_types[:, slot] = lepton_token_type(
            arr[f"lep_type_{k}"][idx], arr[f"lep_charge_{k}"][idx]
        )
    valid[:, SLOT_LEP0] = True
    valid[:, SLOT_LEP1] = True

    # --- jets: absent slots are zero vectors; never use the -7 fillers ---------
    for slot, k in ((SLOT_JET0, 0), (SLOT_JET1, 1)):
        present = np.asarray(arr[f"jet_pt_{k}"][idx], dtype=np.float64) != _JET_ABSENT_FILLER
        e_gev = np.asarray(arr[f"jet_E_{k}"][idx]) * _MEV_TO_GEV
        vec = _vec4(arr[f"jet_pt_{k}"][idx], arr[f"jet_eta_{k}"][idx],
                    arr[f"jet_phi_{k}"][idx], e_gev)
        fourvec[:, slot] = np.where(present[:, None], vec, 0.0).astype(np.float32)
        token_types[:, slot] = np.where(present, TOK_JET, TOK_PAD).astype(np.int64)
        valid[:, slot] = present

    # --- MET transverse proxy ---------------------------------------------------
    met_et = np.asarray(arr["met_et"][idx], dtype=np.float64)
    met_phi = np.asarray(arr["met_phi"][idx], dtype=np.float64)
    fourvec[:, SLOT_MET] = np.stack(
        (met_et * np.cos(met_phi), met_et * np.sin(met_phi),
         np.zeros(n), met_et),
        axis=-1,
    ).astype(np.float32)
    token_types[:, SLOT_MET] = TOK_MET
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


def select_events(ev: EventBatch, idx: np.ndarray) -> EventBatch:
    """Return a new EventBatch restricted to the given (pre-sorted) indices."""
    idx = np.asarray(idx, dtype=np.int64)
    return EventBatch(
        fourvec=ev.fourvec[idx],
        token_types=ev.token_types[idx],
        valid=ev.valid[idx],
        event_index=ev.event_index[idx],
        event_number=ev.event_number[idx],
        run_number=ev.run_number[idx],
        channel_number=ev.channel_number[idx],
        label=ev.label[idx],
        mc_weight=ev.mc_weight[idx],
    )


def split_events(ev: EventBatch, fractions=(0.7, 0.15, 0.15)) -> tuple:
    """Deterministic, disjoint 70/15/15 split ordered by event index.

    Returns (train, val, test) EventBatches.  Ordering by ``event_index`` makes
    the split reproducible and keeps the three sets disjoint.
    """
    order = np.argsort(ev.event_index, kind="stable")
    n = len(order)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    return (
        select_events(ev, order[:n_train]),
        select_events(ev, order[n_train:n_train + n_val]),
        select_events(ev, order[n_train + n_val:]),
    )


def format_event(ev: EventBatch, i: int) -> str:
    """Human-readable dump of event i for inspection."""
    lines = [
        f"event #{i}: file_row={ev.event_index[i]} run={ev.run_number[i]} "
        f"eventNumber={ev.event_number[i]} channel={ev.channel_number[i]} "
        f"label={ev.label[i]} mcWeight={ev.mc_weight[i]:.5g}"
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

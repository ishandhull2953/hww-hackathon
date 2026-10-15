"""Reproducible 70/15/15 event split and feature scaling for minihep.

Split
-----
Splits are assigned by a stable hash of the event's file row (`event_index`),
so the same event always lands in the same split no matter which subsample was
loaded (different `max_events` / `seed` in `load_events`).  The splits are
disjoint by construction.  A plain hash is used (splitmix64) instead of
Python's built-in `hash`, which is salted per process and not reproducible.

Scaling
-------
Per event, only *valid* tokens (from training events) are used to fit:
first `asinh(component / 50 GeV)`, then per-component standardization
(mean/std of the 4 components, ddof=0).  All splits are transformed with the
training statistics; padded (invalid) tokens are reset to exactly zero after
transforming.

Event identity, labels and MC weights are metadata and are never token
features.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict

import numpy as np

from .data import EventBatch

ASINH_SCALE_GEV = 50.0
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
_NUM_BUCKETS = 100
_MASK64 = (1 << 64) - 1


# ----------------------------------------------------------------------------
# Split
# ----------------------------------------------------------------------------
def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Deterministic, process-independent 64-bit hash (splitmix64)."""
    x = np.asarray(x, dtype=np.uint64)
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (x ^ (x >> 31)) & _MASK64


def split_masks(event_index: np.ndarray) -> Dict[str, np.ndarray]:
    """Return disjoint boolean masks for the 70/15/15 split keyed on
    `event_index` (row in dataWW_d1.root)."""
    bucket = _splitmix64(np.asarray(event_index)) % _NUM_BUCKETS
    return {
        "train": bucket < TRAIN_FRAC * _NUM_BUCKETS,
        "val": (bucket >= TRAIN_FRAC * _NUM_BUCKETS)
        & (bucket < (TRAIN_FRAC + VAL_FRAC) * _NUM_BUCKETS),
        "test": bucket >= (TRAIN_FRAC + VAL_FRAC) * _NUM_BUCKETS,
    }


def _subset(ev: EventBatch, mask: np.ndarray) -> EventBatch:
    """EventBatch restricted to the events selected by boolean `mask`."""
    return EventBatch(
        fourvec=ev.fourvec[mask],
        token_types=ev.token_types[mask],
        valid=ev.valid[mask],
        event_index=ev.event_index[mask],
        event_number=ev.event_number[mask],
        run_number=ev.run_number[mask],
        channel_number=ev.channel_number[mask],
        label=ev.label[mask],
        mc_weight=ev.mc_weight[mask],
    )


# ----------------------------------------------------------------------------
# Scaler
# ----------------------------------------------------------------------------
class FeatureScaler:
    """asinh(x / scale) then per-component standardisation.

    Fit on valid tokens only; `event_mask` optionally restricts which events
    are used (e.g. training events only).  `transform` zeroes padded tokens.
    """

    def __init__(self, scale_gev: float = ASINH_SCALE_GEV) -> None:
        self.scale_gev = float(scale_gev)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def _asinh(self, fourvec: np.ndarray) -> np.ndarray:
        return np.arcsinh(np.asarray(fourvec, dtype=np.float64) / self.scale_gev)

    def fit(
        self,
        fourvec: np.ndarray,
        valid: np.ndarray,
        event_mask: np.ndarray | None = None,
    ) -> "FeatureScaler":
        """Fit on valid tokens of the (optionally masked) events."""
        fourvec = np.asarray(fourvec, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool)
        if event_mask is not None:
            event_mask = np.asarray(event_mask, dtype=bool)
            fourvec = fourvec[event_mask]
            valid = valid[event_mask]
        tokens = self._asinh(fourvec)[valid]
        if tokens.size == 0:
            raise ValueError("no valid tokens to fit the scaler on")
        self.mean = tokens.mean(axis=0)
        self.std = tokens.std(axis=0)
        if np.any(self.std == 0):
            raise ValueError("a component has zero std across valid tokens")
        return self

    def transform(self, fourvec: np.ndarray, valid: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("fit() must be called before transform()")
        out = (self._asinh(fourvec) - self.mean) / self.std
        out = np.where(np.asarray(valid, dtype=bool)[..., None], out, 0.0)
        return out.astype(np.float32)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
@dataclass
class Prepared:
    """Split event batches with scaled features plus the fitted scaler."""

    scaler: FeatureScaler
    train: EventBatch
    val: EventBatch
    test: EventBatch
    masks: Dict[str, np.ndarray]  # masks over the original, unscaled batch


def prepare(ev: EventBatch, scaler: FeatureScaler | None = None) -> Prepared:
    """Split `ev` 70/15/15 by event_index, fit the scaler on validity tokens
    of training events only, and transform all splits with those stats."""
    masks = split_masks(ev.event_index)
    train = _subset(ev, masks["train"])
    val = _subset(ev, masks["val"])
    test = _subset(ev, masks["test"])
    if scaler is None:
        scaler = FeatureScaler()
    scaler.fit(train.fourvec, train.valid)
    train = replace(train, fourvec=scaler.transform(train.fourvec, train.valid))
    val = replace(val, fourvec=scaler.transform(val.fourvec, val.valid))
    test = replace(test, fourvec=scaler.transform(test.fourvec, test.valid))
    return Prepared(scaler=scaler, train=train, val=val, test=test, masks=masks)


def summarize(prepared: Prepared) -> str:
    """Split sizes, label ratio, and a no-overlap check on event_index."""
    lines = []
    seen: set[int] = set()
    for name in ("train", "val", "test"):
        split_ev = getattr(prepared, name)
        ids = set(int(i) for i in split_ev.event_index)
        overlap = seen & ids
        seen |= ids
        lines.append(
            f"{name:5s}: n={split_ev.n_events:5d}  label_1={int((split_ev.label == 1).sum()):5d} "
            f"label_0={int((split_ev.label == 0).sum()):5d}  overlap_with_earlier_splits={len(overlap)}"
        )
    total_all = sum(getattr(prepared, n).n_events for n in ("train", "val", "test"))
    lines.append(f"total: {total_all}")
    return "\n".join(lines)

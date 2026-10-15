"""Feature scaling for MiniHEP four-vector tokens.

The transform is applied per component (px, py, pz, E) of the four-vector::

    1.  asinh(component / 50 GeV)     (compresses the wide tails)
    2.  standardize each component    (z = (x - mean) / std)

The scaler is fitted using **only the valid tokens of the training split**, so
the validation and test splits never influence the fitted statistics.  Padded
tokens are reset to zero after transform.  Metadata (event identity, label, MC
weight) is never scaled.
"""

from __future__ import annotations

import numpy as np

from .data import EventBatch, N_FEATURES

_ASINH_SCALE_GEV = 50.0  # the asinh argument, in GeV


class FourVecScaler:
    """asinh-then-standardize scaler for the four components of a token."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None  # (4,)
        self.std: np.ndarray | None = None   # (4,)
        self._fitted = False

    def fit(self, fourvec: np.ndarray, valid: np.ndarray) -> "FourVecScaler":
        """Fit using only the valid tokens of the (training) events given."""
        x = np.arcsinh(np.asarray(fourvec, dtype=np.float64) / _ASINH_SCALE_GEV)
        vals = x[np.asarray(valid, dtype=bool)]
        self.mean = vals.mean(axis=0)
        self.std = vals.std(axis=0) + 1e-8
        self._fitted = True
        return self

    def transform(self, fourvec: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Apply asinh + standardization, then zero the padded tokens."""
        if not self._fitted:
            raise RuntimeError("FourVecScaler must be fit before transform")
        x = np.arcsinh(np.asarray(fourvec, dtype=np.float64) / _ASINH_SCALE_GEV)
        x = (x - self.mean) / self.std
        return np.where(np.asarray(valid, dtype=bool)[..., None], x, 0.0)

    fit_transform = lambda self, fourvec, valid: self.fit(fourvec, valid).transform(fourvec, valid)  # noqa: E731


def fit_scaler_for_split(train: EventBatch) -> FourVecScaler:
    """Convenience: fit a scaler on the training split's valid tokens."""
    return FourVecScaler().fit(train.fourvec, train.valid)


def transform_batch(scaler: FourVecScaler, ev: EventBatch) -> np.ndarray:
    """Scaled four-vectors for a whole batch (float32)."""
    return scaler.transform(ev.fourvec, ev.valid).astype(np.float32)

"""MiniHEP: a small masked four-vector transformer trained on ATLAS Open Data WW events.

Modules
-------
data   : load + tokenize events from dataWW_d1.root (7-ID token scheme)
scale  : asinh + standardize scaler fitted on training-only valid tokens
model  : the MiniHEP transformer encoder (nn.Module)
train  : masking, training loop, loss helpers
"""

from .data import (  # noqa: F401
    EventBatch, load_events, split_events, select_events, format_event,
    TOKEN_NAMES, N_SLOTS, N_FEATURES, N_TOKEN_TYPES,
)
from .scale import FourVecScaler, fit_scaler_for_split, transform_batch  # noqa: F401

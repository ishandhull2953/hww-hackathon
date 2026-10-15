"""Masking, loss and the CPU training loop for MiniHEP.

Masking
-------
About 25% of the **valid** tokens of each event are masked per training pass
(always at least one per event, always at least one left visible).  A masked
token keeps its type but loses all four continuous input values; padding is
never masked or scored.  The reconstruction loss is the mean squared error over
the four scaled components of the hidden tokens only.

Training
--------
AdamW, batch size 64, learning rate 1e-3, a fixed number of epochs, one CPU
thread.  Training masks are resampled every pass; a single fixed, seeded mask
sequence is reused for validation so the validation curve is comparable across
epochs.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .data import EventBatch, N_SLOTS, N_FEATURES
from .model import MiniHEP
from .scale import FourVecScaler

DEVICE = torch.device("cpu")


def mask_valid(valid: np.ndarray, rng: np.random.Generator, rate: float = 0.25) -> np.ndarray:
    """Return a (B, 5) bool mask with ~rate of each event's valid tokens hidden.

    Always at least one token is masked, and at least one valid token is left
    visible.  Padded (invalid) tokens are never masked.
    """
    valid = np.asarray(valid, dtype=bool)
    b, s = valid.shape
    hidden = np.zeros((b, s), dtype=bool)
    for i in range(b):
        pos = np.flatnonzero(valid[i])
        n = pos.size
        if n < 2:  # nothing sensible to hide; keep all visible
            continue
        k = max(1, int(round(rate * n)))
        k = min(k, n - 1)  # leave at least one visible token
        hidden[i, rng.choice(pos, size=k, replace=False)] = True
    return hidden


def to_tensors(ev: EventBatch, scaler: FourVecScaler) -> tuple:
    """Scaled four-vectors, type ids and valid masks as (CPU) torch tensors."""
    x = torch.from_numpy(scaler.transform(ev.fourvec, ev.valid).astype(np.float32))
    tt = torch.from_numpy(ev.token_types.astype(np.int64))
    v = torch.from_numpy(ev.valid.astype(np.bool_))
    return x, tt, v


def masked_recon_loss(
    model: MiniHEP,
    x: torch.Tensor,
    tt: torch.Tensor,
    valid: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error over the four scaled components of hidden tokens."""
    out = model(x, tt, valid, hidden)
    mask = hidden.float()[..., None]  # (B, 5, 1)
    diff = (out - x) ** 2
    denom = float(mask.sum() * N_FEATURES)
    if denom == 0.0:
        return torch.tensor(0.0, device=x.device)
    return (diff * mask).sum() / denom


def _epoch(
    model: MiniHEP,
    x: torch.Tensor,
    tt: torch.Tensor,
    valid: torch.Tensor,
    batch_size: int,
    seed: int,
) -> float:
    """Not used by ``train_model``; kept as a deterministic-one-pass helper."""
    n = len(x)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    model.train(False)
    total, nb = 0.0, 0
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        hid = torch.from_numpy(mask_valid(valid[idx].numpy(), np.random.default_rng()))
        loss = masked_recon_loss(model, x[idx], tt[idx], valid[idx], hid)
        total += float(loss) * idx.size
        nb += idx.size
    return total / max(nb, 1)


def train_model(
    model: MiniHEP,
    Xtr, TTtr, Vtr,
    Xva, TTva, Vva,
    epochs: int = 4,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """Train on CPU with AdamW and return ([train_loss], [val_loss]) per epoch.

    Training masks are resampled each pass; a fixed seeded masking sequence is
    used for validation so epochs are comparable.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    val_masks = [
        mask_valid(Vva[start:start + batch_size].numpy(), np.random.default_rng(seed + 1))
        for start in range(0, len(Vva), batch_size)
    ]
    train_hist, val_hist = [], []
    for epoch in range(epochs):
        model.train(True)
        opt.zero_grad(set_to_none=True)
        n = len(Xtr)
        rng = np.random.default_rng(seed + 1000 + epoch)
        perm = rng.permutation(n)
        running, counts = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, ttb, vb = Xtr[idx], TTtr[idx], Vtr[idx]
            hidden = torch.from_numpy(mask_valid(vb.numpy(), np.random.default_rng()))
            loss = masked_recon_loss(model, xb, ttb, vb, hidden)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.detach().item() * idx.size
            counts += idx.size
        train_hist.append(running / max(counts, 1))

        model.train(False)
        running, counts = 0.0, 0
        with torch.no_grad():
            for bi, start in enumerate(range(0, len(Vva), batch_size)):
                idx = np.arange(start, min(start + batch_size, len(Vva)))
                hid = torch.from_numpy(val_masks[bi])
                running += masked_recon_loss(model, Xva[idx], TTva[idx], Vva[idx], hid).item() * idx.size
                counts += idx.size
        val_hist.append(running / max(counts, 1))
    return train_hist, val_hist


def evaluate_recon(model: MiniHEP, x, tt, valid, mask: np.ndarray) -> float:
    """Mean masked reconstruction error on a fixed mask (in scaled components)."""
    with torch.no_grad():
        return float(masked_recon_loss(model, x, tt, valid, torch.from_numpy(mask)).item())

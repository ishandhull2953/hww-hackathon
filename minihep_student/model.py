"""The MiniHEP masked four-vector transformer (``MiniHEP``).

This is the fixed first-draft architecture from the MiniHEP tutorial.  It is a
small transformer encoder that reconstructs the (scaled) four-vector of a
hidden object token from the visible objects of the same event.  The class
name is ``MiniHEP``; it is not a distinct transformer architecture.

Interfaces (see the tutorial prompt)::

    forward(x, token_type, valid, hidden) -> (B, 5, 4)
        x          : (B, 5, 4)  standardized four-vectors (float)
        token_type : (B, 5)     integer type IDs (0..6)
        valid      : (B, 5)     bool, False for padded tokens
        hidden     : (B, 5)     bool, True for tokens to reconstruct

    encode_events(x, token_type, valid) -> (B, 32)
        runs with no hidden tokens and averages only valid encoder states.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .data import N_SLOTS, N_FEATURES, N_TOKEN_TYPES

D_MODEL = 32
D_FEEDFORWARD = 64
N_HEAD = 4
N_LAYERS = 2
DROPOUT = 0.1


class MiniHEP(nn.Module):
    """A small masked transformer encoder over five object tokens."""

    def __init__(self, d_model: int = D_MODEL, n_layers: int = N_LAYERS) -> None:
        super().__init__()
        self.input_proj = nn.Linear(N_FEATURES, d_model)      # 4 -> d_model
        self.type_embed = nn.Embedding(N_TOKEN_TYPES, d_model, padding_idx=0)
        self.mask_embed = nn.Parameter(torch.randn(d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=N_HEAD,
            dim_feedforward=D_FEEDFORWARD,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        # The two encoder blocks must be initialized independently so they do
        # not start from identical weights.
        for layer in self.encoder.layers:
            self._reset_layer(layer)

        self.output_proj = nn.Linear(d_model, N_FEATURES)     # d_model -> 4
        self._init_output_proj()

    @staticmethod
    def _reset_layer(layer: nn.TransformerEncoderLayer) -> None:
        """Re-initialize every parameter of one encoder layer from scratch.

        ``nn.TransformerEncoder`` deepcopies a single prototype layer, so
        without this the two blocks would start from identical weights.  Every
        parameter must be reset, including the ``MultiheadAttention`` raw
        ``in_proj_weight``/``in_proj_bias`` (which are Parameters on the MHA
        module, not ``nn.Linear`` submodules and so are missed by the
        ``nn.Linear`` branch alone).
        """
        for module in layer.modules():
            if isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _init_output_proj(self) -> None:
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _encode(
        self,
        x: torch.Tensor,
        token_type: torch.Tensor,
        valid: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Run the encoder and return the raw (B, 5, d_model) token states."""
        # Zero the continuous inputs where a token is hidden or padded.
        keep = (~hidden).float()[..., None] * valid.float()[..., None]
        h = self.input_proj(x * keep)
        h = h + self.type_embed(token_type)
        # One learned mask embedding is added at hidden positions.
        h = h + self.mask_embed * hidden.float()[..., None]

        key_padding_mask = ~valid  # True entries are ignored by attention
        return self.encoder(h, src_key_padding_mask=key_padding_mask)

    def forward(
        self,
        x: torch.Tensor,
        token_type: torch.Tensor,
        valid: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Run the encoder and predict each token's four scaled components."""
        return self.output_proj(self._encode(x, token_type, valid, hidden))

    def encode_events(
        self,
        x: torch.Tensor,
        token_type: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Encode events with no hidden tokens; average only valid states."""
        hidden = torch.zeros_like(valid, dtype=torch.bool)
        h = self._encode(x, token_type, valid, hidden)
        mask = valid.float()[..., None]
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_parameter_count(model: nn.Module) -> int:
    n = count_parameters(model)
    print(f"MiniHEP trainable parameters: {n:,}")
    return n


if __name__ == "__main__":
    # Sanity dump of the parameter count for the default sizes.
    m = MiniHEP()
    total = print_parameter_count(m)
    expected = 17636
    print(f"(reference size is {expected:,}; delta {total - expected:+d})")
    x = torch.randn(3, N_SLOTS, N_FEATURES)
    t = torch.randint(1, N_TOKEN_TYPES, (3, N_SLOTS))
    v = torch.ones(3, N_SLOTS, dtype=torch.bool)
    h = torch.zeros(3, N_SLOTS, dtype=torch.bool)
    out = m(x, t, v, h)
    emb = m.encode_events(x, t, v)
    print("forward ->", tuple(out.shape), " encode_events ->", tuple(emb.shape))

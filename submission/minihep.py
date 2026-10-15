"""MiniHEP for the pipeline: numpy tokenizer and a numpy forward pass.

Pure numpy + scipy (no torch at analysis time).  The architecture mirrors
minihep_student/model.py exactly (masked four-vector transformer encoder,
d_model=32, 2 layers, 4 heads, norm_first, GELU, mean pooling over valid
tokens) plus a trained analysis head (signal-vs-background score).

The weights travel as text in model.json (scaler statistics, backbone
weights, head weights), produced by minihep_student/train_head.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import erf

N_SLOTS = 5
N_FEATURES = 4
N_TOKEN_TYPES = 7
D_MODEL = 32
N_HEAD = 4
HEAD_DIM = D_MODEL // N_HEAD
ASINH_SCALE_GEV = 50.0
LN_EPS = 1e-5

TOK_EM, TOK_EP, TOK_MUM, TOK_MUP, TOK_JET, TOK_MET = 1, 2, 3, 4, 5, 6


# ---------------------------------------------------------------- tokenizer --

def lepton_code(flavor, charge):
    """PDG flavour/charge -> token type (e-1, e+2, mu-3, mu+4)."""
    flavor = np.asarray(flavor)
    base = np.where(flavor == 13, TOK_MUM, TOK_EM).astype(np.int64)
    return base + (np.asarray(charge) > 0).astype(np.int64)


def tokenize(ev):
    """A dataset's event dict (DATASET.md columns) -> (fourvec, types, valid).

    fourvec (N, 5, 4) float32 tokens (px, py, pz, E) in GeV, zero padded;
    token_types (N, 5) int64; valid (N, 5) bool.  Leptons and jets are taken
    massless (E = pT cosh eta); MET is the transverse proxy
    (MET cos phi, MET sin phi, 0, MET) exactly as minihep_student/data.py.
    """
    n = len(ev["lep_pt_0"])
    fourvec = np.zeros((n, N_SLOTS, N_FEATURES), dtype=np.float32)
    types = np.zeros((n, N_SLOTS), dtype=np.int64)
    valid = np.zeros((n, N_SLOTS), dtype=np.bool_)

    for slot, k in ((0, 0), (1, 1)):
        pt = ev[f"lep_pt_{k}"]; eta = ev[f"lep_eta_{k}"]; phi = ev[f"lep_phi_{k}"]
        fourvec[:, slot] = fourvec_vector(pt, eta, phi, pt * np.cosh(eta))
        types[:, slot] = lepton_code(ev[f"lep_type_{k}"], ev[f"lep_charge_{k}"])
        valid[:, slot] = True

    for slot, k in ((2, 0), (3, 1)):
        pt = np.asarray(ev[f"jet_pt_{k}"], dtype=np.float32)
        present = pt > 0.0
        eta = ev[f"jet_eta_{k}"]; phi = ev[f"jet_phi_{k}"]
        vec = fourvec_vector(pt, eta, phi, pt * np.cosh(eta))
        fourvec[:, slot] = np.where(present[:, None], vec, 0.0).astype(np.float32)
        types[:, slot] = np.where(present, TOK_JET, 0).astype(np.int64)
        valid[:, slot] = present

    met = np.asarray(ev["met_et"], dtype=np.float32); mphi = np.asarray(ev["met_phi"], dtype=np.float32)
    fourvec[:, 4] = np.stack((met * np.cos(mphi), met * np.sin(mphi),
                              np.zeros(n), met), axis=-1).astype(np.float32)
    types[:, 4] = TOK_MET
    valid[:, 4] = True
    return fourvec, types, valid


def fourvec_vector(pt, eta, phi, e):
    pt = np.asarray(pt, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)
    eta = np.asarray(eta, dtype=np.float32)
    return np.stack((pt * np.cos(phi), pt * np.sin(phi),
                     pt * np.sinh(eta), np.asarray(e, dtype=np.float32)), axis=-1)


def scale_fourvec(fourvec, valid, mean, std):
    """asinh-then-standardize (minihep_student/scale.py), zero padded tokens."""
    x = np.arcsinh(fourvec.astype(np.float64) / ASINH_SCALE_GEV)
    x = (x - mean) / std
    return np.where(valid[..., None], x, 0.0).astype(np.float32)


# ------------------------------------------------------------- numpy model --

FEATURE_NAMES = ["lep_pt_0", "lep_pt_1", "met_et", "dphi", "mll", "ptll", "mt",
                 "jet_n", "deta", "same_flavour"]


def features(ev):
    """The engineered feature matrix (same 10 features and order as the kit's)."""
    import hww as H
    k = H.kinematics(ev)
    cols = {"lep_pt_0": ev["lep_pt_0"], "lep_pt_1": ev["lep_pt_1"], "met_et": ev["met_et"],
            "dphi": k["dphi"], "mll": k["mll"], "ptll": k["ptll"], "mt": k["mt"],
            "jet_n": ev["jet_n"],
            "deta": np.abs(ev["lep_eta_0"].astype(float) - ev["lep_eta_1"].astype(float)),
            "same_flavour": (ev["lep_type_0"] == ev["lep_type_1"]).astype(float)}
    return np.column_stack([np.asarray(cols[f], float) for f in FEATURE_NAMES])


class MiniHEPNumpy:
    """Forward pass of the trained MiniHEP backbone + analysis head."""

    def __init__(self, model: dict):
        m = model["model"]
        self.mean = np.array(model["scaler"]["mean"], dtype=np.float32)
        self.std = np.array(model["scaler"]["std"], dtype=np.float32)
        self.inp_w = np.array(m["input_proj"]["weight"], dtype=np.float32)  # (32,4)
        self.inp_b = np.array(m["input_proj"]["bias"], dtype=np.float32)    # (32,)
        self.type_w = np.array(m["type_embed"], dtype=np.float32)           # (7,32)
        self.mask_emb = np.array(m["mask_embed"], dtype=np.float32)         # (32,)
        self.layers = []
        for lm in m["encoder_layers"]:
            wq = np.array(lm["in_proj_weight"][:D_MODEL], dtype=np.float32)
            wk = np.array(lm["in_proj_weight"][D_MODEL:2 * D_MODEL], dtype=np.float32)
            wv = np.array(lm["in_proj_weight"][2 * D_MODEL:], dtype=np.float32)
            bq = np.array(lm["in_proj_bias"][:D_MODEL], dtype=np.float32)
            bk = np.array(lm["in_proj_bias"][D_MODEL:2 * D_MODEL], dtype=np.float32)
            bv = np.array(lm["in_proj_bias"][2 * D_MODEL:], dtype=np.float32)
            self.layers.append({
                "wqkv": np.concatenate([wq, wk, wv], axis=0),   # (96, 32)
                "bqkv": np.concatenate([bq, bk, bv], axis=0),   # (96,)
                "out_w": np.array(lm["out_proj"]["weight"], dtype=np.float32),
                "out_b": np.array(lm["out_proj"]["bias"], dtype=np.float32),
                "lin1_w": np.array(lm["linear1"]["weight"], dtype=np.float32),
                "lin1_b": np.array(lm["linear1"]["bias"], dtype=np.float32),
                "lin2_w": np.array(lm["linear2"]["weight"], dtype=np.float32),
                "lin2_b": np.array(lm["linear2"]["bias"], dtype=np.float32),
                "n1_w": np.array(lm["norm1"]["weight"], dtype=np.float32),
                "n1_b": np.array(lm["norm1"]["bias"], dtype=np.float32),
                "n2_w": np.array(lm["norm2"]["weight"], dtype=np.float32),
                "n2_b": np.array(lm["norm2"]["bias"], dtype=np.float32),
            })
        self.head = [(np.array(l["W"], dtype=np.float32), np.array(l["b"], dtype=np.float32))
                     for l in model["head"]["layers"]]
        self.head_act = model["head"].get("activation", "relu")
        self.head_features = model["head"].get("features")
        if self.head_features:
            self.fmean = np.array(model["head"]["feature_mean"], dtype=np.float32)
            self.fstd = np.array(model["head"]["feature_std"], dtype=np.float32)

    @staticmethod
    def _layernorm(x, w, b):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mu) / np.sqrt(var + LN_EPS) * w + b

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

    def _attention(self, x, key_mask, layer, chunk=65536):
        """Self-attention with key padding (True = ignore), returns (B,5,d)."""
        b, s, d = x.shape
        scale = 1.0 / np.sqrt(HEAD_DIM)
        out = np.empty((b, s, d), dtype=np.float32)
        wqkv_T = layer["wqkv"].T          # (32, 96)
        for i in range(0, b, chunk):
            xc = x[i:i + chunk]
            xr = xc.reshape(-1, d)
            qkv = xr @ wqkv_T + layer["bqkv"]          # (B*5, 96)
            q = qkv[:, :d].reshape(xc.shape[0], s, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)
            k = qkv[:, d:2 * d].reshape(xc.shape[0], s, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)
            v = qkv[:, 2 * d:].reshape(xc.shape[0], s, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)
            scores = q @ k.transpose(0, 1, 3, 2) * scale          # (B,H,5,5)
            pad = ~key_mask[i:i + chunk]                           # (B,5)
            scores = np.where(pad[:, None, None, :], -1e9, scores)
            scores = scores - scores.max(axis=-1, keepdims=True)
            attn = np.exp(scores)
            attn = attn / attn.sum(axis=-1, keepdims=True)
            ctx = attn @ v                                          # (B,H,5,hd)
            ctx = ctx.transpose(0, 2, 1, 3).reshape(ctx.shape[0], s, d)
            out[i:i + chunk] = ctx @ layer["out_w"].T + layer["out_b"]
        return out

    def encode_events(self, fourvec, types, valid, chunk=65536):
        """(N,5,4)+(N,5) -> (N,32) mean-pooled event embedding."""
        x = scale_fourvec(fourvec, valid, self.mean, self.std)
        b = x.shape[0]
        h = np.empty((b, N_SLOTS, D_MODEL), dtype=np.float32)
        for i in range(0, b, chunk):
            xc = x[i:i + chunk]
            keep = valid[i:i + chunk].astype(np.float32)[..., None]
            z = xc * keep
            hc = z @ self.inp_w.T + self.inp_b
            hc = hc + self.type_w[types[i:i + chunk]]
            for layer in self.layers:
                hn = self._layernorm(hc, layer["n1_w"], layer["n1_b"])
                hc = hc + self._attention(hn, valid[i:i + chunk], layer)
                hn = self._layernorm(hc, layer["n2_w"], layer["n2_b"])
                f = hn @ layer["lin1_w"].T + layer["lin1_b"]
                f = self._gelu(f)
                f = f @ layer["lin2_w"].T + layer["lin2_b"]
                hc = hc + f
            h[i:i + chunk] = hc
        mask = valid.astype(np.float32)[..., None]
        pooled = (h * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1.0)
        return pooled

    def score(self, fourvec, types, valid, feats=None):
        """Signal-vs-background probability from the analysis head.

        ``feats``: (N, 10) engineered features, required when the head was
        trained with them (model.json head.features set).
        """
        x = self.encode_events(fourvec, types, valid)
        if self.head_features:
            if feats is None:
                raise ValueError("this head was trained with engineered features; pass feats=")
            x = np.concatenate([x, (np.asarray(feats, float) - self.fmean) / self.fstd], axis=1)
        for j, (W, b) in enumerate(self.head):
            x = x @ W.T + b
            if j < len(self.head) - 1:
                x = self._gelu(x) if self.head_act == "gelu" else np.maximum(x, 0.0)
        return 1.0 / (1.0 + np.exp(-x[:, 0]))


def load_model(path):
    return MiniHEPNumpy(json.loads(Path(path).read_text()))

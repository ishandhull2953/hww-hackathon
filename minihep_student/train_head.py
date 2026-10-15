#!/usr/bin/env python3
"""Train the MiniHEP backbone (self-supervised, masked reconstruction) and the
analysis head (signal vs background) on the pool, and export them as text.

    python3 minihep_student/train_head.py [--pool datasets/public/seed_000/mc/pool.npz]
        [--out submission/model.json] [--pretrain-events 200000] [--pretrain-epochs 3]
        [--head-events 200000] [--head-epochs 40] [--seed 7]

The tokenizer is the numpy one from submission/minihep.py (massless leptons
and jets, MET transverse proxy), so the shipped pipeline sees exactly the same
tokens.  The backbone is pretrained on masked reconstruction; then the head
(an MLP on the mean-pooled 32-dim event embedding) is trained on the labels
with balanced signal/background classes.  Exports scaler + backbone + head as
json and verifies the numpy forward against the torch forward.

Dev-only: needs torch; the evaluation runs the exported model with numpy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))            # repo root
sys.path.insert(0, str(HERE.parent / "submission"))

import hww as H                                  # noqa: E402  (load)
from minihep_student.model import MiniHEP        # noqa: E402
from minihep_student.train import mask_valid     # noqa: E402
import minihep as M                              # noqa: E402  (numpy tokenizer/model/features)
from sklearn.metrics import roc_auc_score        # noqa: E402

torch.set_num_threads(2)
D_MODEL = 32


def split(all_x, all_tt, all_v, all_label, all_weight, all_feats):
    """Deterministic 70/15/15 train/val/test on the given (already shuffled) order."""
    n = len(all_label)
    n_tr = int(round(0.7 * n)); n_va = int(round(0.15 * n))
    tr = (all_x[:n_tr], all_tt[:n_tr], all_v[:n_tr], all_label[:n_tr], all_weight[:n_tr], all_feats[:n_tr])
    va = (all_x[n_tr:n_tr + n_va], all_tt[n_tr:n_tr + n_va], all_v[n_tr:n_tr + n_va],
          all_label[n_tr:n_tr + n_va], all_weight[n_tr:n_tr + n_va], all_feats[n_tr:n_tr + n_va])
    te = (all_x[n_tr + n_va:], all_tt[n_tr + n_va:], all_v[n_tr + n_va:],
          all_label[n_tr + n_va:], all_weight[n_tr + n_va:], all_feats[n_tr + n_va:])
    return tr, va, te


def fit_scaler(fourvec, valid):
    """asinh + standardization, fitted on the valid tokens given."""
    x = np.arcsinh(fourvec.astype(np.float64) / M.ASINH_SCALE_GEV)
    vals = x[np.asarray(valid, dtype=bool)]
    return vals.mean(axis=0), vals.std(axis=0) + 1e-8


def make_scaler(fourvec, valid, mean, std):
    def scale(fv_i, valid_i):
        x = np.arcsinh(fv_i.astype(np.float64) / M.ASINH_SCALE_GEV)
        x = (x - mean) / std
        return np.where(np.asarray(valid_i, dtype=bool)[..., None], x, 0.0).astype(np.float32)
    return scale


def emb_of(model, scale, x, tt, valid, batch=8192):
    """Event embeddings (N, 32) from the frozen backbone, as a numpy array."""
    out = []
    xs = torch.from_numpy(scale(x, valid))
    vs = torch.from_numpy(np.asarray(valid, dtype=bool))
    ts = torch.from_numpy(tt.astype(np.int64))
    for start in range(0, len(xs), batch):
        with torch.no_grad():
            out.append(model.encode_events(xs[start:start + batch], ts[start:start + batch],
                                           vs[start:start + batch]).numpy().astype(np.float32))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=str(HERE.parent / "datasets/public/seed_000/mc/pool.npz"))
    ap.add_argument("--out", default=str(HERE.parent / "submission/model.json"))
    ap.add_argument("--pretrain-events", type=int, default=200000)
    ap.add_argument("--pretrain-epochs", type=int, default=3)
    ap.add_argument("--head-events", type=int, default=200000)
    ap.add_argument("--head-epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    t0 = time.time()
    pool = H.load(a.pool)
    rng = np.random.default_rng(a.seed)
    fv, tt, valid = M.tokenize(pool)
    log = print

    idx = rng.permutation(len(pool["label"]))
    fv4, tt4, valid4 = fv[idx], tt[idx], valid[idx]
    feats = M.features(H.select(pool, idx))
    lab, wgt = pool["label"][idx], pool["weight"][idx]

    (fx, ft, fv_, ly, lw, lf), (vx, vt, vv, vy, vw, vf), (tx, tt_, tv, ty, tw, tf_) = \
        split(fv4, tt4, valid4, lab, wgt, feats)

    n_ev = len(lab)
    log(f"tokenized {n_ev} events: train {len(fx)}, val {len(vx)}, test {len(tx)}")

    # ---- scaler: fit on the training split only ----------------------------
    mean, std = fit_scaler(fx, fv_)
    scale = make_scaler(fx, fv_, mean, std)

    # ---- pretrain the backbone (masked reconstruction) ----------------------
    model = MiniHEP()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    n_pretr = min(a.pretrain_events, len(fx))
    x_t = torch.from_numpy(scale(fx[:n_pretr], fv_[:n_pretr]))
    tt_t = torch.from_numpy(ft[:n_pretr]); v_t = torch.from_numpy(np.asarray(fv_[:n_pretr], dtype=bool))
    bs = a.batch
    for epoch in range(a.pretrain_epochs):
        model.train(True)
        perm = np.random.default_rng(a.seed + 100 + epoch).permutation(n_pretr)
        tot, nb = 0.0, 0
        for start in range(0, n_pretr, bs):
            j = torch.from_numpy(perm[start:start + bs])
            hid = torch.from_numpy(mask_valid(v_t[j].numpy(), np.random.default_rng(a.seed + epoch)))
            out = model(x_t[j], tt_t[j], v_t[j], hid)
            mask = hid.float()[..., None]
            denom = float(mask.sum() * M.N_FEATURES)
            if denom == 0:
                continue
            loss = ((out - x_t[j]) ** 2 * mask).sum() / denom
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * j.numel(); nb += j.numel()
        log(f"pretrain epoch {epoch + 1}: recon loss {tot / max(nb, 1):.5f} ({time.time() - t0:.0f}s elapsed)")
    model.eval()

    # ---- frozen-backbone embeddings -----------------------------------------
    emb_tr = emb_of(model, scale, fx, ft, fv_)
    emb_va = emb_of(model, scale, vx, vt, vv)
    emb_te = emb_of(model, scale, tx, tt_, tv)
    log(f"embeddings computed ({time.time() - t0:.0f}s elapsed)")
    log(f"embedding statistics: train std {emb_tr.std(axis=0).mean():.3f}, "
        f"signal/background mean |Δ| {np.abs(emb_tr[ly == 1].mean(0) - emb_tr[ly == 0].mean(0)).sum():.3f}")

    # ---- analysis head(s) on the embeddings (balanced classes) --------------
    r = np.random.default_rng(a.seed + 5)
    sig = np.flatnonzero(ly == 1); bkg = np.flatnonzero(ly == 0)
    k = a.head_events // 2
    hidx = np.concatenate([r.choice(sig, size=min(k, len(sig)), replace=False),
                           r.choice(bkg, size=min(k, len(bkg)), replace=False)])
    hidx = r.permutation(hidx)
    feats_tr = lf  # engineered features of the training split
    fmean = feats_tr.mean(axis=0); fstd = feats_tr.std(axis=0) + 1e-8

    def train_mlp(Xtr_np, Xva_np, feat_tr, feat_va, n_in, tag):
        """MLP(ReLU) signal-vs-background head; returns (head, best_val_auc, test_p)."""
        if feat_tr is not None:
            Xtr_np = np.concatenate([Xtr_np, (feat_tr - fmean) / fstd], axis=1)
            Xva_np = np.concatenate([Xva_np, (feat_va - fmean) / fstd], axis=1)
        torch.manual_seed(a.seed + 1)
        head = torch.nn.Sequential(torch.nn.Linear(n_in, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
        opt2 = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-4)
        lossf = torch.nn.BCEWithLogitsLoss()
        Xt = torch.from_numpy(Xtr_np.astype(np.float32))
        yt = torch.from_numpy(ly[hidx].astype(np.float32))
        Xv = torch.from_numpy(Xva_np.astype(np.float32))
        yv = torch.from_numpy(vy.astype(np.float32))
        best_auc, best_state = -1.0, None
        for epoch in range(a.head_epochs):
            head.train(True)
            perm = np.random.default_rng(a.seed + 200 + epoch).permutation(len(Xt))
            for start in range(0, len(Xt), bs):
                j = torch.from_numpy(perm[start:start + bs])
                loss = lossf(head(Xt[j]).squeeze(-1), yt[j])
                opt2.zero_grad(); loss.backward(); opt2.step()
            if (epoch + 1) % 5 == 0 or epoch == a.head_epochs - 1:
                head.eval()
                with torch.no_grad():
                    p = torch.sigmoid(head(Xv).squeeze(-1)).numpy()
                auc = roc_auc_score(vy, p)
                auc_w = roc_auc_score(vy, p, sample_weight=vw) if len(np.unique(vy)) > 1 else float("nan")
                log(f"head[{tag}] epoch {epoch + 1}: val AUC {auc:.3f} (weighted {auc_w:.3f})")
                if auc > best_auc:
                    best_auc, best_state = auc, {k2: v2.clone() for k2, v2 in head.state_dict().items()}
        head.load_state_dict(best_state)
        head.eval()
        return head, best_auc

    log(f"head training on {len(hidx)} balanced events")
    head_emb, auc_emb = train_mlp(emb_tr[hidx], emb_va, None, None, D_MODEL, "emb")
    head_both, auc_both = train_mlp(emb_tr[hidx], emb_va, feats_tr[hidx], vf, D_MODEL + len(M.FEATURE_NAMES), "emb+feat")
    use_feat = auc_both > auc_emb
    log(f"val AUC: embedding-only {auc_emb:.3f}; embedding+features {auc_both:.3f} -> "
        f"{'embedding+features' if use_feat else 'embedding-only'}")
    chosen = head_both if use_feat else head_emb
    with torch.no_grad():
        p_te = torch.sigmoid(chosen(torch.from_numpy(
            (np.concatenate([emb_te, (tf_ - fmean) / fstd], axis=1) if use_feat
             else emb_te).astype(np.float32))).squeeze(-1)).numpy()
    log(f"test AUC: {roc_auc_score(ty, p_te):.3f} "
        f"(weighted {roc_auc_score(ty, p_te, sample_weight=tw):.3f})")

    # ---- export --------------------------------------------------------------
    def np_param(t):
        return t.detach().cpu().numpy().tolist()

    enc_layers = [{
        "in_proj_weight": np_param(layer.self_attn.in_proj_weight),
        "in_proj_bias": np_param(layer.self_attn.in_proj_bias),
        "out_proj": {"weight": np_param(layer.self_attn.out_proj.weight),
                     "bias": np_param(layer.self_attn.out_proj.bias)},
        "linear1": {"weight": np_param(layer.linear1.weight), "bias": np_param(layer.linear1.bias)},
        "linear2": {"weight": np_param(layer.linear2.weight), "bias": np_param(layer.linear2.bias)},
        "norm1": {"weight": np_param(layer.norm1.weight), "bias": np_param(layer.norm1.bias)},
        "norm2": {"weight": np_param(layer.norm2.weight), "bias": np_param(layer.norm2.bias)},
    } for layer in model.encoder.layers]
    head_json = {
        "layers": [{"W": np_param(chosen[0].weight), "b": np_param(chosen[0].bias)},
                   {"W": np_param(chosen[2].weight), "b": np_param(chosen[2].bias)}],
        "activation": "relu",
    }
    if use_feat:
        head_json.update({"features": M.FEATURE_NAMES, "feature_mean": fmean.tolist(),
                          "feature_std": fstd.tolist()})
    out = {
        "scaler": {"mean": mean.tolist(), "std": std.tolist()},
        "model": {
            "input_proj": {"weight": np_param(model.input_proj.weight), "bias": np_param(model.input_proj.bias)},
            "type_embed": np_param(model.type_embed.weight),
            "mask_embed": np_param(model.mask_embed),
            "encoder_layers": enc_layers,
        },
        "head": head_json,
        "note": f"MiniHEP backbone ({len(enc_layers)} layers, d_model=32) + MLP head "
                f"({'embedding+engineered features' if use_feat else 'embedding only'}); "
                f"pretrain {n_pretr} events x {a.pretrain_epochs} epochs; head {len(hidx)} balanced events; "
                f"test AUC {roc_auc_score(ty, p_te):.3f}",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))
    log(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e3:.0f} kB)")

    # ---- numpy forward vs torch forward --------------------------------------
    nm = M.load_model(a.out)
    fv16, tt16, v16 = fv4[-64:], tt4[-64:], valid4[-64:]
    emb_np = nm.encode_events(fv16, tt16, v16)
    x16 = torch.from_numpy(scale(fv16, v16))
    with torch.no_grad():
        emb_th = model.encode_events(x16, torch.from_numpy(tt16), torch.from_numpy(np.asarray(v16, dtype=bool))).numpy()
    diff = float(np.abs(emb_np - emb_th).max())
    log(f"numpy vs torch embedding max|diff| = {diff:.3e} ({'OK' if diff < 1e-3 else 'MISMATCH'})")
    assert diff < 1e-3, "numpy forward does not match torch"


if __name__ == "__main__":
    main()

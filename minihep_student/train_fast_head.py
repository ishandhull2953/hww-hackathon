#!/usr/bin/env python3
"""Train the fast features-only analysis head (same MLP anatomy as the MiniHEP
head) and merge it into submission/model.json as ``head_feat``.

    python3 minihep_student/train_fast_head.py [--pool ...] [--model submission/model.json]
        [--head-events 200000] [--head-epochs 60] [--seed 7]

The 10 engineered features of the dataset are standardised and fed to an MLP
(10 -> 32 -> 16 -> 1, ReLU).  Because every feature scales exactly linearly
with the mass-morphing factor (m / 125), the score of a signal event of mass m
is an exact replug of the features scaled by k -- so the 3D template fit
(score, m_T, m_ll) can re-score each template mass at full statistics for a
few seconds, which the transformer-based head cannot do cheaply.  The
transformer head is kept in the same file for the (slower) alternative modes.

Dev-only: needs torch + sklearn; the exported head runs with numpy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "submission"))

import hww as H          # noqa: E402
import minihep as M      # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=str(HERE.parent / "datasets/public/seed_000/mc/pool.npz"))
    ap.add_argument("--model", default=str(HERE.parent / "submission/model.json"))
    ap.add_argument("--head-events", type=int, default=40000)
    ap.add_argument("--head-epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    t0 = time.time()
    pool = H.load(a.pool)
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(pool["label"]))
    feats = M.features(H.select(pool, idx)).astype(np.float32)
    lab = pool["label"][idx].astype(np.float32)
    wgt = pool["weight"][idx]

    n = len(lab)
    n_tr = int(round(0.7 * n)); n_va = int(round(0.15 * n))
    x_tr, y_tr, w_tr = feats[:n_tr], lab[:n_tr], wgt[:n_tr]
    x_va, y_va, w_va = feats[n_tr:n_tr + n_va], lab[n_tr:n_tr + n_va], wgt[n_tr:n_tr + n_va]
    x_te, y_te, w_te = feats[n_tr + n_va:], lab[n_tr + n_va:], wgt[n_tr + n_va:]

    fmean = x_tr.mean(0); fstd = x_tr.std(0) + 1e-8
    Xt_all = torch.from_numpy((x_tr - fmean) / fstd)
    Xv = torch.from_numpy((x_va - fmean) / fstd)
    Xe = torch.from_numpy((x_te - fmean) / fstd)

    r = np.random.default_rng(a.seed + 5)
    sig = np.flatnonzero(y_tr == 1); bkg = np.flatnonzero(y_tr == 0)
    k = a.head_events // 2
    hidx = np.concatenate([r.choice(sig, size=min(k, len(sig)), replace=False),
                           r.choice(bkg, size=min(k, len(bkg)), replace=False)])
    r.shuffle(hidx)
    Xt, yt = Xt_all[hidx], torch.from_numpy(y_tr[hidx])

    torch.manual_seed(1)   # same chain as the validated fast head (seed 7 -> rng 12 -> torch 1)
    head = torch.nn.Sequential(torch.nn.Linear(len(M.FEATURE_NAMES), 32), torch.nn.ReLU(),
                               torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    if DEV == "cuda":
        head = head.cuda(); Xt = Xt.cuda(); yt = yt.cuda(); Xv = Xv.cuda(); Xe = Xe.cuda()

    best_auc, best = -1.0, None
    bs = a.batch
    for epoch in range(a.head_epochs):
        head.train()
        perm = torch.from_numpy(np.random.default_rng(200 + epoch).permutation(len(Xt)))
        for s in range(0, len(Xt), bs):
            j = perm[s:s + bs]
            opt.zero_grad()
            loss = lossf(head(Xt[j]).squeeze(-1), yt[j])
            loss.backward(); opt.step()
        if (epoch + 1) % 5 == 0 or epoch == a.head_epochs - 1:
            head.eval()
            with torch.no_grad():
                p = torch.sigmoid(head(Xv).squeeze(-1)).cpu().numpy()
            auc = roc_auc_score(y_va, p)
            if auc > best_auc:
                best_auc, best = auc, {k2: v2.detach().cpu().clone() for k2, v2 in head.state_dict().items()}
    head.load_state_dict(best); head.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(head(Xe).squeeze(-1)).cpu().numpy()
    auc_te = roc_auc_score(y_te, p_te)
    auc_w = roc_auc_score(y_te, p_te, sample_weight=w_te)
    print(f"fast head: val AUC {best_auc:.3f}, test AUC {auc_te:.3f} (weighted {auc_w:.3f}) "
          f"({time.time() - t0:.0f}s)", flush=True)

    def np_param(t):
        return t.detach().cpu().numpy().tolist()

    head_feat = {
        "layers": [{"W": np_param(head[0].weight), "b": np_param(head[0].bias)},
                   {"W": np_param(head[2].weight), "b": np_param(head[2].bias)},
                   {"W": np_param(head[4].weight), "b": np_param(head[4].bias)}],
        "activation": "relu", "features": M.FEATURE_NAMES,
        "feature_mean": fmean.tolist(), "feature_std": fstd.tolist(),
        "note": f"features-only MLP, {len(hidx)} balanced events x {a.head_epochs} epochs, "
                f"test AUC {auc_te:.3f}",
    }
    path = Path(a.model)
    d = json.loads(path.read_text())
    d["head_feat"] = head_feat
    if "note" in d:
        d["note"] += f"; fast features-only head {auc_te:.3f}"
    path.write_text(json.dumps(d))
    print(f"merged head_feat into {path} ({path.stat().st_size / 1e3:.0f} kB)")

    # sanity: numpy scorer matches torch
    head = head.cpu().eval()
    fh = M.FeaturesHead(head_feat)
    import hww as H2
    ev = H2.select(H2.load(a.pool), np.arange(512))
    p_np = fh.score(M.features(ev))
    with torch.no_grad():
        p_th = torch.sigmoid(head(torch.from_numpy(((M.features(ev) - fmean) / fstd).astype(np.float32))).squeeze(-1)).numpy()
    diff = float(np.abs(p_np - p_th).max())
    print(f"numpy vs torch score max|diff| = {diff:.3e} ({'OK' if diff < 1e-5 else 'MISMATCH'})")
    assert diff < 1e-5


if __name__ == "__main__":
    main()

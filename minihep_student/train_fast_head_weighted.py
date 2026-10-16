#!/usr/bin/env python3
"""Event-weighted analysis-head variants, judged by the physics fit (mu, mH).

    python3 minihep_student/train_fast_head_weighted.py --mode {v0,v1,v2,v3,v4} --out model_vN.json

  v0 balanced_unw : 50/50 balanced subsample, unweighted BCE          (old head)
  v1 balanced_w   : same balanced subsample, event-weighted BCE
  v2 sample_w     : events drawn with probability proportional to mc weight
                    (the data mixture), unweighted BCE
  v3 full_w       : every training event, event-weighted BCE (weighted NLL)
  v4 sample_w_mk  : v2's mixture, but the head is parametrised in the mass
                    hypothesis: signal events are morphed to a random mass in
                    [100, 150] GeV and k = m/125 is an extra network input.
                    At fit time a signal of mass m is scored with the exactly
                    scaled features and its own k (a "ruler" for the mass), so
                    the (score, m_T, m_ll) fit recovers m_H and mu together.

Dev-only: needs torch + sklearn; the exported head runs with numpy.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

KIT = Path("/data/pilots/hsf-india-hww")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "submission"))

import hww as H          # noqa: E402
import minihep as M      # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
K_RANGE = (100.0 / 125.0, 150.0 / 125.0)


def _standardize(x, fmean, fstd):
    return (x - fmean) / fstd


def train_one(mode: str, pool, seed: int, head_events: int, head_epochs: int, batch: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pool["label"]))
    feats = M.features(H.select(pool, idx)).astype(np.float32)
    lab = pool["label"][idx].astype(np.float32)
    wgt = pool["weight"][idx].astype(np.float32)

    n = len(lab)
    n_tr = int(round(0.7 * n)); n_va = int(round(0.15 * n))
    x_tr, y_tr, w_tr = (arr[:n_tr] for arr in (feats, lab, wgt))
    x_va, y_va, w_va = (arr[n_tr:n_tr + n_va] for arr in (feats, lab, wgt))
    x_te, y_te, w_te = (arr[n_tr + n_va:] for arr in (feats, lab, wgt))

    if mode == "v4":
        # mass-parametrised, sampled from the weighted data mixture
        p = wgt[:n_tr] / wgt[:n_tr].sum()
        draws = np.random.default_rng(seed + 11).choice(n_tr, size=head_events, p=p)
        evd = H.select(pool, idx[draws])
        masses = np.random.default_rng(seed + 23).uniform(100.0, 150.0, len(draws))
        is_sig = evd["label"] == 1
        X = np.empty((len(draws), 11), np.float32)
        # Background gets a random k *nput too (features NOT morphed), so the
        # network cannot use "k != 1 => signal" as a shortcut; k is a nuisance.
        kb = np.random.default_rng(seed + 31).uniform(K_RANGE[0], K_RANGE[1], int((~is_sig).sum()))
        X[~is_sig] = M.features_mk(H.select(evd, ~is_sig), kb)
        ev_sig = H.select(evd, is_sig)
        m_sig = masses[is_sig]
        for j in range(len(m_sig)):
            ej = H.select(ev_sig, np.array([j]))
            X[np.flatnonzero(is_sig)[j]] = M.features_mk(H.morph(ej, float(m_sig[j])), float(m_sig[j]) / 125.0)[0]
        fmean = X.mean(0); fstd = X.std(0) + 1e-8
        Xt = torch.from_numpy(_standardize(X, fmean, fstd))
        yt = torch.from_numpy(lab[draws])
        wt = torch.from_numpy(wgt[draws])
        n_in = 11
        Xv = torch.from_numpy(_standardize(M.features_mk(H.select(pool, idx[n_tr:n_tr + n_va]), 1.0), fmean, fstd).astype(np.float32))
        Xe = torch.from_numpy(_standardize(M.features_mk(H.select(pool, idx[n_tr + n_va:]), 1.0), fmean, fstd).astype(np.float32))
        mode_loss = "unweighted"
    else:
        fmean = x_tr.mean(0); fstd = x_tr.std(0) + 1e-8
        Xv = torch.from_numpy(_standardize(x_va, fmean, fstd))
        Xe = torch.from_numpy(_standardize(x_te, fmean, fstd))
        n_in = len(M.FEATURE_NAMES)
        if mode in ("v0", "v1"):
            r = np.random.default_rng(seed + 5)
            sig = np.flatnonzero(y_tr == 1); bkg = np.flatnonzero(y_tr == 0)
            k = head_events // 2
            hidx = np.concatenate([r.choice(sig, size=min(k, len(sig)), replace=False),
                                   r.choice(bkg, size=min(k, len(bkg)), replace=False)])
            r.shuffle(hidx)
            Xt = torch.from_numpy(_standardize(x_tr[hidx], fmean, fstd))
            yt = torch.from_numpy(y_tr[hidx]); wt = torch.from_numpy(w_tr[hidx])
            mode_loss = "weighted" if mode == "v1" else "unweighted"
        elif mode == "v2":
            p = wgt[:n_tr] / wgt[:n_tr].sum()
            draws = np.random.default_rng(seed + 11).choice(n_tr, size=head_events, p=p)
            Xt = torch.from_numpy(_standardize(x_tr[draws], fmean, fstd))
            yt = torch.from_numpy(y_tr[draws]); wt = torch.from_numpy(w_tr[draws])
            mode_loss = "unweighted"
        else:  # v3
            Xt = torch.from_numpy(_standardize(x_tr, fmean, fstd))
            yt = torch.from_numpy(y_tr); wt = torch.from_numpy(w_tr)
            mode_loss = "weighted"

    torch.manual_seed(1)
    head = torch.nn.Sequential(torch.nn.Linear(n_in, 32), torch.nn.ReLU(),
                               torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")
    if DEV == "cuda":
        head = head.cuda(); Xt, yt, wt = Xt.cuda(), yt.cuda(), wt.cuda(); Xv, Xe = Xv.cuda(), Xe.cuda()

    best_auc, best = -1.0, None
    bs = batch
    for epoch in range(head_epochs):
        head.train()
        perm = torch.from_numpy(np.random.default_rng(200 + epoch).permutation(len(Xt)))
        for s in range(0, len(Xt), bs):
            j = perm[s:s + bs]
            opt.zero_grad()
            l = lossf(head(Xt[j]).squeeze(-1), yt[j])
            w = wt[j]
            loss = (l * (w / w.mean())).mean() if mode_loss == "weighted" else l.mean()
            loss.backward(); opt.step()
        if (epoch + 1) % 5 == 0 or epoch == head_epochs - 1:
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
    return head, fmean, fstd, (best_auc, auc_te, auc_w)


def export(model_path: Path, out_path: Path, head, fmean, fstd, features: list, note: str, auc_te: float):
    def np_param(t):
        return t.detach().cpu().numpy().tolist()

    head_feat = {
        "layers": [{"W": np_param(head[0].weight), "b": np_param(head[0].bias)},
                   {"W": np_param(head[2].weight), "b": np_param(head[2].bias)},
                   {"W": np_param(head[4].weight), "b": np_param(head[4].bias)}],
        "activation": "relu", "features": features,
        "feature_mean": fmean.tolist(), "feature_std": fstd.tolist(),
        "note": note + f", test AUC {auc_te:.3f}",
    }
    d = copy.deepcopy(json.loads(Path(model_path).read_text()))
    d["head_feat"] = head_feat
    d["note"] = note
    out_path.write_text(json.dumps(d))
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="v4", choices=["v0", "v1", "v2", "v3", "v4"])
    ap.add_argument("--pool", default=str(KIT / "datasets/public/seed_000/mc/pool.npz"))
    ap.add_argument("--model", default=str(KIT / "submission/model.json"))
    ap.add_argument("--out", default="/tmp/opencode/models/model_v4.json")
    ap.add_argument("--head-events", type=int, default=60000)
    ap.add_argument("--head-epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    t0 = time.time()
    pool = H.load(a.pool)
    head, fmean, fstd, (auc_va, auc_te, auc_w) = train_one(
        a.mode, pool, a.seed, a.head_events, a.head_epochs, a.batch)
    notes = {"v0": "balanced 50/50 sample, unweighted BCE (old head)",
             "v1": "balanced 50/50 sample, event-weighted BCE",
             "v2": "events sampled with p~mcWeight (data mixture), unweighted BCE",
             "v3": "full training sample, event-weighted BCE (weighted NLL)",
             "v4": "weighted mixture, mass-parametrised (features + k=m/125 input), "
                   "signal morphed across [100,150] GeV"}
    feats_list = M.FEATURE_NAMES + ["k"] if a.mode == "v4" else M.FEATURE_NAMES
    print(f"mode {a.mode}: val AUC {auc_va:.3f}, test AUC {auc_te:.3f} (weighted {auc_w:.3f}) "
          f"({time.time() - t0:.0f}s)", flush=True)
    out = export(Path(a.model), Path(a.out), head, fmean, fstd, feats_list, notes[a.mode], auc_te)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

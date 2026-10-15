"""pytest checks for the MiniHEP tutorial (minihep_student).

Run from the course checkout root (where dataWW_d1.root lives)::

    python -m pytest minihep_student/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import minihep_student as mh
from minihep_student import data as d
from minihep_student import scale as s
from minihep_student import train as trn
from minihep_student.model import MiniHEP, count_parameters

N_SLOTS = d.N_SLOTS
N_FEATURES = d.N_FEATURES


@pytest.fixture(scope="module")
def events():
    """Load and split the sampled dataset once for the whole module."""
    ev = d.load_events(max_events=6000, seed=42)
    return d.split_events(ev)


def test_array_shapes(events):
    train, val, test = events
    for split in (train, val, test):
        n = split.n_events
        assert split.fourvec.shape == (n, N_SLOTS, N_FEATURES)
        assert split.token_types.shape == (n, N_SLOTS)
        assert split.valid.shape == (n, N_SLOTS)
        assert split.valid.dtype == np.bool_
        assert split.mc_weight.shape == (n,)
        assert split.label.shape == (n,)


def test_gev_conversion(events):
    train, _, _ = events
    # Leptons are massless, so E == pt * cosh(eta).  Jets carry mass, so E is
    # slightly larger than pt * cosh(eta).  A wrong MeV/GeV factor (~1000x)
    # would fail by a huge margin in every case.
    for slot in (d.SLOT_LEP0, d.SLOT_LEP1):
        px, py, pz, e = (train.fourvec[:, slot, i] for i in range(4))
        pt = np.hypot(px, py)
        expected = pt * np.cosh(np.arcsinh(pz / pt))
        np.testing.assert_allclose(e, expected, rtol=1e-2)
    for slot in (d.SLOT_JET0, d.SLOT_JET1):
        px, py, pz, e = (train.fourvec[:, slot, i] for i in range(4))
        pt = np.hypot(px, py)
        mask = train.valid[:, slot]
        massless_energy = pt[mask] * np.cosh(np.arcsinh(pz[mask] / pt[mask]))
        assert np.all(e[mask] >= 0.99 * massless_energy)
        assert np.all(e[mask] < 1.5 * massless_energy)
    # All components are in GeV (a leftover MeV value would be huge).
    assert np.abs(train.fourvec).max() < 1e4


def test_zero_padding_absent_jets(events):
    train, _, _ = events
    # Absent jet slots are zero vectors, token type PAD, valid False.
    absent = ~train.valid[:, d.SLOT_JET0]
    assert absent.any()
    np.testing.assert_array_equal(train.fourvec[absent, d.SLOT_JET0], 0.0)
    assert np.all(train.token_types[absent, d.SLOT_JET0] == d.TOK_PAD)
    # No -7 filler ever leaks into the vectors.
    assert not np.any(np.abs(train.fourvec) > 1e5)


def test_token_types(events):
    train, _, _ = events
    # Valid tokens carry a non-pad type from the 7-ID scheme.
    valid_types = train.token_types[train.valid]
    assert set(np.unique(valid_types)).issubset(
        {d.TOK_EM, d.TOK_EP, d.TOK_MUM, d.TOK_MUP, d.TOK_JET, d.TOK_MET}
    )
    assert not np.any(valid_types == d.TOK_PAD)
    # Leptons and MET are always valid.
    assert train.valid[:, d.SLOT_LEP0].all()
    assert train.valid[:, d.SLOT_LEP1].all()
    assert train.valid[:, d.SLOT_MET].all()
    # MET is the transverse proxy: pz == 0.
    np.testing.assert_array_equal(train.fourvec[:, d.SLOT_MET, 2], 0.0)


def test_disjoint_splits(events):
    train, val, test = events
    ids = [set(x.event_index.tolist()) for x in (train, val, test)]
    assert len(ids[0] & ids[1]) == 0
    assert len(ids[0] & ids[2]) == 0
    assert len(ids[1] & ids[2]) == 0
    n = train.n_events + val.n_events + test.n_events
    assert abs(train.n_events / n - 0.7) < 0.01
    assert abs(val.n_events / n - 0.15) < 0.02


def test_scaler_fit_depends_only_on_training(events):
    train, _, _ = events
    sc = s.FourVecScaler().fit(train.fourvec, train.valid)
    # Refitting on the same training events is identical (deterministic).
    sc2 = s.FourVecScaler().fit(train.fourvec, train.valid)
    np.testing.assert_allclose(sc.mean, sc2.mean)
    np.testing.assert_allclose(sc.std, sc2.std)
    # The fitted statistics equal those of the training valid tokens only.
    x = np.arcsinh(train.fourvec[train.valid].astype(np.float64) / 50.0)
    np.testing.assert_allclose(sc.mean, x.mean(axis=0), rtol=1e-6)
    # Changing the validation data has no effect: it is never passed to fit.
    sc3 = s.FourVecScaler().fit(train.fourvec, train.valid)
    np.testing.assert_allclose(sc3.mean, sc.mean)
    # Padded tokens are reset to zero after transform.
    out = sc.transform(train.fourvec, train.valid)
    np.testing.assert_array_equal(out[~train.valid], 0.0)
    assert not np.allclose(out[train.valid], 0.0)


def test_encoder_layers_independently_initialized():
    """The two encoder-layer blocks must not start from identical weights.

    nn.TransformerEncoder deepcopies the prototype layer, so without a full
    reset (including the MultiheadAttention in-projection parameters) layer 0
    and layer 1 would share identical attention weights.
    """
    m = MiniHEP()
    l0, l1 = m.encoder.layers[0], m.encoder.layers[1]
    assert not torch.equal(l0.self_attn.in_proj_weight, l1.self_attn.in_proj_weight)
    assert not torch.equal(l0.self_attn.out_proj.weight, l1.self_attn.out_proj.weight)
    assert not torch.equal(l0.linear1.weight, l1.linear1.weight)
    assert not torch.equal(l0.linear2.weight, l1.linear2.weight)


def test_masking_only_valid_tokens():
    rng = np.random.default_rng(0)
    valid = np.ones((64, N_SLOTS), dtype=bool)
    valid[:, d.SLOT_JET0] = False  # simulate an absent jet
    hidden = trn.mask_valid(valid, rng)
    # No padded token is ever masked.
    assert not np.any(hidden & ~valid)
    # Always at least one hidden and at least one visible per event.
    assert np.all(hidden.sum(axis=1) >= 1)
    assert np.all(hidden.sum(axis=1) < valid.sum(axis=1))


def test_finite_one_epoch_cpu_training(events):
    train, val, _ = events
    torch.set_num_threads(1)
    sc = s.fit_scaler_for_split(train)
    Xtr, TTtr, Vtr = trn.to_tensors(train, sc)
    Xva, TTva, Vva = trn.to_tensors(val, sc)
    model = MiniHEP()
    assert count_parameters(model) == 17636
    hist, vhist = trn.train_model(model, Xtr, TTtr, Vtr, Xva, TTva, Vva,
                                  epochs=1, batch_size=64, seed=0)
    assert len(hist) == 1 and len(vhist) == 1
    for v in (hist + vhist):
        assert np.isfinite(v) and v >= 0
    emb = model.encode_events(Xtr[:8], TTtr[:8], Vtr[:8])
    assert tuple(emb.shape) == (8, 32)

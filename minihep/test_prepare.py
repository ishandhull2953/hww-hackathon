"""Tests for minihep.prepare.

Run from the repo root:  python -m unittest minihep.test_prepare -v
or:                      python minihep/test_prepare.py
"""

from __future__ import annotations

import unittest

import numpy as np

from minihep import data as data
from minihep import prepare as prep


class TestSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ev = data.load_events(max_events=6000, seed=42)

    def test_split_70_15_15_disjoint_and_complete(self):
        masks = prep.split_masks(self.ev.event_index)
        n = self.ev.n_events
        union = np.zeros(n, dtype=bool)
        sizes = {}
        for name in ("train", "val", "test"):
            m = masks[name]
            self.assertEqual(len(m), n)
            sizes[name] = int(m.sum())
            # no overlap with anything seen before
            self.assertEqual(int((union & m).sum()), 0, f"{name} overlaps")
            union |= m
        self.assertEqual(int(union.sum()), n, "splits do not cover all events")
        for name, frac in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
            self.assertAlmostEqual(sizes[name] / n, frac, delta=0.01, msg=name)

    def test_event_identifiers_do_not_overlap(self):
        masks = prep.split_masks(self.ev.event_index)
        seen: set[int] = set()
        for name in ("train", "val", "test"):
            sids = set(int(i) for i in self.ev.event_index[masks[name]])
            self.assertEqual(len(sids), int(masks[name].sum()), f"duplicate event_index in {name}")
            self.assertEqual(len(seen & sids), 0, f"event_index appears in two splits ({name})")
            seen |= sids

    def test_split_stable_across_sample_sizes(self):
        small = data.load_events(max_events=6000, seed=42)
        big = data.load_events(max_events=12000, seed=42)
        small_masks = prep.split_masks(small.event_index)
        pos = {int(i): j for j, i in enumerate(big.event_index)}
        big_masks = prep.split_masks(big.event_index)
        for name in ("train", "val", "test"):
            m_small = small_masks[name]
            for i in np.flatnonzero(m_small):
                row = int(small.event_index[i])
                self.assertIn(row, pos, "small sample event not in big sample")
                self.assertTrue(bool(big_masks[name][pos[row]]), f"{name} assignment changed")


class TestScaler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ev = data.load_events(max_events=6000, seed=42)
        cls.prep = prep.prepare(cls.ev)

    def test_scaler_matches_manual_formula(self):
        s = self.prep.scaler
        masks = prep.split_masks(self.ev.event_index)
        raw = self.ev.fourvec[masks["train"]]
        valid = self.ev.valid[masks["train"]]
        t = np.arcsinh(raw.astype(np.float64) / s.scale_gev)[valid]
        np.testing.assert_allclose(s.mean, t.mean(axis=0), rtol=1e-12)
        np.testing.assert_allclose(s.std, t.std(axis=0), rtol=1e-12)

    def test_transform_uses_train_stats_and_zeroes_padding(self):
        s = self.prep.scaler
        ev = self.ev
        masks = prep.split_masks(ev.event_index)
        raw_val = ev.fourvec[masks["val"]]
        valid_val = ev.valid[masks["val"]]
        expected = (np.arcsinh(raw_val.astype(np.float64) / s.scale_gev) - s.mean) / s.std
        scaled_val = self.prep.val.fourvec
        np.testing.assert_allclose(scaled_val[valid_val], expected[valid_val], rtol=1e-4, atol=1e-4)
        self.assertTrue(np.all(scaled_val[~valid_val] == 0), "padded tokens must be exactly zero")
        self.assertEqual(scaled_val.dtype, np.float32)

    def test_validation_change_does_not_affect_fitted_scaler(self):
        """Fitted stats must depend only on training events, not on whatever
        non-training (validation/test) data is present alongside them."""
        ev = self.ev
        masks = prep.split_masks(ev.event_index)
        train_idx = np.flatnonzero(masks["train"])
        val_idx = np.flatnonzero(masks["val"])
        test_idx = np.flatnonzero(masks["test"])

        # two batches with identical training rows but DIFFERENT extra data
        arr_a = ev.fourvec[np.concatenate([train_idx, val_idx])]
        valid_a = ev.valid[np.concatenate([train_idx, val_idx])]
        arr_b = ev.fourvec[np.concatenate([train_idx, test_idx])]
        valid_b = ev.valid[np.concatenate([train_idx, test_idx])]
        mask_a = np.zeros(len(arr_a), dtype=bool)
        mask_a[: len(train_idx)] = True  # training events only
        mask_b = np.zeros(len(arr_b), dtype=bool)
        mask_b[: len(train_idx)] = True

        sa = prep.FeatureScaler().fit(arr_a, valid_a, event_mask=mask_a)
        sb = prep.FeatureScaler().fit(arr_b, valid_b, event_mask=mask_b)
        np.testing.assert_array_equal(sa.mean, sb.mean)
        np.testing.assert_array_equal(sa.std, sb.std)

        # sanity: the two "validation" sets are genuinely different, so a
        # scaler that used them would differ (the test above is meaningful)
        sa_all = prep.FeatureScaler().fit(arr_a, valid_a)
        sb_all = prep.FeatureScaler().fit(arr_b, valid_b)
        self.assertFalse(np.allclose(sa_all.mean, sb_all.mean))


if __name__ == "__main__":
    unittest.main(verbosity=2)

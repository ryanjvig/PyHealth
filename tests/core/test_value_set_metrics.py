"""Tests for the value-set metrics in pyhealth.metrics.value_set.

Author: Ryan Vig (NetID: rvig2)
"""

import unittest

import numpy as np
import pandas as pd

from pyhealth.metrics import (
    best_f1_threshold,
    value_set_bootstrap_ci,
    value_set_paired_bootstrap_diff,
    value_set_scores,
    value_set_summary,
)


def scores_for(rows: list[tuple], threshold: float = 0.5) -> pd.DataFrame:
    """rows: (set_id, code, y_true, y_prob, n_true_codes)."""
    set_ids, codes, y_true, y_prob, n_true = zip(*rows)
    return value_set_scores(set_ids, codes, y_true, y_prob, n_true, threshold)


class TestValueSetScores(unittest.TestCase):
    def test_recall_counts_codes_missed_by_retrieval(self):
        # 2 of 5 true codes retrieved, both predicted.
        scores = scores_for([("a", "1", 1, 0.9, 5), ("a", "2", 1, 0.8, 5)])
        row = scores.iloc[0]
        self.assertEqual(row["precision"], 1.0)
        self.assertAlmostEqual(row["recall"], 0.4)
        self.assertAlmostEqual(row["retrieval_recall"], 0.4)
        self.assertAlmostEqual(row["f1"], 2 * 0.4 / 1.4)

    def test_no_prediction_scores_zero(self):
        scores = scores_for([("a", "1", 1, 0.1, 1), ("a", "2", 0, 0.2, 1)])
        row = scores.iloc[0]
        self.assertEqual((row["n_predicted"], row["precision"], row["f1"]), (0, 0, 0))
        self.assertEqual(row["retrieval_recall"], 1.0)

    def test_code_under_two_systems_counts_once(self):
        # Labels match on code alone, so code "1" appears twice as a positive.
        scores = scores_for(
            [("a", "1", 1, 0.9, 2), ("a", "1", 1, 0.7, 2), ("a", "3", 0, 0.6, 2)]
        )
        row = scores.iloc[0]
        counts = (row["n_candidates"], row["n_predicted"], row["n_hits"])
        self.assertEqual(counts, (2, 2, 1))
        self.assertEqual(row["precision"], 0.5)
        self.assertEqual(row["recall"], 0.5)

    def test_threshold_is_inclusive(self):
        scores = scores_for([("a", "1", 1, 0.5, 1)], threshold=0.5)
        self.assertEqual(scores.iloc[0]["recall"], 1.0)

    def test_one_row_per_set_in_first_seen_order(self):
        scores = scores_for(
            [("b", "1", 1, 0.9, 1), ("a", "1", 0, 0.9, 3), ("b", "2", 0, 0.1, 1)]
        )
        self.assertEqual(scores["set_id"].tolist(), ["b", "a"])
        self.assertEqual(scores["n_candidates"].tolist(), [2, 1])

    def test_matches_brute_force(self):
        rng = np.random.default_rng(0)
        rows = []
        for s in range(20):
            n_true = int(rng.integers(1, 10))
            true_codes = set(range(n_true))
            for c in rng.choice(20, size=int(rng.integers(1, 15)), replace=False):
                label = int(c in true_codes)
                rows.append((f"s{s}", str(c), label, rng.random(), n_true))
        scores = scores_for(rows, threshold=0.6).set_index("set_id")
        for s in {r[0] for r in rows}:
            mine = [r for r in rows if r[0] == s]
            pred = {r[1] for r in mine if r[3] >= 0.6}
            hits = {r[1] for r in mine if r[3] >= 0.6 and r[2]}
            p = len(hits) / len(pred) if pred else 0.0
            r = len(hits) / mine[0][4]
            self.assertAlmostEqual(scores.loc[s, "precision"], p)
            self.assertAlmostEqual(scores.loc[s, "recall"], r)
            self.assertAlmostEqual(
                scores.loc[s, "f1"], 2 * p * r / (p + r) if p + r else 0.0
            )

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "differ in length"):
            value_set_scores(["a"], ["1", "2"], [1], [0.5], [1], 0.5)
        with self.assertRaisesRegex(ValueError, "constant"):
            scores_for([("a", "1", 1, 0.5, 1), ("a", "2", 1, 0.5, 2)])
        with self.assertRaisesRegex(ValueError, "more true candidates"):
            scores_for([("a", "1", 1, 0.5, 1), ("a", "2", 1, 0.5, 1)])


class TestSummaryAndBootstrap(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        n = 40
        self.scores = pd.DataFrame(
            {
                "set_id": [f"s{i}" for i in range(n)],
                "precision": rng.random(n),
                "recall": rng.random(n),
                "retrieval_recall": np.where(rng.random(n) < 0.3, 1.0, 0.5),
            }
        )
        p, r = self.scores["precision"], self.scores["recall"]
        self.scores["f1"] = 2 * p * r / (p + r)

    def test_summary_distinguishes_mean_f1_and_f1_of_macro(self):
        summary = value_set_summary(self.scores)
        p, r = self.scores["precision"].mean(), self.scores["recall"].mean()
        self.assertAlmostEqual(summary["mean_f1"], self.scores["f1"].mean())
        self.assertAlmostEqual(summary["f1_of_macro"], 2 * p * r / (p + r))
        self.assertGreater(summary["f1_of_macro"], summary["mean_f1"])
        self.assertEqual(summary["n_value_sets"], 40)

    def test_summary_of_empty_subset(self):
        summary = value_set_summary(self.scores.iloc[:0])
        self.assertEqual(summary["n_value_sets"], 0)
        self.assertTrue(np.isnan(summary["mean_f1"]))

    def test_bootstrap_ci_is_deterministic_and_brackets_estimate(self):
        ci = value_set_bootstrap_ci(self.scores, n_boot=300, seed=3)
        self.assertEqual(ci, value_set_bootstrap_ci(self.scores, n_boot=300, seed=3))
        for metric, interval in ci.items():
            self.assertLessEqual(interval["lower"], interval["estimate"], metric)
            self.assertGreaterEqual(interval["upper"], interval["estimate"], metric)
            self.assertLess(interval["lower"], interval["upper"], metric)
        self.assertAlmostEqual(
            ci["mean_f1"]["estimate"], value_set_summary(self.scores)["mean_f1"]
        )

    def test_bootstrap_ci_width_shrinks_with_more_sets(self):
        big = pd.concat([self.scores] * 10, ignore_index=True)
        small_ci = value_set_bootstrap_ci(self.scores, ["mean_f1"], n_boot=300)
        big_ci = value_set_bootstrap_ci(big, ["mean_f1"], n_boot=300)
        width = lambda c: c["mean_f1"]["upper"] - c["mean_f1"]["lower"]
        self.assertLess(width(big_ci), width(small_ci))

    def test_paired_diff(self):
        better = self.scores.assign(f1=self.scores["f1"] + 0.1)
        # Row order must not matter: pairs are matched on set_id.
        shuffled = better.sample(frac=1.0, random_state=0)
        diff = value_set_paired_bootstrap_diff(shuffled, self.scores, n_boot=200)
        self.assertAlmostEqual(diff["mean_f1"]["estimate"], 0.1)
        # Constant per-set difference: every resample gives exactly 0.1.
        self.assertAlmostEqual(diff["mean_f1"]["lower"], 0.1)
        self.assertAlmostEqual(diff["macro_precision"]["estimate"], 0.0)

    def test_paired_diff_needs_same_sets(self):
        with self.assertRaisesRegex(ValueError, "same value sets"):
            value_set_paired_bootstrap_diff(self.scores, self.scores.iloc[1:])

    def test_bootstrap_validation(self):
        with self.assertRaisesRegex(ValueError, "Unknown metrics"):
            value_set_bootstrap_ci(self.scores, ["auroc"])
        with self.assertRaisesRegex(ValueError, "n_boot"):
            value_set_bootstrap_ci(self.scores, n_boot=0)
        with self.assertRaisesRegex(ValueError, "at least one"):
            value_set_bootstrap_ci(self.scores.iloc[:0])


class TestBestF1Threshold(unittest.TestCase):
    def test_picks_f1_maximizing_cutoff(self):
        y_true = np.array([0, 0, 1, 1, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.9, 0.7])
        threshold = best_f1_threshold(y_true, y_prob)
        f1 = lambda t: (
            2 * ((y_prob >= t) & (y_true == 1)).sum()
            / ((y_prob >= t).sum() + y_true.sum())
        )
        self.assertEqual(f1(threshold), max(f1(t) for t in y_prob))


if __name__ == "__main__":
    unittest.main()

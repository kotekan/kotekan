"""Small CPU-only contracts for the native replay harness."""

import copy
import unittest

import numpy as np

from replay_chime_native import (
    FRAME_SAMPLES as N,
    aligned_offsets,
    base_config,
    numerical_fixture,
    paired_selection,
    proxy_excess,
    validate_offline_graph,
    validate_weight_norms,
)


class ReplayContractTest(unittest.TestCase):
    def test_alignment_retains_absolute_phase(self):
        self.assertEqual(
            aligned_offsets(N + 3, 4 * N, 8), [N - 3, 2 * N - 3, 3 * N - 3]
        )
        self.assertEqual(aligned_offsets(2 * N, 2 * N, 8), [0, N])
        self.assertEqual(aligned_offsets(1, N - 1, 8), [])
        with self.assertRaises(ValueError):
            aligned_offsets(-1, N, 1)

    def test_pair_phase_not_adjacent_storage_rows(self):
        complete, keep = paired_selection([N, 2 * N, 3 * N, 4 * N], [True] * 4)
        np.testing.assert_array_equal(complete, [False, True, True, False])
        np.testing.assert_array_equal(keep, complete)

    def test_pair_rejects_both_and_is_order_independent(self):
        complete, keep = paired_selection(
            [3 * N, 0, N, 2 * N], [True, True, False, True]
        )
        np.testing.assert_array_equal(complete, [True] * 4)
        np.testing.assert_array_equal(keep, [True, False, False, True])
        with self.assertRaises(ValueError):
            paired_selection([0, 0], [True, True])
        with self.assertRaises(ValueError):
            paired_selection([1], [True])

    def test_fixture_is_copy_and_retains_source_refusal(self):
        original = {
            "profiles": [
                {
                    "receiver_channel_id": 844,
                    "fine_calibration": {
                        "cfar_multiplier_q16": 12345,
                        "provenance": {
                            "threshold_selection": {
                                "qualification": "archive_review_only",
                                "selection_granularity": "detector_frame",
                            }
                        },
                    },
                }
            ]
        }
        saved = copy.deepcopy(original)
        fixture, profile = numerical_fixture(original, 844, 12345)
        self.assertEqual(original, saved)
        self.assertTrue(fixture["offline_test_only"])
        self.assertEqual(
            profile["fine_calibration"]["provenance"]["source_threshold_selection"],
            original["profiles"][0]["fine_calibration"]["provenance"][
                "threshold_selection"
            ],
        )
        with self.assertRaises(ValueError):
            numerical_fixture(original, 844, 12346)
        original["profiles"][0]["fine_calibration"]["provenance"][
            "threshold_selection"
        ]["qualification"] = "different"
        with self.assertRaises(ValueError):
            numerical_fixture(original, 844, 12345)

    def test_graph_refuses_added_consumer_or_live_source(self):
        config = base_config(2048, 844, 4, "offline.json")
        validate_offline_graph(config)
        extra = copy.deepcopy(config)
        extra["consume_mask"] = {"kotekan_stage": "DtvRfiMask"}
        with self.assertRaises(ValueError):
            validate_offline_graph(extra)
        config["read_voltage"]["kotekan_stage"] = "networkReceiver"
        with self.assertRaises(ValueError):
            validate_offline_graph(config)

    def test_exact_template_norm_and_signed_proxy(self):
        weights = bytes([0x10] * 128 + [0x20] * 128 + [0x30] * 128)
        profile = {
            "weight_bank_offset_bytes": 0,
            "weight_bank_nbytes": 384,
            "target_norm_sq": 128,
            "reference_norm_sum_sq": 13 * 128,
        }
        self.assertEqual(validate_weight_norms(profile, weights), [128, 512, 1152])
        self.assertEqual(proxy_excess([10, 52, 78], profile), 0)
        self.assertEqual(proxy_excess([5, 52, 78], profile), -0.5)
        self.assertIsNone(proxy_excess([5, 0, 0], profile))
        profile["target_norm_sq"] = 129
        with self.assertRaises(ValueError):
            validate_weight_norms(profile, weights)


if __name__ == "__main__":
    unittest.main()

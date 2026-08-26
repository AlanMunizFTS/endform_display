import unittest

from verdict_analysis import (
    apply_confidence_thresholds,
    calculate_average_error_rates,
    calculate_position_metrics,
    infer_position_result,
    optimize_confidence_thresholds,
    parse_actual_verdict_values,
)


class TestVerdictAnalysis(unittest.TestCase):
    def test_parse_actual_verdict_values_accepts_excel_column_and_normalizes(self):
        self.assertEqual(
            parse_actual_verdict_values("ok\r\nNOK\n ok\tNOK"),
            ["OK", "NOK", "OK", "NOK"],
        )

    def test_parse_actual_verdict_values_rejects_invalid_token(self):
        with self.assertRaisesRegex(ValueError, "Unsupported verdict"):
            parse_actual_verdict_values("OK\nMAYBE\nNOK")

    def test_calculate_position_metrics_uses_legacy_false_labels(self):
        rows = [
            self._row("OK", "OK", "NOK", None, "OK"),
            self._row("NOK", "NOK", "OK", "NOK", None),
            self._row("", "NOK", "NOK", "NOK", "NOK"),
        ]

        metrics = calculate_position_metrics(rows)

        self.assertEqual(
            metrics[1],
            {
                "true_ok": 1,
                "true_nok": 1,
                "false_negative": 0,
                "false_positive": 0,
                "evaluated": 2,
            },
        )
        self.assertEqual(metrics[2]["false_negative"], 1)
        self.assertEqual(metrics[2]["false_positive"], 1)
        self.assertEqual(metrics[2]["evaluated"], 2)
        self.assertEqual(metrics[3]["true_nok"], 1)
        self.assertEqual(metrics[3]["evaluated"], 1)
        self.assertEqual(metrics[4]["true_ok"], 1)
        self.assertEqual(metrics[4]["evaluated"], 1)

    def test_calculate_position_metrics_skips_not_available_inferred_values(self):
        rows = [self._row("OK", None, "OK", None, None)]

        metrics = calculate_position_metrics(rows)

        self.assertEqual(metrics[1]["evaluated"], 0)
        self.assertEqual(metrics[2]["true_ok"], 1)
        self.assertEqual(metrics[2]["evaluated"], 1)

    def test_confidence_thresholds_use_equality_and_preserve_incomplete_data(self):
        complete = self._confidence_entry(0.7000, 0.4000)
        incomplete = self._confidence_entry(0.9000, 0.9000, complete=False)

        self.assertEqual(
            infer_position_result(
                complete,
                {"side": 0.7000, "diag": 0.5000},
            ),
            "NOK",
        )
        self.assertEqual(
            infer_position_result(
                complete,
                {"side": 0.7001, "diag": 0.5000},
            ),
            "OK",
        )
        self.assertIsNone(
            infer_position_result(
                incomplete,
                {"side": 0.0, "diag": 0.0},
            )
        )

    def test_apply_confidence_thresholds_changes_only_inferred_values(self):
        rows = [
            {
                "actual_result": "NOK",
                "positions": [self._confidence_entry(0.60, 0.20)],
            }
        ]

        apply_confidence_thresholds(rows, {"side": 0.70, "diag": 0.30})

        self.assertEqual(rows[0]["actual_result"], "NOK")
        self.assertEqual(rows[0]["positions"][0]["inferred_result"], "OK")

    def test_average_error_rates_use_all_evaluated_pieces_as_denominator(self):
        rows = []
        for index in range(59):
            rows.append(
                {
                    "actual_result": "OK",
                    "positions": [
                        {
                            "position": 1 if index < 9 else 2,
                            "jsn": f"ok-{index}",
                            "inferred_result": "NOK" if index < 9 else "OK",
                        }
                    ],
                }
            )
        rows.append(
            {
                "actual_result": "NOK",
                "positions": [
                    {"position": 1, "jsn": "nok-1", "inferred_result": "OK"}
                ],
            }
        )

        summary = calculate_average_error_rates(rows, positions=2)

        self.assertEqual(summary["total_evaluated"], 60)
        self.assertEqual(summary["total_false_negative"], 9)
        self.assertEqual(summary["total_false_positive"], 1)
        self.assertEqual(summary["average_false_negative_rate"], 0.15)
        self.assertAlmostEqual(summary["average_false_positive_rate"], 1 / 60)

    def test_optimizer_finds_best_pair_and_uses_higher_threshold_tiebreak(self):
        rows = [
            {
                "actual_result": "OK",
                "positions": [self._confidence_entry(0.20, 0.20)],
            },
            {
                "actual_result": "NOK",
                "positions": [self._confidence_entry(0.80, 0.10)],
            },
            {
                "actual_result": "NOK",
                "positions": [self._confidence_entry(0.10, 0.70)],
            },
        ]

        result = optimize_confidence_thresholds(rows, positions=1)

        self.assertTrue(result["target_met"])
        self.assertEqual(result["average_false_positive_rate"], 0.0)
        self.assertEqual(result["average_false_negative_rate"], 0.0)
        self.assertEqual(result["thresholds"], {"side": 0.8, "diag": 0.7})

    def test_optimizer_finds_best_point_for_side_only(self):
        rows = [
            {
                "actual_result": "OK",
                "positions": [self._confidence_entry(0.20, 0.95)],
            },
            {
                "actual_result": "NOK",
                "positions": [self._confidence_entry(0.80, 0.10)],
            },
        ]

        result = optimize_confidence_thresholds(
            rows,
            required_angles=("side",),
            positions=1,
        )

        self.assertTrue(result["target_met"])
        self.assertEqual(result["average_false_positive_rate"], 0.0)
        self.assertEqual(result["average_false_negative_rate"], 0.0)
        self.assertEqual(result["thresholds"], {"side": 0.8})

    def test_optimizer_finds_best_point_for_diag_only(self):
        rows = [
            {
                "actual_result": "OK",
                "positions": [self._confidence_entry(0.95, 0.15)],
            },
            {
                "actual_result": "NOK",
                "positions": [self._confidence_entry(0.10, 0.70)],
            },
        ]

        result = optimize_confidence_thresholds(
            rows,
            required_angles=("diag",),
            positions=1,
        )

        self.assertTrue(result["target_met"])
        self.assertEqual(result["average_false_positive_rate"], 0.0)
        self.assertEqual(result["average_false_negative_rate"], 0.0)
        self.assertEqual(result["thresholds"], {"diag": 0.7})

    def test_optimizer_requires_both_actual_classes(self):
        rows = [
            {
                "actual_result": "OK",
                "positions": [self._confidence_entry(0.20, 0.20)],
            }
        ]

        with self.assertRaisesRegex(ValueError, "both OK and NOK"):
            optimize_confidence_thresholds(rows, positions=1)

    def test_optimizer_prioritizes_false_positives_within_fn_target(self):
        rows = [
            {
                "actual_result": "OK",
                "positions": [
                    self._confidence_entry(0.60 if index == 0 else 0.10, None)
                ],
            }
            for index in range(20)
        ]
        rows.append(
            {
                "actual_result": "NOK",
                "positions": [self._confidence_entry(0.55, None)],
            }
        )

        result = optimize_confidence_thresholds(rows, positions=1)

        self.assertTrue(result["target_met"])
        self.assertEqual(result["average_false_positive_rate"], 0.0)
        self.assertAlmostEqual(result["average_false_negative_rate"], 1 / 21)
        self.assertEqual(result["thresholds"]["side"], 0.55)

    def test_optimizer_does_not_sacrifice_fp_to_reach_fn_target(self):
        rows = [
            {
                "actual_result": "OK",
                "positions": [self._confidence_entry(0.60, None)],
            },
            {
                "actual_result": "NOK",
                "positions": [self._confidence_entry(0.55, None)],
            },
        ]

        result = optimize_confidence_thresholds(rows, positions=1)

        self.assertFalse(result["target_met"])
        self.assertEqual(result["average_false_positive_rate"], 0.0)
        self.assertEqual(result["average_false_negative_rate"], 0.5)
        self.assertEqual(result["thresholds"]["side"], 0.55)

    def test_optimizer_reports_when_false_negative_target_is_impossible(self):
        rows = [
            {
                "actual_result": "OK",
                "positions": [self._confidence_entry(1.0, None)],
            },
            {
                "actual_result": "NOK",
                "positions": [self._confidence_entry(0.5, None)],
            },
        ]

        result = optimize_confidence_thresholds(rows, positions=1)

        self.assertFalse(result["target_met"])
        self.assertGreaterEqual(result["average_false_negative_rate"], 0.10)

    @staticmethod
    def _row(actual, *inferred_values):
        return {
            "actual_result": actual,
            "positions": [
                {
                    "position": position,
                    "jsn": f"jsn-{position}" if inferred is not None else "",
                    "inferred_result": inferred,
                }
                for position, inferred in enumerate(inferred_values, start=1)
            ],
        }

    @staticmethod
    def _confidence_entry(side, diag, complete=True):
        return {
            "position": 1,
            "jsn": "jsn-confidence",
            "inferred_result": None,
            "confidence_data_complete": complete,
            "max_confidence_by_angle": {"side": side, "diag": diag},
        }


if __name__ == "__main__":
    unittest.main(verbosity=2)

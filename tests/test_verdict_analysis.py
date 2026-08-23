import unittest

from verdict_analysis import (
    calculate_position_metrics,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)

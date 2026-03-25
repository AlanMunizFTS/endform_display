import datetime
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from openpyxl import load_workbook

from report_exporter import export_stats_report


class TestReportExporter(unittest.TestCase):
    def test_export_stats_report_creates_workbook_with_expected_name_and_table(self):
        controller = MagicMock()
        controller.build_piece_stats_report.return_value = {
            "rows": [
                {"class_name": "OK", "OK": 15, "NOK": 0, "FOK": 0, "FNOK": 0, "Total": 15},
                {"class_name": "split", "OK": 0, "NOK": 5, "FOK": 0, "FNOK": 3, "Total": 8},
                {"class_name": "Total", "OK": 15, "NOK": 5, "FOK": 0, "FNOK": 3, "Total": 23, "is_total": True},
            ],
            "start_at": datetime.datetime(2026, 3, 25, 9, 0),
            "end_at": datetime.datetime(2026, 3, 25, 12, 0),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = export_stats_report(controller, output_dir=tmp_dir)

            path = Path(output_path)
            self.assertTrue(path.exists())
            self.assertEqual(
                path.name,
                "reporte_20260325_20260325_0900_1200.xlsx",
            )

            workbook = load_workbook(path)
            sheet = workbook["Stats"]
            self.assertEqual(sheet["A1"].value, "Class Name")
            self.assertEqual(sheet["B1"].value, "OK")
            self.assertEqual(sheet["F1"].value, "Total")
            self.assertEqual(sheet["A2"].value, "OK")
            self.assertEqual(sheet["B2"].value, 15)
            self.assertEqual(sheet["A4"].value, "Total")
            self.assertEqual(sheet["F4"].value, 23)

    def test_export_stats_report_requires_valid_db_date_range(self):
        controller = MagicMock()
        controller.build_piece_stats_report.return_value = {
            "rows": [
                {"class_name": "OK", "OK": 1, "NOK": 0, "FOK": 0, "FNOK": 0, "Total": 1},
                {"class_name": "Total", "OK": 1, "NOK": 0, "FOK": 0, "FNOK": 0, "Total": 1, "is_total": True},
            ],
            "start_at": None,
            "end_at": None,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "valid DB date range"):
                export_stats_report(controller, output_dir=tmp_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)

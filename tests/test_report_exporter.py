import datetime
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from openpyxl import load_workbook

from report_exporter import (
    export_combined_traceability_report,
    export_ok_nok_traceability_report,
    export_stats_report,
    parse_jsn_datetime,
)


class FakeTraceabilityDB:
    def __init__(self, rows):
        self.rows = rows

    def fetch(self, query, data=None):
        normalized = " ".join(query.split())
        if normalized == "SELECT jsn, model_result FROM piece_result ORDER BY jsn":
            return sorted(self.rows, key=lambda row: row["jsn"])
        raise AssertionError(f"Unhandled query: {normalized}")


class TestReportExporter(unittest.TestCase):
    def test_parse_jsn_datetime_reads_date_and_time_tokens(self):
        parsed = parse_jsn_datetime("218620514260607413863")

        self.assertEqual(parsed, datetime.datetime(2026, 5, 14, 6, 7, 41))

    def test_export_ok_nok_traceability_report_groups_by_day_and_hour(self):
        db = FakeTraceabilityDB(
            [
                {"jsn": "218620514260607413863", "model_result": "OK"},
                {"jsn": "218620514260645413864", "model_result": "OK"},
                {"jsn": "218620514260807413865", "model_result": "NOK"},
                {"jsn": "bad-jsn", "model_result": "NOK"},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = export_ok_nok_traceability_report(
                db,
                tmp_dir,
                created_at=datetime.datetime(2026, 5, 14, 9, 30, 0),
            )

            path = Path(output_path)
            self.assertEqual(path.name, "desglose_ok_nok_20260514_093000.xlsx")

            workbook = load_workbook(path)
            self.assertIn("Resumen OK-NOK", workbook.sheetnames)
            self.assertIn("Por dia", workbook.sheetnames)
            self.assertIn("Por hora", workbook.sheetnames)
            self.assertIn("Advertencias", workbook.sheetnames)

            summary = workbook["Resumen OK-NOK"]
            self.assertEqual(summary["C2"].value, 2)
            self.assertEqual(summary["D2"].value, 1)
            self.assertEqual(summary["E2"].value, 3)

            by_day = workbook["Por dia"]
            self.assertEqual(by_day["A2"].value, "2026-05-14")
            self.assertEqual(by_day["B2"].value, 2)
            self.assertEqual(by_day["C2"].value, 1)

            by_hour = workbook["Por hora"]
            self.assertEqual(by_hour["B2"].value, "06")
            self.assertEqual(by_hour["D2"].value, 2)
            self.assertEqual(by_hour["B3"].value, "07")
            self.assertEqual(by_hour["F3"].value, 0)
            self.assertEqual(by_hour["B4"].value, "08")
            self.assertEqual(by_hour["E4"].value, 1)

            warnings = workbook["Advertencias"]
            self.assertEqual(warnings["A2"].value, "bad-jsn")

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

    def test_export_combined_traceability_report_includes_stats_matrix_sheet(self):
        db = FakeTraceabilityDB(
            [
                {"jsn": "218620514260607413863", "model_result": "OK"},
                {"jsn": "218620514260645413864", "model_result": "OK"},
                {"jsn": "218620514260807413865", "model_result": "NOK"},
            ]
        )
        controller = MagicMock()
        controller.display.db = db
        controller.build_piece_stats_report.return_value = {
            "rows": [
                {"class_name": "OK", "OK": 2, "NOK": 0, "FOK": 0, "FNOK": 0, "Total": 2},
                {"class_name": "scratch", "OK": 0, "NOK": 1, "FOK": 0, "FNOK": 0, "Total": 1},
                {"class_name": "Total", "OK": 2, "NOK": 1, "FOK": 0, "FNOK": 0, "Total": 3, "is_total": True},
            ],
            "start_at": datetime.datetime(2026, 5, 14, 6, 0),
            "end_at": datetime.datetime(2026, 5, 14, 8, 0),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = export_combined_traceability_report(
                controller,
                db_client=db,
                output_dir=tmp_dir,
                created_at=datetime.datetime(2026, 5, 14, 9, 30, 0),
            )

            path = Path(output_path)
            self.assertEqual(path.name, "desglose_ok_nok_20260514_093000.xlsx")
            workbook = load_workbook(path)
            self.assertEqual(
                workbook.sheetnames,
                ["Resumen OK-NOK", "Por dia", "Por hora", "Advertencias", "Stats"],
            )
            self.assertEqual(workbook["Resumen OK-NOK"]["E2"].value, 3)
            self.assertEqual(workbook["Stats"]["F4"].value, 3)

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

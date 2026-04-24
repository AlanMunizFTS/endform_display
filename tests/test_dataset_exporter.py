import tempfile
import unittest
from pathlib import Path

from dataset_exporter import export_piece_stats_dataset, sanitize_dataset_folder_name
from file_manager import FileManager


class _DatasetControllerStub:
    def __init__(self, historic_dir, records):
        self.file_manager = FileManager()
        self.display = type("DisplayStub", (), {"db": object()})()
        self._historic_dir = str(historic_dir)
        self._records = list(records)

    def _get_export_historic_dir(self):
        return self._historic_dir

    def get_piece_stats_dataset_records(self, db_client=None):
        return list(self._records)


class TestDatasetExporter(unittest.TestCase):
    def test_sanitize_dataset_folder_name_normalizes_invalid_chars(self):
        self.assertEqual(
            sanitize_dataset_folder_name("dent / severe:01"),
            "dent_severe_01",
        )

    def test_export_piece_stats_dataset_creates_expected_tree_and_duplicates_multi_defect(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            historic_dir = Path(tmp_dir) / "historic"
            output_dir = Path(tmp_dir) / "datasets"
            historic_dir.mkdir()
            (historic_dir / "11861_Cam1_Side1_OK.png").write_bytes(b"side")
            (historic_dir / "11861_Cam2_Front_NOK.png").write_bytes(b"front")

            controller = _DatasetControllerStub(
                historic_dir=historic_dir,
                records=[
                    {
                        "img_name": "11861_Cam1_Side1_OK.png",
                        "result": "OK",
                        "angle": "side",
                        "class_names": ["UNCLASSIFIED"],
                    },
                    {
                        "img_name": "11861_Cam2_Front_NOK.png",
                        "result": "FOK",
                        "angle": "front",
                        "class_names": ["dent", "scratch / deep"],
                    },
                ],
            )

            result = export_piece_stats_dataset(
                controller,
                filters={
                    "results": ["OK", "FOK"],
                    "angles": ["side", "front"],
                    "class_names": ["All"],
                },
                output_dir=str(output_dir),
            )

            self.assertTrue(result["ok"])
            dataset_dir = Path(result["output_path"])
            self.assertTrue((dataset_dir / "OK" / "UNCLASSIFIED" / "side" / "11861_Cam1_Side1_OK.png").exists())
            self.assertTrue((dataset_dir / "FOK" / "dent" / "front" / "11861_Cam2_Front_NOK.png").exists())
            self.assertTrue(
                (
                    dataset_dir
                    / "FOK"
                    / "scratch_deep"
                    / "front"
                    / "11861_Cam2_Front_NOK.png"
                ).exists()
            )
            self.assertEqual(result["matched_images"], 2)
            self.assertEqual(result["copied_files"], 3)

    def test_export_piece_stats_dataset_reports_missing_files_without_failing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            historic_dir = Path(tmp_dir) / "historic"
            output_dir = Path(tmp_dir) / "datasets"
            historic_dir.mkdir()
            (historic_dir / "11861_Cam1_Side1_OK.png").write_bytes(b"side")

            controller = _DatasetControllerStub(
                historic_dir=historic_dir,
                records=[
                    {
                        "img_name": "11861_Cam1_Side1_OK.png",
                        "result": "OK",
                        "angle": "side",
                        "class_names": ["UNCLASSIFIED"],
                    },
                    {
                        "img_name": "11861_Cam2_Front_NOK.png",
                        "result": "NOK",
                        "angle": "front",
                        "class_names": ["dent"],
                    },
                ],
            )

            result = export_piece_stats_dataset(
                controller,
                filters={
                    "results": ["OK", "NOK"],
                    "angles": ["side", "front"],
                    "class_names": ["All"],
                },
                output_dir=str(output_dir),
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["missing_count"], 1)
            self.assertEqual(result["missing_files"], ["11861_Cam2_Front_NOK.png"])
            self.assertEqual(result["copied_files"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

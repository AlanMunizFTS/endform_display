import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from file_manager import FileManager
from state_package import (
    estimate_display_state_export_size,
    export_display_state,
    import_display_state,
)


def _unwrap_json_param(value):
    if hasattr(value, "adapted"):
        return value.adapted
    return value


class FakePackageCursor:
    def __init__(self, db):
        self.db = db
        self._results = []

    def execute(self, query, params=None):
        normalized = " ".join(query.split())
        params = params or ()

        if normalized == "SELECT img_name FROM img_results":
            self._results = [(row["img_name"],) for row in self.db.img_results]
            return
        if normalized == "SELECT jsn FROM piece_result":
            self._results = [(row["jsn"],) for row in self.db.piece_result]
            return
        if normalized == "SELECT img_name FROM classified_images":
            self._results = [(row["img_name"],) for row in self.db.classified_images]
            return
        if normalized == "SELECT img_name, class_name, confidence, model_name, geometry_type, coordinates, image_width, image_height FROM model_results":
            self._results = [
                (
                    row["img_name"],
                    row["class_name"],
                    row["confidence"],
                    row.get("model_name"),
                    row.get("geometry_type"),
                    row.get("coordinates"),
                    row.get("image_width"),
                    row.get("image_height"),
                )
                for row in self.db.model_results
            ]
            return
        if normalized.startswith(
            "SELECT ci.img_name, cid.class_name, cid.confidence"
        ):
            rows = []
            for row in self.db.classified_image_defects:
                classified = self.db.classified_by_id[row["classified_image_id"]]
                rows.append(
                    (
                        classified["img_name"],
                        row["class_name"],
                        row["confidence"],
                        row.get("remote_model_result_id"),
                    )
                )
            self._results = rows
            return
        if normalized == "SELECT remote_model_result_id FROM classified_image_defects WHERE remote_model_result_id IS NOT NULL":
            self._results = [
                (row["remote_model_result_id"],)
                for row in self.db.classified_image_defects
                if row.get("remote_model_result_id") is not None
            ]
            return
        if normalized.startswith(
            "SELECT pr.jsn, prd.class_name FROM piece_result_defects"
        ):
            rows = []
            for row in self.db.piece_result_defects:
                piece = self.db.piece_by_id[row["piece_result_id"]]
                rows.append((piece["jsn"], row["class_name"]))
            self._results = rows
            return
        if normalized == "SELECT id FROM piece_result WHERE jsn = %s":
            jsn = params[0]
            piece = self.db.piece_by_jsn.get(jsn)
            self._results = [] if piece is None else [(piece["id"],)]
            return
        if normalized == "SELECT id, piece_id FROM classified_images WHERE img_name = %s":
            img_name = params[0]
            classified = self.db.classified_by_img.get(img_name)
            self._results = [] if classified is None else [(classified["id"], classified["piece_id"])]
            return
        if normalized == "SELECT jsn FROM piece_result WHERE id = %s":
            piece_id = params[0]
            piece = self.db.piece_by_id.get(piece_id)
            self._results = [] if piece is None else [(piece["jsn"],)]
            return
        if normalized == "INSERT INTO img_results (img_name, result) VALUES (%s, %s)":
            self.db.img_results.append({"img_name": params[0], "result": params[1]})
            self.db._reindex()
            self._results = []
            return
        if normalized.startswith(
            "INSERT INTO piece_result (jsn, operator_result, model_result, created_at)"
        ):
            new_row = {
                "id": self.db.next_piece_id,
                "jsn": params[0],
                "operator_result": params[1],
                "model_result": params[2],
                "created_at": params[3],
            }
            self.db.next_piece_id += 1
            self.db.piece_result.append(new_row)
            self.db._reindex()
            self._results = []
            return
        if normalized.startswith(
            "INSERT INTO classified_images (img_name, operator_result, model_result, piece_id, created_at)"
        ):
            new_row = {
                "id": self.db.next_classified_id,
                "img_name": params[0],
                "operator_result": params[1],
                "model_result": params[2],
                "piece_id": params[3],
                "created_at": params[4],
            }
            self.db.next_classified_id += 1
            self.db.classified_images.append(new_row)
            self.db._reindex()
            self._results = []
            return
        if normalized.startswith(
            "INSERT INTO classified_image_defects (classified_image_id, class_name, confidence, created_at"
        ):
            self.db.classified_image_defects.append(
                {
                    "classified_image_id": params[0],
                    "class_name": params[1],
                    "confidence": params[2],
                    "created_at": params[3],
                    "remote_model_result_id": params[4] if len(params) > 4 else None,
                    "model_name": params[5] if len(params) > 5 else None,
                    "geometry_type": params[6] if len(params) > 6 else None,
                    "coordinates": _unwrap_json_param(params[7]) if len(params) > 7 else None,
                    "image_width": params[8] if len(params) > 8 else None,
                    "image_height": params[9] if len(params) > 9 else None,
                }
            )
            self._results = []
            return
        if normalized.startswith(
            "INSERT INTO model_results (img_name, class_name, confidence, created_at, model_name, geometry_type, coordinates, image_width, image_height)"
        ):
            self.db.model_results.append(
                {
                    "img_name": params[0],
                    "class_name": params[1],
                    "confidence": params[2],
                    "created_at": params[3],
                    "model_name": params[4],
                    "geometry_type": params[5],
                    "coordinates": _unwrap_json_param(params[6]),
                    "image_width": params[7],
                    "image_height": params[8],
                }
            )
            self._results = []
            return
        if normalized.startswith(
            "INSERT INTO piece_result_defects (piece_result_id, class_name, confidence, created_at)"
        ):
            self.db.piece_result_defects.append(
                {
                    "piece_result_id": params[0],
                    "class_name": params[1],
                    "confidence": params[2],
                    "created_at": params[3],
                }
            )
            self._results = []
            return

        raise AssertionError(f"Unhandled query: {normalized}")

    def fetchall(self):
        return list(self._results)

    def fetchone(self):
        return self._results[0] if self._results else None


class FakePackageDB:
    def __init__(
        self,
        img_results=None,
        piece_result=None,
        classified_images=None,
        classified_image_defects=None,
        model_results=None,
        piece_result_defects=None,
    ):
        self.img_results = list(img_results or [])
        self.piece_result = list(piece_result or [])
        self.classified_images = list(classified_images or [])
        self.classified_image_defects = list(classified_image_defects or [])
        self.model_results = list(model_results or [])
        self.piece_result_defects = list(piece_result_defects or [])
        self.next_piece_id = max((row["id"] for row in self.piece_result), default=0) + 1
        self.next_classified_id = max((row["id"] for row in self.classified_images), default=0) + 1
        self._reindex()

    def _reindex(self):
        self.piece_by_id = {row["id"]: row for row in self.piece_result}
        self.piece_by_jsn = {row["jsn"]: row for row in self.piece_result}
        self.classified_by_id = {row["id"]: row for row in self.classified_images}
        self.classified_by_img = {row["img_name"]: row for row in self.classified_images}

    def fetch(self, query, data=None):
        normalized = " ".join(query.split())
        if normalized == "SELECT img_name, result FROM img_results ORDER BY img_name":
            return sorted(self.img_results, key=lambda row: row["img_name"])
        if normalized == "SELECT jsn, operator_result, model_result, created_at FROM piece_result ORDER BY jsn":
            return [
                {
                    "jsn": row["jsn"],
                    "operator_result": row["operator_result"],
                    "model_result": row["model_result"],
                    "created_at": row["created_at"],
                }
                for row in sorted(self.piece_result, key=lambda row: row["jsn"])
            ]
        if normalized.startswith(
            "SELECT ci.img_name, pr.jsn, ci.operator_result, ci.model_result, ci.created_at FROM classified_images"
        ):
            rows = []
            for row in sorted(self.classified_images, key=lambda entry: entry["img_name"]):
                piece = self.piece_by_id.get(row["piece_id"])
                rows.append(
                    {
                        "img_name": row["img_name"],
                        "jsn": None if piece is None else piece["jsn"],
                        "operator_result": row["operator_result"],
                        "model_result": row["model_result"],
                        "created_at": row["created_at"],
                    }
                )
            return rows
        if normalized.startswith(
            "SELECT ci.img_name, cid.class_name, cid.confidence, cid.created_at"
        ):
            rows = []
            for row in self.classified_image_defects:
                classified = self.classified_by_id[row["classified_image_id"]]
                rows.append(
                    {
                        "img_name": classified["img_name"],
                        "class_name": row["class_name"],
                        "confidence": row["confidence"],
                        "created_at": row["created_at"],
                        "remote_model_result_id": row.get("remote_model_result_id"),
                        "model_name": row.get("model_name"),
                        "geometry_type": row.get("geometry_type"),
                        "coordinates": row.get("coordinates"),
                        "image_width": row.get("image_width"),
                        "image_height": row.get("image_height"),
                    }
                )
            return sorted(rows, key=lambda entry: (entry["img_name"], entry["class_name"]))
        if normalized.startswith(
            "SELECT img_name, class_name, confidence, created_at, model_name, geometry_type, coordinates, image_width, image_height FROM model_results"
        ):
            return sorted(
                [
                    {
                        "img_name": row["img_name"],
                        "class_name": row["class_name"],
                        "confidence": row["confidence"],
                        "created_at": row["created_at"],
                        "model_name": row.get("model_name"),
                        "geometry_type": row.get("geometry_type"),
                        "coordinates": row.get("coordinates"),
                        "image_width": row.get("image_width"),
                        "image_height": row.get("image_height"),
                    }
                    for row in self.model_results
                ],
                key=lambda entry: (
                    entry["img_name"],
                    entry["class_name"],
                    entry["confidence"],
                ),
            )
        if normalized.startswith(
            "SELECT pr.jsn, prd.class_name, prd.confidence, prd.created_at FROM piece_result_defects"
        ):
            rows = []
            for row in self.piece_result_defects:
                piece = self.piece_by_id[row["piece_result_id"]]
                rows.append(
                    {
                        "jsn": piece["jsn"],
                        "class_name": row["class_name"],
                        "confidence": row["confidence"],
                        "created_at": row["created_at"],
                    }
                )
            return sorted(rows, key=lambda entry: (entry["jsn"], entry["class_name"]))

        raise AssertionError(f"Unhandled fetch query: {normalized}")

    @contextmanager
    def get_cursor(self):
        cursor = FakePackageCursor(self)
        yield cursor
        self._reindex()


def build_controller(tmp_dir, db, historic_mode=False):
    tmp_path = Path(tmp_dir)
    annotated_dir = tmp_path / "annotated"
    historic_dir = tmp_path / "historic"
    annotated_dir.mkdir(exist_ok=True)
    historic_dir.mkdir(exist_ok=True)
    display = SimpleNamespace(db=db, historic_mode=historic_mode)
    controller = SimpleNamespace(
        file_manager=FileManager(),
        display=display,
        config=SimpleNamespace(image_extensions=(".png", ".jpg", ".jpeg", ".bmp")),
        _get_visible_historic_dir=lambda: str(annotated_dir),
        _get_export_historic_dir=lambda: str(historic_dir),
        _recalculate_piece_result=MagicMock(),
        _invalidate_dataset_runtime_state=MagicMock(),
        enter_historic_mode=MagicMock(),
    )
    return controller, annotated_dir, historic_dir


def build_source_db():
    return FakePackageDB(
        img_results=[
            {"img_name": "11861_A_side_OK.png", "result": "OK"},
            {"img_name": "11861_A_front_NOK.png", "result": "NOK"},
        ],
        piece_result=[
            {
                "id": 1,
                "jsn": "11861",
                "operator_result": "NOK",
                "model_result": "NOK",
                "created_at": "2026-03-26 10:00:00",
            }
        ],
        classified_images=[
            {
                "id": 1,
                "img_name": "11861_A_side_OK.png",
                "operator_result": "OK",
                "model_result": "OK",
                "piece_id": 1,
                "created_at": "2026-03-26 10:00:01",
            },
            {
                "id": 2,
                "img_name": "11861_A_front_NOK.png",
                "operator_result": "NOK",
                "model_result": "NOK",
                "piece_id": 1,
                "created_at": "2026-03-26 10:00:02",
            },
        ],
        classified_image_defects=[
            {
                "classified_image_id": 2,
                "class_name": "dent",
                "confidence": "0.9000",
                "created_at": "2026-03-26 10:00:03",
                "remote_model_result_id": 25,
                "model_name": "remote-yolo",
                "geometry_type": "bbox",
                "coordinates": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
                "image_width": 360,
                "image_height": 360,
            }
        ],
        model_results=[
            {
                "img_name": "11861_A_side_OK.png",
                "class_name": "OK",
                "confidence": "1.0000",
                "created_at": "2026-03-26 10:00:01",
                "model_name": "remote-yolo",
                "geometry_type": "classification",
                "coordinates": None,
                "image_width": 360,
                "image_height": 360,
            },
            {
                "img_name": "11861_A_front_NOK.png",
                "class_name": "dent",
                "confidence": "0.9000",
                "created_at": "2026-03-26 10:00:03",
                "model_name": "remote-yolo",
                "geometry_type": "bbox",
                "coordinates": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
                "image_width": 360,
                "image_height": 360,
            },
        ],
        piece_result_defects=[
            {
                "piece_result_id": 1,
                "class_name": "dent",
                "confidence": "0.9000",
                "created_at": "2026-03-26 10:00:04",
            }
        ],
    )


class TestStatePackage(unittest.TestCase):
    def test_export_display_state_creates_expected_folder_structure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_db = build_source_db()
            controller, annotated_dir, historic_dir = build_controller(tmp_dir, source_db)
            (historic_dir / "11861_A_side_OK.png").write_bytes(b"historic-side")
            (historic_dir / "11861_A_front_NOK.png").write_bytes(b"historic-front")

            export_dir = Path(tmp_dir) / "exports"
            result = export_display_state(
                controller,
                output_dir=str(export_dir),
                db_client=source_db,
            )

            self.assertTrue(result["ok"])
            package_path = Path(result["package_path"])
            self.assertTrue(package_path.is_dir())
            self.assertFalse(Path(f"{package_path}.partial").exists())
            self.assertTrue((package_path / "manifest.json").is_file())
            self.assertTrue((package_path / "db" / "data.json").is_file())
            self.assertTrue((package_path / "db" / "database.sql").is_file())
            self.assertFalse((package_path / "annotated").exists())
            self.assertTrue((package_path / "historic" / "11861_A_front_NOK.png").is_file())

            manifest = json.loads((package_path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["package_kind"], "display_state")
            self.assertTrue(manifest["export_complete"])
            self.assertEqual(manifest["annotated_count"], 0)
            self.assertEqual(manifest["historic_count"], 2)
            self.assertEqual(len(manifest["annotated_images"]), 0)
            self.assertEqual(len(manifest["historic_images"]), 2)
            self.assertEqual(manifest["table_counts"]["img_results"], 2)
            self.assertEqual(manifest["table_counts"]["model_results"], 2)

            data_payload = json.loads((package_path / "db" / "data.json").read_text(encoding="utf-8"))
            self.assertEqual(len(data_payload["classified_images"]), 2)
            self.assertEqual(len(data_payload["classified_image_defects"]), 1)
            self.assertEqual(len(data_payload["model_results"]), 2)
            self.assertEqual(
                data_payload["classified_image_defects"][0]["coordinates"],
                {"x1": 10, "x2": 100, "y1": 20, "y2": 120},
            )

    def test_estimate_display_state_export_size_includes_files_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_db = build_source_db()
            controller, annotated_dir, historic_dir = build_controller(tmp_dir, source_db)
            (annotated_dir / "11861_A_side_OK.png").write_bytes(b"annotated")
            (historic_dir / "11861_A_side_OK.png").write_bytes(b"historic")

            result = estimate_display_state_export_size(controller, db_client=source_db)

            self.assertTrue(result["ok"])
            self.assertGreaterEqual(result["required_bytes"], len(b"annotated") + len(b"historic"))
            self.assertEqual(result["annotated_count"], 1)
            self.assertEqual(result["historic_count"], 1)
            self.assertEqual(result["table_counts"]["img_results"], 2)

    def test_import_display_state_merges_missing_data_and_skips_duplicates(self):
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as target_tmp:
            source_db = build_source_db()
            source_controller, annotated_dir, historic_dir = build_controller(source_tmp, source_db)
            (historic_dir / "11861_A_side_OK.png").write_bytes(b"source-historic-side")
            (historic_dir / "11861_A_front_NOK.png").write_bytes(b"source-historic-front")
            package_result = export_display_state(source_controller, output_dir=source_tmp, db_client=source_db)

            target_db = FakePackageDB(
                img_results=[{"img_name": "11861_A_side_OK.png", "result": "OK"}],
                piece_result=[
                    {
                        "id": 1,
                        "jsn": "11861",
                        "operator_result": "OK",
                        "model_result": "OK",
                        "created_at": "2026-03-26 08:00:00",
                    }
                ],
                classified_images=[
                    {
                        "id": 1,
                        "img_name": "11861_A_side_OK.png",
                        "operator_result": "OK",
                        "model_result": "OK",
                        "piece_id": 1,
                        "created_at": "2026-03-26 08:00:01",
                    }
                ],
            )
            target_controller, target_annotated, target_historic = build_controller(target_tmp, target_db)
            (target_annotated / "11861_A_side_OK.png").write_bytes(b"existing-annotated")
            (target_historic / "11861_A_side_OK.png").write_bytes(b"existing-historic")

            result = import_display_state(
                target_controller,
                package_result["package_path"],
                db_client=target_db,
            )

            self.assertTrue(result["ok"])
            self.assertEqual((target_annotated / "11861_A_side_OK.png").read_bytes(), b"existing-annotated")
            self.assertFalse((target_annotated / "11861_A_front_NOK.png").exists())
            self.assertTrue((target_historic / "11861_A_front_NOK.png").exists())
            self.assertEqual(result["annotated"]["copied"], 0)
            self.assertEqual(result["annotated"]["skipped"], 0)
            self.assertEqual(result["historic"]["copied"], 1)
            self.assertEqual(result["historic"]["skipped"], 1)
            self.assertEqual(result["db"]["inserted"]["img_results"], 1)
            self.assertEqual(result["db"]["inserted"]["classified_images"], 1)
            self.assertEqual(result["db"]["inserted"]["model_results"], 2)
            dent_result = next(
                row for row in target_db.model_results if row["class_name"] == "dent"
            )
            self.assertEqual(
                dent_result["coordinates"],
                {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
            )
            target_controller._recalculate_piece_result.assert_called_with("11861", db_client=target_db)
            target_controller._invalidate_dataset_runtime_state.assert_called_once_with(
                clear_historic_images=False
            )

    def test_import_display_state_with_existing_jsn_imports_new_image_and_recalculates(self):
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as target_tmp:
            source_db = FakePackageDB(
                img_results=[{"img_name": "11861_extra_diag_NOK.png", "result": "NOK"}],
                piece_result=[
                    {
                        "id": 1,
                        "jsn": "11861",
                        "operator_result": "NOK",
                        "model_result": "NOK",
                        "created_at": "2026-03-26 10:30:00",
                    }
                ],
                classified_images=[
                    {
                        "id": 1,
                        "img_name": "11861_extra_diag_NOK.png",
                        "operator_result": "NOK",
                        "model_result": "NOK",
                        "piece_id": 1,
                        "created_at": "2026-03-26 10:30:01",
                    }
                ],
            )
            source_controller, annotated_dir, historic_dir = build_controller(source_tmp, source_db)
            (historic_dir / "11861_extra_diag_NOK.png").write_bytes(b"new-historic")
            package_result = export_display_state(source_controller, output_dir=source_tmp, db_client=source_db)

            target_db = FakePackageDB(
                img_results=[{"img_name": "11861_existing_side_OK.png", "result": "OK"}],
                piece_result=[
                    {
                        "id": 7,
                        "jsn": "11861",
                        "operator_result": "OK",
                        "model_result": "OK",
                        "created_at": "2026-03-26 09:00:00",
                    }
                ],
                classified_images=[
                    {
                        "id": 4,
                        "img_name": "11861_existing_side_OK.png",
                        "operator_result": "OK",
                        "model_result": "OK",
                        "piece_id": 7,
                        "created_at": "2026-03-26 09:00:01",
                    }
                ],
            )
            target_controller, target_annotated, target_historic = build_controller(target_tmp, target_db)

            result = import_display_state(
                target_controller,
                package_result["package_path"],
                db_client=target_db,
            )

            self.assertTrue(result["ok"])
            self.assertFalse((target_annotated / "11861_extra_diag_NOK.png").exists())
            self.assertTrue((target_historic / "11861_extra_diag_NOK.png").exists())
            self.assertIn("11861_extra_diag_NOK.png", [row["img_name"] for row in target_db.img_results])
            self.assertIn("11861_extra_diag_NOK.png", [row["img_name"] for row in target_db.classified_images])
            target_controller._recalculate_piece_result.assert_called_with("11861", db_client=target_db)

    def test_import_display_state_rejects_invalid_package(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            invalid_package = Path(tmp_dir) / "invalid_package"
            (invalid_package / "db").mkdir(parents=True)
            (invalid_package / "db" / "data.json").write_text("{}", encoding="utf-8")

            controller, _annotated_dir, _historic_dir = build_controller(tmp_dir, FakePackageDB())

            with self.assertRaises(ValueError):
                import_display_state(controller, str(invalid_package), db_client=controller.display.db)

    def test_import_display_state_rejects_partial_export_folder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            partial_package = Path(tmp_dir) / "display_state_20260429_120000.partial"
            partial_package.mkdir()
            controller, _annotated_dir, _historic_dir = build_controller(tmp_dir, FakePackageDB())

            with self.assertRaises(ValueError):
                import_display_state(controller, str(partial_package), db_client=controller.display.db)

    def test_import_display_state_rejects_missing_manifest_listed_image(self):
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as target_tmp:
            source_db = build_source_db()
            source_controller, annotated_dir, historic_dir = build_controller(source_tmp, source_db)
            (historic_dir / "11861_A_side_OK.png").write_bytes(b"historic-side")
            (historic_dir / "11861_A_front_NOK.png").write_bytes(b"historic-front")
            package_result = export_display_state(source_controller, output_dir=source_tmp, db_client=source_db)
            package_path = Path(package_result["package_path"])
            (package_path / "historic" / "11861_A_front_NOK.png").unlink()

            target_controller, _target_annotated, _target_historic = build_controller(
                target_tmp,
                FakePackageDB(),
            )

            with self.assertRaises(ValueError):
                import_display_state(
                    target_controller,
                    package_result["package_path"],
                    db_client=target_controller.display.db,
                )

    def test_export_import_roundtrip_reproduces_files_and_db_rows(self):
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as target_tmp:
            source_db = build_source_db()
            source_controller, annotated_dir, historic_dir = build_controller(source_tmp, source_db)
            (historic_dir / "11861_A_side_OK.png").write_bytes(b"historic-side")
            (historic_dir / "11861_A_front_NOK.png").write_bytes(b"historic-front")

            package_result = export_display_state(source_controller, output_dir=source_tmp, db_client=source_db)

            target_db = FakePackageDB()
            target_controller, target_annotated, target_historic = build_controller(target_tmp, target_db)
            result = import_display_state(
                target_controller,
                package_result["package_path"],
                db_client=target_db,
            )

            self.assertTrue(result["ok"])
            self.assertEqual(sorted(p.name for p in target_annotated.iterdir()), [])
            self.assertEqual(sorted(p.name for p in target_historic.iterdir()), [
                "11861_A_front_NOK.png",
                "11861_A_side_OK.png",
            ])
            self.assertEqual(len(target_db.img_results), 2)
            self.assertEqual(len(target_db.classified_images), 2)
            self.assertEqual(len(target_db.model_results), 2)
            self.assertEqual(len(target_db.piece_result), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

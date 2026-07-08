import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from utilities import infer_to_historic


class _FakeBoxes:
    def __init__(self, conf):
        self.conf = conf


class _FakeResult:
    def __init__(self, conf, annotated_value=200):
        self.obb = None
        self.boxes = _FakeBoxes(conf)
        self.annotated_value = annotated_value

    def plot(self):
        return np.full((8, 8, 3), self.annotated_value, dtype=np.uint8)


class _FakeModel:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def predict(self, image_path, **kwargs):
        self.calls.append((image_path, kwargs))
        return [self.result]


class TestInferToHistoric(unittest.TestCase):
    def test_build_status_filename_strips_existing_status(self):
        self.assertEqual(
            infer_to_historic.build_status_filename("11861_side.png", "OK"),
            "11861_side_OK.png",
        )
        self.assertEqual(
            infer_to_historic.build_status_filename("11861_side_OK.png", "NOK"),
            "11861_side_NOK.png",
        )
        self.assertEqual(
            infer_to_historic.build_status_filename("11861_side_nok.jpg", "OK"),
            "11861_side_OK.jpg",
        )

    def test_load_models_routes_pt_files_by_position(self):
        loaded_paths = []

        class FakeYOLO:
            def __init__(self, path):
                loaded_paths.append(Path(path).name)

        with tempfile.TemporaryDirectory() as tmp_dir:
            models_dir = Path(tmp_dir)
            (models_dir / "wrinkle_side.pt").write_bytes(b"model")
            (models_dir / "breakage_front.pt").write_bytes(b"model")
            (models_dir / "nylon_diag.pt").write_bytes(b"model")
            (models_dir / "unmatched.pt").write_bytes(b"model")
            (models_dir / "notes.txt").write_text("ignore", encoding="utf-8")

            models = infer_to_historic.load_models(models_dir, yolo_cls=FakeYOLO)

        self.assertEqual(loaded_paths, ["breakage_front.pt", "nylon_diag.pt", "wrinkle_side.pt"])
        self.assertEqual(len(models["side"]), 1)
        self.assertEqual(len(models["front"]), 1)
        self.assertEqual(len(models["diag"]), 1)
        self.assertEqual(len(models["unknown"]), 0)

    def test_no_detection_copies_original_to_historic_ok_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            input_dir = base_dir / "input"
            historic_dir = base_dir / "historic"
            input_dir.mkdir()
            source = input_dir / "11861_cam_side.png"
            source.write_bytes(b"raw-image-bytes")

            model = _FakeModel(_FakeResult([0.1]))
            summary = infer_to_historic.process_images(
                input_dir=input_dir,
                models_by_position={"side": [model]},
                historic_dir=historic_dir,
                confidence=0.33,
                device="cpu",
            )

            self.assertEqual(summary, {"processed": 1, "ok": 1, "nok": 0, "skipped": 0})
            self.assertEqual((historic_dir / "11861_cam_side_OK.png").read_bytes(), b"raw-image-bytes")
            self.assertFalse((historic_dir / "11861_cam_side_NOK.png").exists())
            self.assertFalse((base_dir / "annotated").exists())
            self.assertEqual(model.calls[0][1]["device"], "cpu")

    def test_detection_saves_annotated_image_to_historic_nok_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            input_dir = base_dir / "input"
            historic_dir = base_dir / "historic"
            input_dir.mkdir()
            source = input_dir / "11861_cam_front_OK.png"
            source.write_bytes(b"raw-image-bytes")

            model = _FakeModel(_FakeResult([0.91], annotated_value=255))
            summary = infer_to_historic.process_images(
                input_dir=input_dir,
                models_by_position={"front": [model]},
                historic_dir=historic_dir,
                confidence=0.33,
                device="cpu",
            )

            output_path = historic_dir / "11861_cam_front_NOK.png"
            saved = cv2.imread(str(output_path))

            self.assertEqual(summary, {"processed": 1, "ok": 0, "nok": 1, "skipped": 0})
            self.assertTrue(output_path.exists())
            self.assertIsNotNone(saved)
            self.assertGreater(int(saved.sum()), 0)
            self.assertFalse((historic_dir / "11861_cam_front_OK.png").exists())
            self.assertFalse((base_dir / "annotated").exists())

    def test_rerun_removes_stale_status_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            input_dir = base_dir / "input"
            historic_dir = base_dir / "historic"
            input_dir.mkdir()
            historic_dir.mkdir()
            source = input_dir / "11861_cam_diag.png"
            source.write_bytes(b"raw-image-bytes")
            (historic_dir / "11861_cam_diag_OK.png").write_bytes(b"old-ok")

            model = _FakeModel(_FakeResult([0.88]))
            infer_to_historic.process_images(
                input_dir=input_dir,
                models_by_position={"diag": [model]},
                historic_dir=historic_dir,
                confidence=0.33,
                device="cpu",
            )

            self.assertTrue((historic_dir / "11861_cam_diag_NOK.png").exists())
            self.assertFalse((historic_dir / "11861_cam_diag_OK.png").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)

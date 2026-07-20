"""Run YOLO defect inference or piece segmentation into historic.

This utility intentionally does not write to the database. The display app will
pick up saved ``*_OK`` and ``*_NOK`` files from ``tmp_display/historic`` using
its existing bootstrap flow. Defect inference remains the default workflow;
piece segmentation is enabled explicitly with ``--mode segment``.
"""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    from paths_config import HISTORIC_LOCAL_DIR
except ModuleNotFoundError:
    # Support direct execution via ``python utilities/infer_to_historic.py``.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paths_config import HISTORIC_LOCAL_DIR


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
POSITIONS = ("side", "front", "diag")
DEFAULT_CONFIDENCE = 0.33
DEFAULT_HISTORIC_DIR = HISTORIC_LOCAL_DIR
SEGMENT_IMAGE_SIZE = 640


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run YOLO defect inference or piece segmentation into historic."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing images to process")
    parser.add_argument(
        "--mode",
        choices=("defects", "segment"),
        default="defects",
        help="Inference workflow to run. Default: %(default)s",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Folder containing YOLO .pt models. Default: %(default)s",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="YOLO segmentation .pt model. Required when --mode segment is used.",
    )
    parser.add_argument(
        "--historic-dir",
        type=Path,
        default=DEFAULT_HISTORIC_DIR,
        help="Historic output folder. Default: %(default)s",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Detection confidence threshold. Default: %(default)s",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Ultralytics device value, for example cpu, 0, or 0,1. Default: %(default)s",
    )
    args = parser.parse_args(argv)
    if args.mode == "segment" and args.model is None:
        parser.error("--model is required when --mode segment is used")
    return args


def get_position(filename):
    lower_name = str(filename or "").lower()
    for position in POSITIONS:
        if position in lower_name:
            return position
    return None


def strip_trailing_status(filename):
    path = Path(filename)
    stem = path.stem
    lower_stem = stem.lower()
    if lower_stem.endswith("_nok"):
        stem = stem[:-4]
    elif lower_stem.endswith("_ok"):
        stem = stem[:-3]
    return f"{stem}{path.suffix}"


def build_status_filename(filename, status):
    normalized = Path(strip_trailing_status(filename))
    status_value = str(status or "").strip().upper()
    if status_value not in {"OK", "NOK"}:
        raise ValueError(f"Unsupported status: {status!r}")
    return f"{normalized.stem}_{status_value}{normalized.suffix}"


def collect_images(input_dir):
    input_path = Path(input_dir)
    return [
        path
        for path in sorted(input_path.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def load_models(models_dir, yolo_cls=None):
    if yolo_cls is None:
        from ultralytics import YOLO

        yolo_cls = YOLO

    models_by_position = defaultdict(list)
    models_path = Path(models_dir)
    if not models_path.is_dir():
        print(f"[WARN] Models folder not found: {models_path}")
        return models_by_position

    for model_path in sorted(models_path.iterdir()):
        if not model_path.is_file() or model_path.suffix.lower() != ".pt":
            continue
        position = get_position(model_path.name)
        if position is None:
            continue
        models_by_position[position].append(yolo_cls(str(model_path)))
        print(f"[MODEL] Loaded {model_path.name} -> {position}")

    for position in POSITIONS:
        count = len(models_by_position.get(position, []))
        if count:
            print(f"[MODEL] {position}: {count} model(s)")

    return models_by_position


def _prediction_result(results):
    if isinstance(results, (list, tuple)):
        return results[0] if results else None
    return results


def _float_values(values):
    if values is None:
        return []
    try:
        iterable = values.tolist()
    except AttributeError:
        iterable = values
    try:
        return [float(value) for value in iterable]
    except TypeError:
        try:
            return [float(iterable)]
        except (TypeError, ValueError):
            return []
    except ValueError:
        return []


def _has_confidence_above(values, confidence_threshold):
    return any(value > confidence_threshold for value in _float_values(values))


def has_high_confidence_detection(result, confidence_threshold):
    if result is None:
        return False

    obb = getattr(result, "obb", None)
    if obb is not None and _has_confidence_above(getattr(obb, "conf", None), confidence_threshold):
        return True

    boxes = getattr(result, "boxes", None)
    if boxes is not None and _has_confidence_above(
        getattr(boxes, "conf", None),
        confidence_threshold,
    ):
        return True

    return False


def _run_model(model, image_path, confidence, device):
    kwargs = {"verbose": False, "conf": confidence}
    if device:
        kwargs["device"] = device

    predict = getattr(model, "predict", None)
    if callable(predict):
        return predict(str(image_path), **kwargs)
    return model(str(image_path), **kwargs)


def infer_image(image_path, models_by_position, confidence, device):
    position = get_position(Path(image_path).name)
    models = list(models_by_position.get(position, [])) if position else []

    for model in models:
        results = _run_model(model, image_path, confidence, device)
        result = _prediction_result(results)
        if has_high_confidence_detection(result, confidence):
            return {
                "position": position,
                "model_count": len(models),
                "status": "NOK",
                "result": result,
            }

    return {
        "position": position,
        "model_count": len(models),
        "status": "OK",
        "result": None,
    }


def _same_path(left, right):
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return False


def _remove_stale_status_outputs(historic_dir, source_path, output_path, filename):
    for status in ("OK", "NOK"):
        candidate = Path(historic_dir) / build_status_filename(filename, status)
        if _same_path(candidate, output_path):
            continue
        if _same_path(candidate, source_path):
            continue
        if candidate.exists():
            candidate.unlink()


def save_result_to_historic(image_path, inference, historic_dir):
    historic_path = Path(historic_dir)
    historic_path.mkdir(parents=True, exist_ok=True)

    status = inference["status"]
    output_name = build_status_filename(Path(image_path).name, status)
    output_path = historic_path / output_name

    if status == "NOK":
        result = inference.get("result")
        annotated = None
        if result is not None:
            try:
                annotated = result.plot()
            except Exception as exc:
                print(f"[WARN] Could not plot {Path(image_path).name}: {exc}")
        if annotated is not None:
            if not cv2.imwrite(str(output_path), annotated):
                raise IOError(f"Could not write annotated image: {output_path}")
        elif not _same_path(image_path, output_path):
            shutil.copy2(image_path, output_path)
    elif not _same_path(image_path, output_path):
        shutil.copy2(image_path, output_path)

    _remove_stale_status_outputs(
        historic_dir=historic_path,
        source_path=image_path,
        output_path=output_path,
        filename=Path(image_path).name,
    )
    return output_path


def process_images(input_dir, models_by_position, historic_dir, confidence, device):
    image_paths = collect_images(input_dir)
    if not image_paths:
        print(f"[INFO] No images found in {input_dir}")
        return {"processed": 0, "ok": 0, "nok": 0, "skipped": 0}

    summary = {"processed": 0, "ok": 0, "nok": 0, "skipped": 0}
    for image_path in image_paths:
        try:
            inference = infer_image(
                image_path=image_path,
                models_by_position=models_by_position,
                confidence=confidence,
                device=device,
            )
            output_path = save_result_to_historic(image_path, inference, historic_dir)
            status = inference["status"]
            summary["processed"] += 1
            summary[status.lower()] += 1
            position = inference.get("position") or "unknown"
            model_count = inference.get("model_count", 0)
            print(
                f"[{status}] {image_path.name} -> {output_path.name} "
                f"(position={position}, models={model_count})"
            )
        except Exception as exc:
            summary["skipped"] += 1
            print(f"[ERROR] Skipping {image_path}: {exc}")
    return summary


def load_segmentation_model(model_path, yolo_cls=None):
    """Load the single YOLO model used by the segmentation workflow."""
    if yolo_cls is None:
        from ultralytics import YOLO

        yolo_cls = YOLO

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Segmentation model does not exist: {path}")
    if path.suffix.lower() != ".pt":
        raise ValueError(f"Segmentation model must be a .pt file: {path}")
    model = yolo_cls(str(path))
    print(f"[MODEL] Loaded segmentation model: {path.name}")
    return model


def _to_numpy(values):
    if values is None:
        return None
    cpu = getattr(values, "cpu", None)
    if callable(cpu):
        values = cpu()
    numpy_method = getattr(values, "numpy", None)
    if callable(numpy_method):
        values = numpy_method()
    return np.asarray(values)


def pad_to_square(image, size=SEGMENT_IMAGE_SIZE):
    """Center an image on a black square canvas and resize it."""
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("Object crop image is empty")

    height, width = image.shape[:2]
    max_dimension = max(height, width)
    if max_dimension <= 0:
        raise ValueError("Object crop has invalid dimensions")

    if image.ndim == 2:
        padded = np.zeros((max_dimension, max_dimension), dtype=image.dtype)
    else:
        padded = np.zeros(
            (max_dimension, max_dimension, image.shape[2]),
            dtype=image.dtype,
        )
    x_offset = (max_dimension - width) // 2
    y_offset = (max_dimension - height) // 2
    padded[y_offset : y_offset + height, x_offset : x_offset + width] = image
    return cv2.resize(padded, (size, size))


def extract_first_segmented_object(result, original_image, image_size=SEGMENT_IMAGE_SIZE):
    """Return the first valid masked object crop in an Ultralytics result."""
    if result is None:
        raise ValueError("Segmentation model returned no result")
    if original_image is None or getattr(original_image, "size", 0) == 0:
        raise ValueError("Could not read input image")

    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    polygons = getattr(masks, "xy", None) if masks is not None else None
    box_values = _to_numpy(getattr(boxes, "xyxy", None)) if boxes is not None else None
    if polygons is None or box_values is None:
        raise ValueError("Segmentation result has no masks or bounding boxes")

    box_values = np.atleast_2d(box_values)
    height, width = original_image.shape[:2]
    for polygon, bbox in zip(polygons, box_values):
        try:
            contour = _to_numpy(polygon).astype(np.int32).reshape(-1, 1, 2)
            if len(contour) < 3 or np.asarray(bbox).size < 4:
                continue

            x1, y1, x2, y2 = np.asarray(bbox).reshape(-1)[:4]
            x1 = max(0, min(width, int(np.floor(x1))))
            y1 = max(0, min(height, int(np.floor(y1))))
            x2 = max(0, min(width, int(np.ceil(x2))))
            y2 = max(0, min(height, int(np.ceil(y2))))
            if x2 <= x1 or y2 <= y1:
                continue

            binary_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(binary_mask, [contour], -1, 255, cv2.FILLED)
            masked = cv2.bitwise_and(original_image, original_image, mask=binary_mask)
            crop = masked[y1:y2, x1:x2]
            if crop.size == 0 or not np.any(binary_mask[y1:y2, x1:x2]):
                continue
            return pad_to_square(crop, size=image_size)
        except Exception:
            continue

    raise ValueError("No valid segmented objects found")


def segment_image(image_path, model, confidence, device):
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        raise ValueError("Could not read input image")
    results = _run_model(model, image_path, confidence, device)
    result = _prediction_result(results)
    return extract_first_segmented_object(result, original_image)


def save_segment_to_historic(image_path, segmented_image, historic_dir):
    historic_path = Path(historic_dir)
    historic_path.mkdir(parents=True, exist_ok=True)
    output_name = build_status_filename(Path(image_path).name, "OK")
    output_path = historic_path / output_name
    if not cv2.imwrite(str(output_path), segmented_image):
        raise IOError(f"Could not write segmented image: {output_path}")
    _remove_stale_status_outputs(
        historic_dir=historic_path,
        source_path=image_path,
        output_path=output_path,
        filename=Path(image_path).name,
    )
    return output_path


def process_segment_images(input_dir, model, historic_dir, confidence, device):
    image_paths = collect_images(input_dir)
    if not image_paths:
        print(f"[INFO] No images found in {input_dir}")
        return {"processed": 0, "skipped": 0}

    summary = {"processed": 0, "skipped": 0}
    for image_path in image_paths:
        try:
            segmented_image = segment_image(image_path, model, confidence, device)
            output_path = save_segment_to_historic(
                image_path,
                segmented_image,
                historic_dir,
            )
            summary["processed"] += 1
            print(f"[SEGMENTED] {image_path.name} -> {output_path.name}")
        except Exception as exc:
            summary["skipped"] += 1
            print(f"[ERROR] Skipping {image_path}: {exc}")
    return summary


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    historic_dir = args.historic_dir.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {input_dir}")

    if args.mode == "segment":
        model = load_segmentation_model(args.model.resolve())
        summary = process_segment_images(
            input_dir=input_dir,
            model=model,
            historic_dir=historic_dir,
            confidence=float(args.conf),
            device=args.device,
        )
        print(
            "Done. "
            f"processed={summary['processed']} "
            f"skipped={summary['skipped']}"
        )
    else:
        models_dir = args.models_dir.resolve()
        models_by_position = load_models(models_dir)
        if not any(models_by_position.values()):
            print("[WARN] No position-matched models loaded; all images will be saved as OK.")

        summary = process_images(
            input_dir=input_dir,
            models_by_position=models_by_position,
            historic_dir=historic_dir,
            confidence=float(args.conf),
            device=args.device,
        )
        print(
            "Done. "
            f"processed={summary['processed']} "
            f"ok={summary['ok']} "
            f"nok={summary['nok']} "
            f"skipped={summary['skipped']}"
        )
    return 0 if summary["skipped"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

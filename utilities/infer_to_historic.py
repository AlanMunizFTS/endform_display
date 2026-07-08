"""Run YOLO inference on a folder and save results into historic.

This utility intentionally does not write to the database. The display app will
pick up saved ``*_OK`` and ``*_NOK`` files from ``tmp_display/historic`` using
its existing bootstrap flow.
"""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
POSITIONS = ("side", "front", "diag")
DEFAULT_CONFIDENCE = 0.33
DEFAULT_HISTORIC_DIR = Path("tmp_display") / "historic"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on images and save _OK/_NOK outputs to historic."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing images to process")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Folder containing YOLO .pt models. Default: %(default)s",
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
    return parser.parse_args()


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


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    models_dir = args.models_dir.resolve()
    historic_dir = args.historic_dir.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {input_dir}")

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

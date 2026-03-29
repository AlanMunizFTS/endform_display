"""
classifier.py
Watches test_images/ for new segmented images, runs all matching YOLO models
for each position, writes display-compatible outputs, and mirrors model
results/defects into the local PostgreSQL schema.

Images are processed JSN by JSN. After each piece, the script waits
PIECE_DISPLAY_DURATION seconds before clearing tmp_display/ and showing
the next piece.

Run from the project root:
    python utilities/classifier.py
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db import get_db_connection

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
TEST_IMAGES_DIR = Path("./test_images")
TMP_DISPLAY_DIR = Path("./tmp_display")
HISTORIC_DIR = TMP_DISPLAY_DIR / "historic"
ANNOTATED_DIR = TMP_DISPLAY_DIR / "annotated"
MODELS_FOLDER = Path("./models")

# ---------------------------------------------------------------------------
# Config - adjust these as needed
# ---------------------------------------------------------------------------
CONFIDENCE_THR = 0.33
# Zero removes the hold between pieces so the next batch is shown immediately.
PIECE_DISPLAY_DURATION = 0.0
# Keep a tiny poll delay to avoid pegging a CPU core while still reacting quickly.
POLL_INTERVAL = 0.02
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
POSITIONS = ["side", "front", "diag"]
LOCAL_MODEL_RESULTS_TABLE = "model_results_local"


def load_models(models_folder: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Return {position: [{"name": filename, "model": YOLO(...)}]}."""
    models: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if not models_folder.is_dir():
        print(f"[WARN] Models folder not found: {models_folder}")
        return models

    for path in sorted(models_folder.iterdir()):
        if path.suffix.lower() != ".pt":
            continue

        lower_name = path.name.lower()
        for pos in POSITIONS:
            if pos in lower_name:
                models[pos].append({"name": path.name, "model": YOLO(str(path))})
                print(f"[MODEL] Loaded '{path.name}' -> position '{pos}'")
                break

    for pos, entries in models.items():
        print(f"[MODEL] {pos}: {len(entries)} model(s)")

    return models


def get_position(filename: str) -> Optional[str]:
    lower = filename.lower()
    for pos in POSITIONS:
        if pos in lower:
            return pos
    return None


def strip_trailing_status(filename: str) -> str:
    """Remove a trailing _ok/_nok from the basename, case-insensitively."""
    basename, ext = os.path.splitext(filename)
    lower_basename = basename.lower()
    if lower_basename.endswith("_nok"):
        basename = basename[:-4]
    elif lower_basename.endswith("_ok"):
        basename = basename[:-3]
    return f"{basename}{ext}"


def get_jsn(filename: str) -> str:
    """Extract JSN prefix (everything before the first '_')."""
    return filename.split("_")[0] if "_" in filename else os.path.splitext(filename)[0]


def build_processed_set() -> set[str]:
    """Scan historic/ once at startup; return set of original basenames already processed."""
    processed: set[str] = set()
    if not HISTORIC_DIR.is_dir():
        return processed

    for fname in os.listdir(HISTORIC_DIR):
        base = os.path.splitext(fname)[0]
        if base.endswith("_OK"):
            processed.add(base[:-3])
        elif base.endswith("_NOK"):
            processed.add(base[:-4])
    return processed


def already_processed(basename: str, processed_set: set[str]) -> bool:
    return basename in processed_set


def clear_tmp_display() -> None:
    """Remove image files from tmp_display/ root (not subdirectories)."""
    if not TMP_DISPLAY_DIR.exists():
        return

    for fname in os.listdir(TMP_DISPLAY_DIR):
        if Path(fname).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            (TMP_DISPLAY_DIR / fname).unlink()
        except Exception as exc:
            print(f"  [WARN] Could not remove {fname}: {exc}")


def get_display_defect_name(model_name: str, defect_name: str) -> str:
    normalized_model_name = Path(model_name).stem.lower()
    normalized_defect_name = defect_name.lower()

    if normalized_defect_name == "nok":
        if "streaked" in normalized_model_name:
            return "streaked"

        normalized_model_name = normalized_model_name.replace("best_", "")
        return normalized_model_name

    return defect_name


def get_defect_info(
    result: Any,
    confidence_threshold: float,
    model_name: str = "",
) -> List[Dict[str, Any]]:
    """Extract all defects above threshold from a YOLO result."""
    defects: List[Dict[str, Any]] = []

    if result is None:
        return defects

    try:
        if hasattr(result, "obb") and result.obb is not None:
            if result.obb.conf is None or len(result.obb.conf) == 0:
                return defects

            for i, conf in enumerate(result.obb.conf):
                conf_value = float(conf)
                if conf_value <= confidence_threshold:
                    continue
                class_id = int(result.obb.cls[i])
                defects.append(
                    {
                        "defect_name": get_display_defect_name(model_name, result.names[class_id]),
                        "confidence": conf_value,
                    }
                )
            return defects

        if hasattr(result, "boxes") and result.boxes is not None:
            if result.boxes.conf is None or len(result.boxes.conf) == 0:
                return defects

            for i, conf in enumerate(result.boxes.conf):
                conf_value = float(conf)
                if conf_value <= confidence_threshold:
                    continue
                class_id = int(result.boxes.cls[i])
                defects.append(
                    {
                        "defect_name": get_display_defect_name(model_name, result.names[class_id]),
                        "confidence": conf_value,
                    }
                )
            return defects

        if hasattr(result, "probs") and result.probs is not None:
            if result.probs.data is None or len(result.probs.data) == 0:
                return defects

            top_class_id = int(result.probs.top1)
            top_conf = float(result.probs.top1conf)
            class_name = result.names[top_class_id].lower()
            if class_name == "nok" and top_conf > confidence_threshold:
                defects.append(
                    {
                        "defect_name": get_display_defect_name(model_name, class_name),
                        "confidence": top_conf,
                    }
                )
            return defects
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        print(f"  [WARN] Could not extract defects from model '{model_name}': {exc}")

    return defects


def draw_text_block(image: Any, lines: List[str]) -> Any:
    if not lines:
        return image

    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    padding = 10
    line_gap = 8

    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_width = max(width for width, _ in text_sizes)
    line_height = max(height for _, height in text_sizes)
    block_height = (line_height * len(lines)) + (line_gap * (len(lines) - 1)) + (padding * 2)
    block_width = max_width + (padding * 2)

    cv2.rectangle(annotated, (10, 10), (10 + block_width, 10 + block_height), (0, 0, 0), -1)

    y = 10 + padding + line_height
    for line in lines:
        cv2.putText(
            annotated,
            line,
            (10 + padding, y),
            font,
            font_scale,
            (0, 0, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_height + line_gap

    return annotated


def draw_detected_defects(
    image: Any,
    detected_results: List[Dict[str, Any]],
    confidence_threshold: float,
) -> Any:
    annotated = image.copy()
    classification_lines: List[str] = []

    for detected_result in detected_results:
        model_name = detected_result["model_name"]
        result = detected_result["result"]

        if result is None:
            continue

        try:
            if hasattr(result, "obb") and result.obb is not None and result.obb.conf is not None:
                for i, conf in enumerate(result.obb.conf):
                    conf_value = float(conf)
                    if conf_value <= confidence_threshold:
                        continue

                    class_id = int(result.obb.cls[i])
                    defect_name = get_display_defect_name(model_name, result.names[class_id])
                    points = result.obb.xyxyxyxy[i].cpu().numpy().astype("int32").reshape((-1, 1, 2))
                    cv2.polylines(annotated, [points], True, (0, 0, 255), 2)
                    x, y = points[0][0]
                    cv2.putText(
                        annotated,
                        f"{defect_name} {conf_value:.2f}",
                        (int(x), max(20, int(y) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                continue

            if hasattr(result, "boxes") and result.boxes is not None and result.boxes.conf is not None:
                for i, conf in enumerate(result.boxes.conf):
                    conf_value = float(conf)
                    if conf_value <= confidence_threshold:
                        continue

                    class_id = int(result.boxes.cls[i])
                    defect_name = get_display_defect_name(model_name, result.names[class_id])
                    x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().astype("int32")
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        annotated,
                        f"{defect_name} {conf_value:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                continue

            if hasattr(result, "probs") and result.probs is not None and result.probs.data is not None:
                top_class_id = int(result.probs.top1)
                top_conf = float(result.probs.top1conf)
                class_name = result.names[top_class_id]
                if top_conf > confidence_threshold and class_name.lower() == "nok":
                    defect_name = get_display_defect_name(model_name, class_name)
                    classification_lines.append(f"{defect_name} {top_conf:.2f}")
        except Exception as exc:
            print(f"  [WARN] Could not draw detections for {model_name}: {exc}")

    return draw_text_block(annotated, classification_lines)


def run_inference(
    image: Any,
    model_entries: List[Dict[str, Any]],
    confidence_threshold: float,
) -> tuple[bool, List[Dict[str, Any]], List[Dict[str, Any]]]:
    detected_defects: List[Dict[str, Any]] = []
    detected_results: List[Dict[str, Any]] = []

    if not model_entries:
        return False, detected_defects, detected_results

    for entry in model_entries:
        model_name = entry["name"]
        model = entry["model"]

        try:
            results = model(image, verbose=False, conf=confidence_threshold)
            classification_result = results[0] if isinstance(results, list) else results
        except Exception as exc:
            print(f"  [WARN] Inference failed for model '{model_name}': {exc}")
            continue

        model_defects = get_defect_info(
            classification_result,
            confidence_threshold=confidence_threshold,
            model_name=model_name,
        )

        if not model_defects:
            continue

        detected_results.append({"model_name": model_name, "result": classification_result})
        for defect in model_defects:
            detected_defects.append(
                {
                    "model_name": model_name,
                    "defect_name": defect["defect_name"],
                    "confidence": defect["confidence"],
                }
            )

    return len(detected_defects) > 0, detected_defects, detected_results


def get_db_client() -> Optional[Any]:
    try:
        db_client = get_db_connection()
        print("[DB] Connected to local PostgreSQL")
        return db_client
    except Exception as exc:
        print(f"[WARN] DB unavailable, continuing without DB sync: {exc}")
        return None


def queue_model_results(
    db_client: Any,
    img_name: str,
    detected_defects: List[Dict[str, Any]],
) -> None:
    if db_client is None:
        return

    staged_rows = detected_defects or [
        {
            "defect_name": "OK",
            "confidence": 1.0,
        }
    ]

    for defect in staged_rows:
        class_name = str(defect.get("defect_name") or "").strip()
        confidence = round(float(defect.get("confidence") or 0.0), 4)
        if not class_name:
            continue

        db_client.execute(
            f"INSERT INTO {LOCAL_MODEL_RESULTS_TABLE} "
            "(img_name, class_name, confidence) "
            "VALUES (%s, %s, %s)",
            (img_name, class_name, confidence),
        )


def ensure_local_model_results_table(db_client: Optional[Any]) -> None:
    if db_client is None:
        return

    db_client.execute(
        f"CREATE TABLE IF NOT EXISTS {LOCAL_MODEL_RESULTS_TABLE} ("
        "id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
        "img_name TEXT NOT NULL, "
        "class_name TEXT NOT NULL, "
        "confidence DECIMAL(5,4) NOT NULL, "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ")"
    )
    db_client.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{LOCAL_MODEL_RESULTS_TABLE}_img_name "
        f"ON {LOCAL_MODEL_RESULTS_TABLE} (img_name)"
    )
    db_client.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{LOCAL_MODEL_RESULTS_TABLE}_created_at "
        f"ON {LOCAL_MODEL_RESULTS_TABLE} (created_at)"
    )


def save_image(path: Path, image: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"cv2.imwrite failed for {path}")


def main() -> None:
    TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DISPLAY_DIR.mkdir(parents=True, exist_ok=True)
    HISTORIC_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    models = load_models(MODELS_FOLDER)
    if not any(models.values()):
        print("[ERROR] No models loaded. Make sure ./models/ contains .pt files with 'side'/'front'/'diag' in their name.")
        return

    db_client = get_db_client()
    ensure_local_model_results_table(db_client)
    processed_set = build_processed_set()

    print(f"[INFO] Watching '{TEST_IMAGES_DIR}' - polling every {POLL_INTERVAL}s")
    print(f"[INFO] Confidence threshold: {CONFIDENCE_THR}")
    print(f"[INFO] Piece display duration: {PIECE_DISPLAY_DURATION}s")
    print(f"[INFO] Already processed (from historic/): {len(processed_set)} image(s)")

    try:
        while True:
            try:
                jsn_groups: Dict[str, List[str]] = defaultdict(list)
                for filename in os.listdir(TEST_IMAGES_DIR):
                    ext = Path(filename).suffix.lower()
                    if ext not in IMAGE_EXTENSIONS:
                        continue

                    normalized_filename = strip_trailing_status(filename)
                    basename = Path(normalized_filename).stem
                    if already_processed(basename, processed_set):
                        continue

                    jsn_groups[get_jsn(normalized_filename)].append(filename)

                if not jsn_groups:
                    if POLL_INTERVAL > 0:
                        time.sleep(POLL_INTERVAL)
                    continue

                for jsn in sorted(jsn_groups.keys()):
                    files = sorted(jsn_groups[jsn])
                    print(f"\n[JSN] {jsn}  ({len(files)} images)")

                    display_batch: List[Dict[str, Any]] = []
                    historic_batch: List[Dict[str, Any]] = []

                    for filename in files:
                        ext = Path(filename).suffix.lower()
                        normalized_filename = strip_trailing_status(filename)
                        basename = Path(normalized_filename).stem
                        position = get_position(normalized_filename)
                        model_entries = models.get(position or "", [])
                        image_path = TEST_IMAGES_DIR / filename
                        original_image = cv2.imread(str(image_path))

                        if original_image is None:
                            print(f"  [WARN] Could not read image: {image_path}")
                            continue

                        parts = basename.split("_", 1)
                        display_basename = parts[1] if len(parts) > 1 else basename
                        status_reason = ""

                        if not model_entries:
                            if position == "front":
                                status_reason = "front fallback: no model"
                            else:
                                status_reason = f"no model for '{position}'"
                            has_detection = False
                            detected_defects = []
                            detected_results = []
                        else:
                            has_detection, detected_defects, detected_results = run_inference(
                                original_image,
                                model_entries,
                                CONFIDENCE_THR,
                            )

                        status = "NOK" if has_detection else "OK"
                        out_name = f"{display_basename}_{status}{ext}"
                        hist_name = f"{basename}_{status}{ext}"
                        annotated_image = (
                            draw_detected_defects(original_image, detected_results, CONFIDENCE_THR)
                            if has_detection
                            else original_image
                        )

                        defect_summary = ", ".join(
                            f"{defect['defect_name']} {float(defect['confidence']):.2f}"
                            for defect in detected_defects
                        )
                        if defect_summary:
                            print(f"  {filename} -> {status} [{defect_summary}]")
                        elif status_reason:
                            print(f"  {filename} -> {status} ({status_reason})")
                        else:
                            print(f"  {filename} -> {status}")

                        try:
                            queue_model_results(
                                db_client=db_client,
                                img_name=hist_name,
                                detected_defects=detected_defects,
                            )
                        except Exception as exc:
                            print(f"  [WARN] Could not queue local model results for {hist_name}: {exc}")

                        display_batch.append(
                            {
                                "name": out_name,
                                "image": annotated_image,
                            }
                        )
                        historic_batch.append(
                            {
                                "name": hist_name,
                                "original": original_image,
                                "annotated": annotated_image,
                            }
                        )

                    clear_tmp_display()
                    for entry in display_batch:
                        save_image(TMP_DISPLAY_DIR / entry["name"], entry["image"])

                    for entry in historic_batch:
                        save_image(HISTORIC_DIR / entry["name"], entry["original"])
                        save_image(ANNOTATED_DIR / entry["name"], entry["annotated"])
                        original_basename = Path(entry["name"]).stem
                        if original_basename.endswith("_OK"):
                            processed_set.add(original_basename[:-3])
                        elif original_basename.endswith("_NOK"):
                            processed_set.add(original_basename[:-4])
                        else:
                            processed_set.add(original_basename)

                    if PIECE_DISPLAY_DURATION > 0:
                        time.sleep(PIECE_DISPLAY_DURATION)

            except Exception as exc:
                print(f"[ERROR] Unexpected error: {exc}")
                if POLL_INTERVAL > 0:
                    time.sleep(POLL_INTERVAL)
    finally:
        if db_client is not None:
            try:
                db_client.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()

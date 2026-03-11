"""
classifier.py
Watches test_images/ for new images, classifies them with YOLO models,
and injects results into tmp_display/ (normal view), tmp_display/historic/
(historic view), and tmp_display/annotated/ (annotated, NOK only).

Run from the project root:
    python utilities/classifier.py
"""

import os
import shutil
import time

import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
TEST_IMAGES_DIR = "./test_images"
TMP_DISPLAY_DIR = "./tmp_display"
HISTORIC_DIR    = "./tmp_display/historic"
ANNOTATED_DIR   = "./tmp_display/annotated"
MODELS_FOLDER   = "./models"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIDENCE_THR   = 0.33
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
POLL_INTERVAL    = 1.0   # seconds between directory scans
POSITIONS        = ["side", "front", "diag"]


# ---------------------------------------------------------------------------
# Load models: map position → YOLO model
# ---------------------------------------------------------------------------
def load_models(models_folder):
    models = {}
    if not os.path.isdir(models_folder):
        print(f"[WARN] Models folder not found: {models_folder}")
        return models

    for fname in os.listdir(models_folder):
        if not fname.endswith(".pt"):
            continue
        lower = fname.lower()
        for pos in POSITIONS:
            if pos in lower:
                path = os.path.join(models_folder, fname)
                models[pos] = YOLO(path)
                print(f"[MODEL] Loaded '{fname}' → position '{pos}'")
                break

    return models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_position(filename):
    lower = filename.lower()
    for pos in POSITIONS:
        if pos in lower:
            return pos
    return None


def already_processed(basename, ext):
    """True if either _OK or _NOK output already exists in tmp_display/."""
    for status in ("OK", "NOK"):
        if os.path.exists(os.path.join(TMP_DISPLAY_DIR, f"{basename}_{status}{ext}")):
            return True
    return False


def has_high_confidence_detection(result, confidence_threshold):
    if result is None:
        return False
    try:
        # OBB model
        if hasattr(result, "obb") and result.obb is not None:
            if result.obb.conf is None or len(result.obb.conf) == 0:
                return False
            return any(float(conf) > confidence_threshold for conf in result.obb.conf)

        # Segmentation model
        if hasattr(result, "masks") and result.masks is not None:
            if (
                result.boxes is None
                or result.boxes.conf is None
                or len(result.boxes.conf) == 0
                or result.masks.xyn is None
                or len(result.masks.xyn) == 0
            ):
                return False
            return any(float(conf) > confidence_threshold for conf in result.boxes.conf)

        return False

    except AttributeError:
        return False


def get_result(image_path, models_list, confidence):
    """
    Try each model in order. Return (result, True) on first detection,
    or (None, False) if none detects above threshold.
    """
    for model in models_list:
        results = model(image_path, verbose=False, conf=confidence)
        classification_result = results[0] if isinstance(results, list) else results
        has_detection = has_high_confidence_detection(classification_result, confidence)
        if has_detection:
            return (classification_result, True)
    return (None, False)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    os.makedirs(TMP_DISPLAY_DIR, exist_ok=True)
    os.makedirs(HISTORIC_DIR, exist_ok=True)
    os.makedirs(ANNOTATED_DIR, exist_ok=True)

    models = load_models(MODELS_FOLDER)
    if not models:
        print("[ERROR] No models loaded. Make sure ./models/ contains .pt files with 'side'/'front'/'diag' in their name.")
        return

    print(f"[INFO] Watching '{TEST_IMAGES_DIR}' — polling every {POLL_INTERVAL}s")
    print(f"[INFO] Confidence threshold: {CONFIDENCE_THR}")

    while True:
        try:
            for filename in os.listdir(TEST_IMAGES_DIR):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    continue

                basename = os.path.splitext(filename)[0]

                if already_processed(basename, ext):
                    continue

                position = get_position(filename)
                model = models.get(position)
                if model is None:
                    print(f"[SKIP] No model for position of '{filename}'")
                    continue

                image_path = os.path.join(TEST_IMAGES_DIR, filename)

                print(f"[CLASSIFY] {filename} (position={position})")
                classification_result, has_detection = get_result(
                    image_path, [model], CONFIDENCE_THR
                )

                status = "NOK" if has_detection else "OK"
                out_name = f"{basename}_{status}{ext}"
                print(f"  → {status}  ({out_name})")

                # Original → tmp_display/ (normal view)
                shutil.copy2(image_path, os.path.join(TMP_DISPLAY_DIR, out_name))

                # Original → tmp_display/historic/ (historic view)
                shutil.copy2(image_path, os.path.join(HISTORIC_DIR, out_name))

                # Annotated → tmp_display/annotated/ (only if NOK)
                if classification_result is not None and has_detection:
                    try:
                        annotated_image = classification_result.plot()
                        cv2.imwrite(os.path.join(ANNOTATED_DIR, out_name), annotated_image)
                    except Exception as e:
                        print(f"  [WARN] plot() failed for {filename}: {e}")

        except Exception as e:
            print(f"[ERROR] Unexpected error in poll loop: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

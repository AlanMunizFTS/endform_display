"""
classifier.py
Watches test_images/ for new images, classifies them with YOLO models,
and injects results into tmp_display/ (normal view), tmp_display/historic/
(historic view).

Images are processed JSN by JSN. After each piece, the script waits
PIECE_DISPLAY_DURATION seconds before clearing tmp_display/ and showing
the next piece.

Run from the project root:
    python utilities/classifier.py
"""

import os
import shutil
import time
from collections import defaultdict

import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
TEST_IMAGES_DIR = "./test_images"
TMP_DISPLAY_DIR = "./tmp_display"
HISTORIC_DIR    = "./tmp_display/historic"
MODELS_FOLDER   = "./models"

# ---------------------------------------------------------------------------
# Config — adjust these as needed
# ---------------------------------------------------------------------------
CONFIDENCE_THR        = 0.33
PIECE_DISPLAY_DURATION = 1.0   # seconds each JSN group stays visible in normal view
POLL_INTERVAL         = 0.5   # seconds between scans when no new images are found
IMAGE_EXTENSIONS      = {".jpg", ".jpeg", ".png", ".bmp"}
POSITIONS             = ["side", "front", "diag"]


# ---------------------------------------------------------------------------
# Load models: map position → YOLO model
# ---------------------------------------------------------------------------
def load_models(models_folder):
    """Returns Dict[position, List[YOLO]] — all models per position."""
    models = defaultdict(list)
    if not os.path.isdir(models_folder):
        print(f"[WARN] Models folder not found: {models_folder}")
        return models

    for fname in sorted(os.listdir(models_folder)):
        if not fname.endswith(".pt"):
            continue
        lower = fname.lower()
        for pos in POSITIONS:
            if pos in lower:
                path = os.path.join(models_folder, fname)
                models[pos].append(YOLO(path))
                print(f"[MODEL] Loaded '{fname}' → position '{pos}'")
                break

    for pos, lst in models.items():
        print(f"[MODEL] {pos}: {len(lst)} model(s)")

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


def strip_trailing_status(filename):
    """
    Remove a trailing _ok or _nok from the basename, case-insensitively.
    If no trailing status is present, return the original filename.
    """
    basename, ext = os.path.splitext(filename)
    lower_basename = basename.lower()
    if lower_basename.endswith("_nok"):
        basename = basename[:-4]
    elif lower_basename.endswith("_ok"):
        basename = basename[:-3]
    return f"{basename}{ext}"


def get_jsn(filename):
    """Extract JSN prefix (everything before the first '_')."""
    return filename.split("_")[0] if "_" in filename else os.path.splitext(filename)[0]


def build_processed_set():
    """Scan historic/ once at startup; return set of original basenames already processed."""
    processed = set()
    if not os.path.isdir(HISTORIC_DIR):
        return processed
    for fname in os.listdir(HISTORIC_DIR):
        base = os.path.splitext(fname)[0]
        if base.endswith("_OK"):
            processed.add(base[:-3])
        elif base.endswith("_NOK"):
            processed.add(base[:-4])
    return processed


def already_processed(basename, processed_set):
    """True if basename is in the in-memory processed set."""
    return basename in processed_set


def clear_tmp_display():
    """Remove image files from tmp_display/ root (not subdirectories)."""
    for fname in os.listdir(TMP_DISPLAY_DIR):
        if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
            try:
                os.remove(os.path.join(TMP_DISPLAY_DIR, fname))
            except Exception as e:
                print(f"  [WARN] Could not remove {fname}: {e}")


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

    models = load_models(MODELS_FOLDER)
    if not models:
        print("[ERROR] No models loaded. Make sure ./models/ contains .pt files with 'side'/'front'/'diag' in their name.")
        return

    processed_set = build_processed_set()
    print(f"[INFO] Watching '{TEST_IMAGES_DIR}' — polling every {POLL_INTERVAL}s")
    print(f"[INFO] Confidence threshold: {CONFIDENCE_THR}")
    print(f"[INFO] Piece display duration: {PIECE_DISPLAY_DURATION}s")
    print(f"[INFO] Already processed (from historic/): {len(processed_set)} image(s)")

    while True:
        try:
            # --- Scan and group unprocessed images by JSN ---
            jsn_groups = defaultdict(list)
            for filename in os.listdir(TEST_IMAGES_DIR):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    continue
                normalized_filename = strip_trailing_status(filename)
                basename = os.path.splitext(normalized_filename)[0]
                if already_processed(basename, processed_set):
                    continue
                jsn_groups[get_jsn(normalized_filename)].append(filename)

            if not jsn_groups:
                time.sleep(POLL_INTERVAL)
                continue

            # --- Process one JSN group at a time ---
            for jsn in sorted(jsn_groups.keys()):
                files = sorted(jsn_groups[jsn])
                print(f"\n[JSN] {jsn}  ({len(files)} images)")

                # --- Classify all images first, collect outputs in memory ---
                display_batch = []  # (out_name, annotated_array_or_None, src_path)
                historic_batch = []  # (hist_name, annotated_array_or_None, src_path)

                for filename in files:
                    ext = os.path.splitext(filename)[1].lower()
                    normalized_filename = strip_trailing_status(filename)
                    basename = os.path.splitext(normalized_filename)[0]

                    position = get_position(filename)
                    models_list = models.get(position, [])
                    image_path = os.path.join(TEST_IMAGES_DIR, filename)

                    # Strip JSN prefix for output filename (JSN_angulo → angulo)
                    _parts = basename.split("_", 1)
                    display_basename = _parts[1] if len(_parts) > 1 else basename

                    if not models_list:
                        out_name  = f"{display_basename}_OK{ext}"
                        hist_name = f"{basename}_OK{ext}"
                        print(f"  {filename} → OK (no model for '{position}')")
                        display_batch.append((out_name, None, image_path))
                        historic_batch.append((hist_name, None, image_path))
                        processed_set.add(basename)
                        continue

                    classification_result, has_detection = get_result(
                        image_path, models_list, CONFIDENCE_THR
                    )

                    status = "NOK" if has_detection else "OK"
                    out_name  = f"{display_basename}_{status}{ext}"
                    hist_name = f"{basename}_{status}{ext}"
                    print(f"  {filename} → {status}")

                    annotated_image = None
                    if classification_result is not None and has_detection:
                        try:
                            annotated_image = classification_result.plot()
                        except Exception as e:
                            print(f"  [WARN] plot() failed for {filename}: {e}")

                    display_batch.append((out_name, annotated_image, image_path))
                    historic_batch.append((hist_name, annotated_image, image_path))
                    processed_set.add(basename)

                # --- Write all results to tmp_display/ in one shot ---
                clear_tmp_display()
                for out_name, annotated_image, src_path in display_batch:
                    if annotated_image is not None:
                        cv2.imwrite(os.path.join(TMP_DISPLAY_DIR, out_name), annotated_image)
                    else:
                        shutil.copy2(src_path, os.path.join(TMP_DISPLAY_DIR, out_name))

                for hist_name, annotated_image, src_path in historic_batch:
                    is_nok = hist_name.rsplit("_", 1)[-1].startswith("NOK")
                    historic_path = os.path.join(HISTORIC_DIR, hist_name)
                    if is_nok and annotated_image is not None:
                        cv2.imwrite(historic_path, annotated_image)
                    else:
                        shutil.copy2(src_path, historic_path)

                # Hold this JSN visible for PIECE_DISPLAY_DURATION seconds
                time.sleep(PIECE_DISPLAY_DURATION)

        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

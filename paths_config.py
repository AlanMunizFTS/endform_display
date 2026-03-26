from pathlib import Path

# ----------------------------
# Local folders and file paths
# ----------------------------
TMP_DISPLAY_DIR = Path("./tmp_display")
HISTORIC_SUBDIR_NAME = "historic"
HISTORIC_LOCAL_DIR = TMP_DISPLAY_DIR / HISTORIC_SUBDIR_NAME
ANNOTATED_SUBDIR_NAME = "annotated"
ANNOTATED_LOCAL_DIR = TMP_DISPLAY_DIR / ANNOTATED_SUBDIR_NAME

RESOURCES_DIR = Path("./resources")
CAMERA_ICON_PATH = RESOURCES_DIR / "camara.png"
TRASH_ICON_PATH = RESOURCES_DIR / "trash.png"
BASE_SCREEN_PATH = RESOURCES_DIR / "base_screen.png"

LOG_FILE_PATH = Path("log.txt")

# Base directory used by sync_images_by_status for status output folders
SYNC_IMAGES_BASE_DIR = Path("./classified")

# ----------------------------
# Remote folders
# ----------------------------
REMOTE_TEST_DISPLAY_DIR = "/media/ssd/test_display"
REMOTE_HIST_DISPLAY_DIR = "/media/ssd/hist_display"
REMOTE_ANNOTATED_DIR = "/media/ssd/annotated"

# --------------------------------------
# Status folders used by sync operations
# --------------------------------------
STATUS_SYNC_DIRS = {
    "side": {"OK": "side_ok", "NOK": "side_nok"},
    "front": {"OK": "front_ok", "NOK": "front_nok"},
    "diag": {"OK": "diag_ok", "NOK": "diag_nok"},
}

# --------------------------------------------------
# Final classification folders (model vs operator)
# --------------------------------------------------
FINAL_CLASSIFICATION_DIR = Path("./final_classification")
REPORTS_DIR = Path("./reports")
EXPORTS_DIR = Path("./exports")
STATE_PACKAGE_VERSION = 1

FINAL_CLASSIFICATION_DIRS = {
    "side": {"P": "Side_P", "N": "Side_N", "FP": "Side_FP", "FN": "Side_FN"},
    "front": {"P": "Front_P", "N": "Front_N", "FP": "Front_FP", "FN": "Front_FN"},
    "diag": {"P": "Diag_P", "N": "Diag_N", "FP": "Diag_FP", "FN": "Diag_FN"},
}

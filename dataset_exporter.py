import re
from datetime import datetime
from pathlib import Path

from paths_config import DATASETS_DIR


DEFAULT_RESULT_OPTIONS = ("OK", "NOK", "FOK", "FNOK")
DEFAULT_ANGLE_OPTIONS = ("side", "diag", "front")
ALL_CLASSES_LABEL = "All"


def sanitize_dataset_folder_name(label):
    text = str(label or "").strip() or "UNCLASSIFIED"
    text = re.sub(r"[<>:\"/\\\\|?*\x00-\x1F]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._ ")
    return text or "UNCLASSIFIED"


def _normalize_filter_values(values, normalizer=str):
    normalized = []
    seen = set()
    for value in values or []:
        item = normalizer(value)
        if not item or item in seen:
            continue
        normalized.append(item)
        seen.add(item)
    return normalized


def export_piece_stats_dataset(
    controller,
    filters=None,
    output_dir=None,
    progress_callback=None,
    db_client=None,
    historic_dir=None,
):
    db = db_client or getattr(controller.display, "db", None)
    if db is None:
        return {"ok": False, "error": "No database connection available"}

    file_manager = controller.file_manager
    records = controller.get_piece_stats_dataset_records(db_client=db)
    if not records:
        return {"ok": False, "error": "No image dataset rows available"}

    filter_data = dict(filters or {})
    selected_results = _normalize_filter_values(
        filter_data.get("results") or DEFAULT_RESULT_OPTIONS,
        lambda value: str(value or "").strip().upper(),
    )
    selected_angles = _normalize_filter_values(
        filter_data.get("angles") or DEFAULT_ANGLE_OPTIONS,
        lambda value: str(value or "").strip().lower(),
    )
    selected_classes = _normalize_filter_values(
        filter_data.get("class_names") or [ALL_CLASSES_LABEL],
        lambda value: str(value or "").strip(),
    )

    include_all_classes = not selected_classes or ALL_CLASSES_LABEL in selected_classes
    selected_class_set = set(selected_classes)
    selected_result_set = set(selected_results)
    selected_angle_set = set(selected_angles)

    export_entries = []
    matched_image_names = set()
    for record in records:
        result_value = str(record.get("result") or "").strip().upper()
        angle_value = str(record.get("angle") or "").strip().lower()
        if result_value not in selected_result_set or angle_value not in selected_angle_set:
            continue

        class_names = list(record.get("class_names") or ["UNCLASSIFIED"])
        if include_all_classes:
            matching_classes = class_names
        else:
            matching_classes = [
                class_name for class_name in class_names if class_name in selected_class_set
            ]

        if not matching_classes:
            continue

        img_name = str(record.get("img_name") or "").strip()
        if not img_name:
            continue

        matched_image_names.add(img_name)
        for class_name in matching_classes:
            export_entries.append(
                {
                    "img_name": img_name,
                    "result": result_value,
                    "angle": angle_value,
                    "class_name": class_name,
                }
            )

    if not export_entries:
        return {"ok": False, "error": "No images match the selected dataset filters"}

    historic_source_dir = historic_dir or controller._get_export_historic_dir()
    output_root = Path(output_dir or DATASETS_DIR)
    dataset_name = f"dataset_{datetime.now():%Y%m%d_%H%M%S}"
    dataset_dir = output_root / dataset_name
    file_manager.makedirs(str(dataset_dir), exist_ok=True)

    copied_count = 0
    missing_names = set()
    total_entries = len(export_entries)
    if callable(progress_callback):
        progress_callback(0, total_entries, "Exporting filtered dataset")

    for idx, entry in enumerate(export_entries, start=1):
        img_name = entry["img_name"]
        source_path = file_manager.join(historic_source_dir, img_name)
        if not file_manager.exists(source_path):
            missing_names.add(img_name)
            if callable(progress_callback):
                progress_callback(idx, total_entries, "Exporting filtered dataset")
            continue

        target_dir = dataset_dir / entry["result"] / sanitize_dataset_folder_name(
            entry["class_name"]
        ) / entry["angle"]
        file_manager.makedirs(str(target_dir), exist_ok=True)
        file_manager.copy2(source_path, str(target_dir / img_name))
        copied_count += 1
        if callable(progress_callback):
            progress_callback(idx, total_entries, "Exporting filtered dataset")

    return {
        "ok": True,
        "dataset_name": dataset_name,
        "output_path": str(dataset_dir),
        "copied_files": copied_count,
        "matched_images": len(matched_image_names),
        "matched_variants": total_entries,
        "missing_files": sorted(missing_names),
        "missing_count": len(missing_names),
        "filters": {
            "results": selected_results,
            "angles": selected_angles,
            "class_names": selected_classes or [ALL_CLASSES_LABEL],
        },
    }

import json
import os
import socket
from datetime import datetime
from decimal import Decimal

try:
    from psycopg2.extras import Json
except Exception:  # pragma: no cover - psycopg2 is available in the app runtime.
    Json = None

from paths_config import EXPORTS_DIR, STATE_PACKAGE_VERSION


PACKAGE_KIND = "display_state"
REQUIRED_PACKAGE_FILES = (
    "manifest.json",
    "db/data.json",
    "db/database.sql",
)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Decimal):
        return format(value, "f")
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat(sep=" ")
        except TypeError:
            return value.isoformat()
    return value


def _sql_literal(value):
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float, Decimal)):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(_json_safe(value), sort_keys=True).replace("'", "''")
        return f"'{text}'"

    text = str(value).replace("'", "''")
    return f"'{text}'"


def _json_db_value(value):
    if value is None:
        return None
    if Json is None:
        return json.dumps(_json_safe(value), sort_keys=True)
    return Json(_json_safe(value))


def _canonical_json_key(value):
    if value is None:
        return None
    if hasattr(value, "adapted"):
        value = value.adapted
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return value
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))


def _model_result_key(row):
    return (
        row.get("img_name"),
        row.get("class_name"),
        str(row.get("confidence")),
        row.get("model_name"),
        row.get("geometry_type"),
        _canonical_json_key(row.get("coordinates")),
        row.get("image_width"),
        row.get("image_height"),
    )


def _list_image_names(file_manager, directory, image_extensions):
    if not file_manager.exists(directory):
        return []
    image_names = []
    for name in file_manager.listdir(directory):
        path = file_manager.join(directory, name)
        if not file_manager.is_file(path):
            continue
        if name.lower().endswith(tuple(ext.lower() for ext in image_extensions)):
            image_names.append(name)
    return sorted(image_names)


def _fetch_rows(db_client, query):
    return [_normalize_row(row) for row in (db_client.fetch(query) or [])]


def _normalize_row(row):
    return {key: _json_safe(value) for key, value in dict(row).items()}


def _build_data_payload(db_client):
    payload = {
        "img_results": _fetch_rows(
            db_client,
            "SELECT img_name, result FROM img_results ORDER BY img_name",
        ),
        "piece_result": _fetch_rows(
            db_client,
            "SELECT jsn, operator_result, model_result, created_at "
            "FROM piece_result ORDER BY jsn",
        ),
        "classified_images": _fetch_rows(
            db_client,
            "SELECT ci.img_name, pr.jsn, ci.operator_result, ci.model_result, ci.created_at "
            "FROM classified_images ci "
            "LEFT JOIN piece_result pr ON pr.id = ci.piece_id "
            "ORDER BY ci.img_name",
        ),
        "classified_image_defects": _fetch_rows(
            db_client,
            "SELECT ci.img_name, cid.class_name, cid.confidence, cid.created_at, "
            "cid.remote_model_result_id, cid.model_name, cid.geometry_type, "
            "cid.coordinates, cid.image_width, cid.image_height "
            "FROM classified_image_defects cid "
            "JOIN classified_images ci ON ci.id = cid.classified_image_id "
            "ORDER BY ci.img_name, cid.class_name, cid.confidence",
        ),
        "model_results": _fetch_rows(
            db_client,
            "SELECT img_name, class_name, confidence, created_at, model_name, "
            "geometry_type, coordinates, image_width, image_height "
            "FROM model_results "
            "ORDER BY img_name, class_name, confidence, created_at",
        ),
        "piece_result_defects": _fetch_rows(
            db_client,
            "SELECT pr.jsn, prd.class_name, prd.confidence, prd.created_at "
            "FROM piece_result_defects prd "
            "JOIN piece_result pr ON pr.id = prd.piece_result_id "
            "ORDER BY pr.jsn, prd.class_name, prd.confidence",
        ),
    }
    return payload


def _build_manifest(annotated_names, historic_names, data_payload, image_extensions):
    return {
        "package_version": STATE_PACKAGE_VERSION,
        "package_kind": PACKAGE_KIND,
        "export_complete": True,
        "created_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "source_host": socket.gethostname(),
        "annotated_count": len(annotated_names),
        "historic_count": len(historic_names),
        "annotated_images": list(annotated_names),
        "historic_images": list(historic_names),
        "table_counts": {
            table_name: len(rows)
            for table_name, rows in data_payload.items()
        },
        "image_extensions": list(image_extensions),
    }


def _build_database_sql(data_payload):
    lines = [
        "-- Display State Package v1",
        f"-- Generated at {datetime.now().isoformat(sep=' ', timespec='seconds')}",
        "BEGIN;",
        "",
    ]

    for row in data_payload.get("img_results", []):
        lines.append(
            "INSERT INTO img_results (img_name, result) "
            f"VALUES ({_sql_literal(row.get('img_name'))}, {_sql_literal(row.get('result'))}) "
            "ON CONFLICT DO NOTHING;"
        )

    lines.append("")

    for row in data_payload.get("piece_result", []):
        lines.append(
            "INSERT INTO piece_result (jsn, operator_result, model_result, created_at) "
            f"VALUES ({_sql_literal(row.get('jsn'))}, {_sql_literal(row.get('operator_result'))}, "
            f"{_sql_literal(row.get('model_result'))}, {_sql_literal(row.get('created_at'))}) "
            "ON CONFLICT (jsn) DO NOTHING;"
        )

    lines.append("")

    for row in data_payload.get("classified_images", []):
        lines.append(
            "INSERT INTO classified_images (img_name, operator_result, model_result, piece_id, created_at) "
            "SELECT "
            f"{_sql_literal(row.get('img_name'))}, {_sql_literal(row.get('operator_result'))}, "
            f"{_sql_literal(row.get('model_result'))}, pr.id, {_sql_literal(row.get('created_at'))} "
            "FROM piece_result pr "
            f"WHERE pr.jsn = {_sql_literal(row.get('jsn'))} "
            "ON CONFLICT (img_name) DO NOTHING;"
        )

    lines.append("")

    for row in data_payload.get("classified_image_defects", []):
        lines.append(
            "INSERT INTO classified_image_defects "
            "(classified_image_id, class_name, confidence, created_at, remote_model_result_id, "
            "model_name, geometry_type, coordinates, image_width, image_height) "
            "SELECT "
            f"ci.id, {_sql_literal(row.get('class_name'))}, {_sql_literal(row.get('confidence'))}, "
            f"{_sql_literal(row.get('created_at'))}, {_sql_literal(row.get('remote_model_result_id'))}, "
            f"{_sql_literal(row.get('model_name'))}, {_sql_literal(row.get('geometry_type'))}, "
            f"{_sql_literal(row.get('coordinates'))}, {_sql_literal(row.get('image_width'))}, "
            f"{_sql_literal(row.get('image_height'))} "
            "FROM classified_images ci "
            f"WHERE ci.img_name = {_sql_literal(row.get('img_name'))} "
            "ON CONFLICT (classified_image_id, class_name, confidence) DO NOTHING;"
        )

    lines.append("")

    for row in data_payload.get("model_results", []):
        lines.append(
            "INSERT INTO model_results "
            "(img_name, class_name, confidence, created_at, model_name, geometry_type, "
            "coordinates, image_width, image_height) "
            f"VALUES ({_sql_literal(row.get('img_name'))}, {_sql_literal(row.get('class_name'))}, "
            f"{_sql_literal(row.get('confidence'))}, {_sql_literal(row.get('created_at'))}, "
            f"{_sql_literal(row.get('model_name'))}, {_sql_literal(row.get('geometry_type'))}, "
            f"{_sql_literal(row.get('coordinates'))}, {_sql_literal(row.get('image_width'))}, "
            f"{_sql_literal(row.get('image_height'))}) "
            "ON CONFLICT DO NOTHING;"
        )

    lines.append("")

    for row in data_payload.get("piece_result_defects", []):
        lines.append(
            "INSERT INTO piece_result_defects (piece_result_id, class_name, confidence, created_at) "
            "SELECT "
            f"pr.id, {_sql_literal(row.get('class_name'))}, {_sql_literal(row.get('confidence'))}, "
            f"{_sql_literal(row.get('created_at'))} "
            "FROM piece_result pr "
            f"WHERE pr.jsn = {_sql_literal(row.get('jsn'))} "
            "ON CONFLICT (piece_result_id, class_name) DO NOTHING;"
        )

    lines.extend(["", "COMMIT;", ""])
    return "\n".join(lines)


def _validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("Invalid package manifest")
    if manifest.get("package_kind") != PACKAGE_KIND:
        raise ValueError("Unsupported package kind")
    if manifest.get("export_complete") is not True:
        raise ValueError("Export folder is incomplete")
    if int(manifest.get("package_version") or 0) != STATE_PACKAGE_VERSION:
        raise ValueError(
            f"Unsupported package version: {manifest.get('package_version')}"
        )


def _package_file_path(package_path, relative_path):
    return os.path.join(package_path, *relative_path.split("/"))


def _validate_manifest_image_entries(package_path, manifest):
    expected_groups = (
        ("annotated", manifest.get("annotated_images"), manifest.get("annotated_count")),
        ("historic", manifest.get("historic_images"), manifest.get("historic_count")),
    )
    missing = []
    count_mismatches = []

    for group_name, image_names, expected_count in expected_groups:
        if not isinstance(image_names, list):
            raise ValueError(f"Export folder manifest is missing {group_name}_images")
        if expected_count is not None and int(expected_count) != len(image_names):
            count_mismatches.append(
                f"{group_name}: count={expected_count}, listed={len(image_names)}"
            )
        for image_name in image_names:
            if not isinstance(image_name, str) or not image_name:
                raise ValueError(f"Export folder manifest has invalid {group_name} image name")
            if os.path.basename(image_name) != image_name:
                raise ValueError(f"Export folder manifest has unsafe image name: {image_name}")
            image_path = os.path.join(package_path, group_name, image_name)
            if not os.path.isfile(image_path):
                missing.append(f"{group_name}/{image_name}")

    if count_mismatches:
        raise ValueError(
            "Export folder manifest count mismatch: " + ", ".join(count_mismatches)
        )
    if missing:
        raise ValueError(
            "Export folder is missing manifest-listed images: " + ", ".join(missing[:20])
        )


def _load_package(package_path):
    if not package_path or not os.path.isdir(package_path):
        raise FileNotFoundError(f"Export folder not found: {package_path}")
    if os.path.basename(os.path.normpath(package_path)).endswith(".partial"):
        raise ValueError("Export folder is incomplete")

    missing_entries = [
        entry
        for entry in REQUIRED_PACKAGE_FILES
        if not os.path.isfile(_package_file_path(package_path, entry))
    ]
    if missing_entries:
        raise ValueError(
            "Export folder is missing required files: " + ", ".join(missing_entries)
        )

    with open(_package_file_path(package_path, "manifest.json"), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    _validate_manifest(manifest)
    _validate_manifest_image_entries(package_path, manifest)

    with open(_package_file_path(package_path, "db/data.json"), "r", encoding="utf-8") as handle:
        data_payload = json.load(handle)

    return manifest, data_payload


def _write_text_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _build_unique_export_dir(file_manager, output_root):
    base_name = f"display_state_{datetime.now():%Y%m%d_%H%M%S}"
    export_path = file_manager.join(output_root, base_name)
    partial_path = f"{export_path}.partial"
    if not file_manager.exists(export_path) and not file_manager.exists(partial_path):
        return base_name, export_path

    suffix = 1
    while True:
        export_name = f"{base_name}_{suffix:02d}"
        export_path = file_manager.join(output_root, export_name)
        partial_path = f"{export_path}.partial"
        if not file_manager.exists(export_path) and not file_manager.exists(partial_path):
            return export_name, export_path
        suffix += 1


def _copy_missing_files(
    file_manager,
    source_dir,
    target_dir,
    image_extensions,
    progress_callback,
    progress_state,
    stage_label,
):
    copied = 0
    skipped = 0
    copied_names = []

    if not os.path.isdir(source_dir):
        return {"copied": copied, "skipped": skipped, "copied_names": copied_names}

    file_manager.makedirs(target_dir, exist_ok=True)
    names = _list_image_names(file_manager, source_dir, image_extensions)
    for name in names:
        src = file_manager.join(source_dir, name)
        dst = file_manager.join(target_dir, name)
        if file_manager.exists(dst):
            skipped += 1
        else:
            file_manager.copy2(src, dst)
            copied += 1
            copied_names.append(name)
        progress_state["done"] += 1
        if callable(progress_callback):
            progress_callback(
                progress_state["done"],
                progress_state["total"],
                f"{stage_label} ({progress_state['done']}/{progress_state['total']})",
            )

    return {"copied": copied, "skipped": skipped, "copied_names": copied_names}


def _row_get(row, key_or_index, default=None):
    if row is None:
        return default
    try:
        if isinstance(key_or_index, str):
            return row.get(key_or_index, default)
        return row[key_or_index]
    except Exception:
        return default


def _fetch_existing_set(cursor, query):
    cursor.execute(query)
    values = set()
    for row in cursor.fetchall():
        if hasattr(row, "keys"):
            keys = list(row.keys())
            if keys:
                values.add(row.get(keys[0]))
        else:
            values.add(row[0])
    return values


def _merge_payload_into_db(controller, db_client, data_payload, progress_callback, progress_state):
    affected_jsns = set()
    stats = {
        "db_inserted": {
            "img_results": 0,
            "piece_result": 0,
            "classified_images": 0,
            "classified_image_defects": 0,
            "model_results": 0,
            "piece_result_defects": 0,
        },
        "db_skipped": {
            "img_results": 0,
            "piece_result": 0,
            "classified_images": 0,
            "classified_image_defects": 0,
            "model_results": 0,
            "piece_result_defects": 0,
        },
        "affected_jsns": affected_jsns,
    }

    with db_client.get_cursor() as cursor:
        existing_img_results = _fetch_existing_set(cursor, "SELECT img_name FROM img_results")
        existing_piece_result = _fetch_existing_set(cursor, "SELECT jsn FROM piece_result")
        existing_classified_images = _fetch_existing_set(
            cursor, "SELECT img_name FROM classified_images"
        )
        cursor.execute(
            "SELECT img_name, class_name, confidence, model_name, geometry_type, "
            "coordinates, image_width, image_height FROM model_results"
        )
        existing_model_result_rows = cursor.fetchall()
        existing_model_results = {
            _model_result_key(
                {
                    "img_name": _row_get(row, "img_name", _row_get(row, 0)),
                    "class_name": _row_get(row, "class_name", _row_get(row, 1)),
                    "confidence": _row_get(row, "confidence", _row_get(row, 2)),
                    "model_name": _row_get(row, "model_name", _row_get(row, 3)),
                    "geometry_type": _row_get(row, "geometry_type", _row_get(row, 4)),
                    "coordinates": _row_get(row, "coordinates", _row_get(row, 5)),
                    "image_width": _row_get(row, "image_width", _row_get(row, 6)),
                    "image_height": _row_get(row, "image_height", _row_get(row, 7)),
                }
            )
            for row in existing_model_result_rows
        }
        existing_model_result_ok_img_names = {
            _row_get(row, "img_name", _row_get(row, 0))
            for row in existing_model_result_rows
            if _row_get(row, "class_name", _row_get(row, 1)) == "OK"
        }

        cursor.execute(
            "SELECT ci.img_name, cid.class_name, cid.confidence, cid.remote_model_result_id "
            "FROM classified_image_defects cid "
            "JOIN classified_images ci ON ci.id = cid.classified_image_id"
        )
        existing_classified_defects = {
            (
                _row_get(row, "img_name", _row_get(row, 0)),
                _row_get(row, "class_name", _row_get(row, 1)),
                str(_row_get(row, "confidence", _row_get(row, 2))),
            )
            for row in cursor.fetchall()
        }
        cursor.execute(
            "SELECT remote_model_result_id FROM classified_image_defects "
            "WHERE remote_model_result_id IS NOT NULL"
        )
        existing_remote_model_result_ids = {
            _row_get(row, "remote_model_result_id", _row_get(row, 0))
            for row in cursor.fetchall()
        }

        cursor.execute(
            "SELECT pr.jsn, prd.class_name "
            "FROM piece_result_defects prd "
            "JOIN piece_result pr ON pr.id = prd.piece_result_id"
        )
        existing_piece_defects = {
            (
                _row_get(row, "jsn", _row_get(row, 0)),
                _row_get(row, "class_name", _row_get(row, 1)),
            )
            for row in cursor.fetchall()
        }

        for row in data_payload.get("img_results", []):
            img_name = row.get("img_name")
            if not img_name or img_name in existing_img_results:
                stats["db_skipped"]["img_results"] += 1
            else:
                cursor.execute(
                    "INSERT INTO img_results (img_name, result) VALUES (%s, %s)",
                    (img_name, row.get("result")),
                )
                existing_img_results.add(img_name)
                stats["db_inserted"]["img_results"] += 1
                affected_jsns.add(img_name.split("_")[0] if "_" in img_name else img_name)
            progress_state["done"] += 1
            if callable(progress_callback):
                progress_callback(progress_state["done"], progress_state["total"], "Merging database")

        for row in data_payload.get("piece_result", []):
            jsn = row.get("jsn")
            if not jsn or jsn in existing_piece_result:
                stats["db_skipped"]["piece_result"] += 1
            else:
                cursor.execute(
                    "INSERT INTO piece_result (jsn, operator_result, model_result, created_at) "
                    "VALUES (%s, %s, %s, %s)",
                    (
                        jsn,
                        row.get("operator_result"),
                        row.get("model_result"),
                        row.get("created_at"),
                    ),
                )
                existing_piece_result.add(jsn)
                stats["db_inserted"]["piece_result"] += 1
                affected_jsns.add(jsn)
            progress_state["done"] += 1
            if callable(progress_callback):
                progress_callback(progress_state["done"], progress_state["total"], "Merging database")

        for row in data_payload.get("classified_images", []):
            img_name = row.get("img_name")
            jsn = row.get("jsn")
            if not img_name or not jsn or img_name in existing_classified_images:
                stats["db_skipped"]["classified_images"] += 1
            else:
                cursor.execute("SELECT id FROM piece_result WHERE jsn = %s", (jsn,))
                piece_row = cursor.fetchone()
                if piece_row is None:
                    stats["db_skipped"]["classified_images"] += 1
                else:
                    cursor.execute(
                        "INSERT INTO classified_images "
                        "(img_name, operator_result, model_result, piece_id, created_at) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (
                            img_name,
                            row.get("operator_result"),
                            row.get("model_result"),
                            _row_get(piece_row, "id", _row_get(piece_row, 0)),
                            row.get("created_at"),
                        ),
                    )
                    existing_classified_images.add(img_name)
                    stats["db_inserted"]["classified_images"] += 1
                    affected_jsns.add(jsn)
            progress_state["done"] += 1
            if callable(progress_callback):
                progress_callback(progress_state["done"], progress_state["total"], "Merging database")

        for row in data_payload.get("model_results", []):
            img_name = row.get("img_name")
            class_name = row.get("class_name")
            result_key = _model_result_key(row)
            if (
                not img_name
                or not class_name
                or result_key in existing_model_results
                or (class_name == "OK" and img_name in existing_model_result_ok_img_names)
            ):
                stats["db_skipped"]["model_results"] += 1
            else:
                cursor.execute(
                    "INSERT INTO model_results "
                    "(img_name, class_name, confidence, created_at, model_name, geometry_type, "
                    "coordinates, image_width, image_height) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        img_name,
                        class_name,
                        row.get("confidence"),
                        row.get("created_at"),
                        row.get("model_name"),
                        row.get("geometry_type"),
                        _json_db_value(row.get("coordinates")),
                        row.get("image_width"),
                        row.get("image_height"),
                    ),
                )
                existing_model_results.add(result_key)
                if class_name == "OK":
                    existing_model_result_ok_img_names.add(img_name)
                stats["db_inserted"]["model_results"] += 1
                affected_jsns.add(img_name.split("_")[0] if "_" in img_name else img_name)
            progress_state["done"] += 1
            if callable(progress_callback):
                progress_callback(progress_state["done"], progress_state["total"], "Merging database")

        for row in data_payload.get("classified_image_defects", []):
            img_name = row.get("img_name")
            class_name = row.get("class_name")
            confidence = row.get("confidence")
            confidence_key = str(confidence)
            defect_key = (img_name, class_name, confidence_key)
            remote_model_result_id = row.get("remote_model_result_id")
            if (
                not img_name
                or not class_name
                or defect_key in existing_classified_defects
                or (
                    remote_model_result_id is not None
                    and remote_model_result_id in existing_remote_model_result_ids
                )
            ):
                stats["db_skipped"]["classified_image_defects"] += 1
            else:
                cursor.execute(
                    "SELECT id, piece_id FROM classified_images WHERE img_name = %s",
                    (img_name,),
                )
                classified_row = cursor.fetchone()
                if classified_row is None:
                    stats["db_skipped"]["classified_image_defects"] += 1
                else:
                    cursor.execute(
                        "INSERT INTO classified_image_defects "
                        "(classified_image_id, class_name, confidence, created_at, "
                        "remote_model_result_id, model_name, geometry_type, coordinates, "
                        "image_width, image_height) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            _row_get(classified_row, "id", _row_get(classified_row, 0)),
                            class_name,
                            confidence,
                            row.get("created_at"),
                            remote_model_result_id,
                            row.get("model_name"),
                            row.get("geometry_type"),
                            _json_db_value(row.get("coordinates")),
                            row.get("image_width"),
                            row.get("image_height"),
                        ),
                    )
                    existing_classified_defects.add(defect_key)
                    if remote_model_result_id is not None:
                        existing_remote_model_result_ids.add(remote_model_result_id)
                    stats["db_inserted"]["classified_image_defects"] += 1
                    piece_id = _row_get(classified_row, "piece_id", _row_get(classified_row, 1))
                    if piece_id is not None:
                        cursor.execute(
                            "SELECT jsn FROM piece_result WHERE id = %s",
                            (piece_id,),
                        )
                        piece_row = cursor.fetchone()
                        piece_jsn = _row_get(piece_row, "jsn", _row_get(piece_row, 0))
                        if piece_jsn:
                            affected_jsns.add(piece_jsn)
            progress_state["done"] += 1
            if callable(progress_callback):
                progress_callback(progress_state["done"], progress_state["total"], "Merging database")

        for row in data_payload.get("piece_result_defects", []):
            jsn = row.get("jsn")
            class_name = row.get("class_name")
            defect_key = (jsn, class_name)
            if not jsn or not class_name or defect_key in existing_piece_defects:
                stats["db_skipped"]["piece_result_defects"] += 1
            else:
                cursor.execute("SELECT id FROM piece_result WHERE jsn = %s", (jsn,))
                piece_row = cursor.fetchone()
                if piece_row is None:
                    stats["db_skipped"]["piece_result_defects"] += 1
                else:
                    cursor.execute(
                        "INSERT INTO piece_result_defects "
                        "(piece_result_id, class_name, confidence, created_at) "
                        "VALUES (%s, %s, %s, %s)",
                        (
                            _row_get(piece_row, "id", _row_get(piece_row, 0)),
                            class_name,
                            row.get("confidence"),
                            row.get("created_at"),
                        ),
                    )
                    existing_piece_defects.add(defect_key)
                    stats["db_inserted"]["piece_result_defects"] += 1
                    affected_jsns.add(jsn)
            progress_state["done"] += 1
            if callable(progress_callback):
                progress_callback(progress_state["done"], progress_state["total"], "Merging database")

    return stats


def export_display_state(controller, output_dir=None, progress_callback=None, db_client=None):
    file_manager = controller.file_manager
    db = db_client or getattr(controller.display, "db", None)
    if db is None:
        return {"ok": False, "error": "No database connection available"}

    annotated_dir = controller._get_visible_historic_dir()
    historic_dir = controller._get_export_historic_dir()
    image_extensions = tuple(getattr(controller.config, "image_extensions", IMAGE_EXTENSIONS))

    annotated_names = _list_image_names(file_manager, annotated_dir, image_extensions)
    historic_names = _list_image_names(file_manager, historic_dir, image_extensions)
    data_payload = _build_data_payload(db)
    manifest = _build_manifest(
        annotated_names=annotated_names,
        historic_names=historic_names,
        data_payload=data_payload,
        image_extensions=image_extensions,
    )

    output_root = str(output_dir or EXPORTS_DIR)
    file_manager.makedirs(output_root, exist_ok=True)
    package_name, package_path = _build_unique_export_dir(file_manager, output_root)
    partial_package_path = f"{package_path}.partial"
    total_steps = len(annotated_names) + len(historic_names) + 3
    done = 0

    if callable(progress_callback):
        progress_callback(done, total_steps, "Preparing export")

    database_sql = _build_database_sql(data_payload)
    done += 1
    if callable(progress_callback):
        progress_callback(done, total_steps, "Serializing database")

    file_manager.makedirs(partial_package_path, exist_ok=False)
    metadata_files = {
        "db/data.json": json.dumps(data_payload, indent=2, sort_keys=True),
        "db/database.sql": database_sql,
    }
    for relative_path, content in metadata_files.items():
        _write_text_file(_package_file_path(partial_package_path, relative_path), content)

    annotated_package_dir = file_manager.join(partial_package_path, "annotated")
    historic_package_dir = file_manager.join(partial_package_path, "historic")
    file_manager.makedirs(annotated_package_dir, exist_ok=True)
    file_manager.makedirs(historic_package_dir, exist_ok=True)

    done += 1
    if callable(progress_callback):
        progress_callback(done, total_steps, "Writing metadata")

    for name in annotated_names:
        file_manager.copy2(
            file_manager.join(annotated_dir, name),
            file_manager.join(annotated_package_dir, name),
        )
        done += 1
        if callable(progress_callback):
            progress_callback(done, total_steps, "Copying annotated images")

    for name in historic_names:
        file_manager.copy2(
            file_manager.join(historic_dir, name),
            file_manager.join(historic_package_dir, name),
        )
        done += 1
        if callable(progress_callback):
            progress_callback(done, total_steps, "Copying historic images")

    _write_text_file(
        _package_file_path(partial_package_path, "manifest.json"),
        json.dumps(manifest, indent=2, sort_keys=True),
    )
    os.rename(partial_package_path, package_path)

    done = total_steps
    if callable(progress_callback):
        progress_callback(done, total_steps, "Completed")

    return {
        "ok": True,
        "package_path": str(package_path),
        "package_name": package_name,
        "manifest": manifest,
    }


def import_display_state(controller, package_path, progress_callback=None, db_client=None):
    db = db_client or getattr(controller.display, "db", None)
    if db is None:
        return {"ok": False, "error": "No database connection available"}

    manifest, data_payload = _load_package(package_path)
    file_manager = controller.file_manager
    annotated_target_dir = controller._get_visible_historic_dir()
    historic_target_dir = controller._get_export_historic_dir()
    image_extensions = tuple(
        manifest.get("image_extensions") or getattr(controller.config, "image_extensions", IMAGE_EXTENSIONS)
    )

    db_row_total = sum(len(rows) for rows in data_payload.values())
    annotated_source_dir = file_manager.join(package_path, "annotated")
    historic_source_dir = file_manager.join(package_path, "historic")
    if not os.path.isdir(annotated_source_dir):
        raise ValueError("Export folder is missing annotated directory")
    if not os.path.isdir(historic_source_dir):
        raise ValueError("Export folder is missing historic directory")

    total_steps = (
        len(_list_image_names(file_manager, annotated_source_dir, image_extensions))
        + len(_list_image_names(file_manager, historic_source_dir, image_extensions))
        + db_row_total
        + 1
    )
    progress_state = {"done": 0, "total": max(1, total_steps)}

    if callable(progress_callback):
        progress_callback(0, progress_state["total"], "Preparing import")

    annotated_stats = _copy_missing_files(
        file_manager=file_manager,
        source_dir=annotated_source_dir,
        target_dir=annotated_target_dir,
        image_extensions=image_extensions,
        progress_callback=progress_callback,
        progress_state=progress_state,
        stage_label="Importing annotated images",
    )
    historic_stats = _copy_missing_files(
        file_manager=file_manager,
        source_dir=historic_source_dir,
        target_dir=historic_target_dir,
        image_extensions=image_extensions,
        progress_callback=progress_callback,
        progress_state=progress_state,
        stage_label="Importing historic images",
    )

    merge_stats = _merge_payload_into_db(
        controller=controller,
        db_client=db,
        data_payload=data_payload,
        progress_callback=progress_callback,
        progress_state=progress_state,
    )

    for jsn in sorted(merge_stats["affected_jsns"]):
        try:
            controller._recalculate_piece_result(jsn, db_client=db)
        except Exception:
            pass

    progress_state["done"] += 1
    if callable(progress_callback):
        progress_callback(progress_state["done"], progress_state["total"], "Refreshing runtime state")

    controller._invalidate_dataset_runtime_state(clear_historic_images=False)
    if getattr(controller.display, "historic_mode", False):
        controller.enter_historic_mode()

    return {
        "ok": True,
        "manifest": manifest,
        "annotated": annotated_stats,
        "historic": historic_stats,
        "db": {
            "inserted": merge_stats["db_inserted"],
            "skipped": merge_stats["db_skipped"],
            "affected_jsns": len(merge_stats["affected_jsns"]),
        },
    }

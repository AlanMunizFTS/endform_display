#!/usr/bin/env python3
"""
Import local images into img_results and set status to OK.

This utility is local-only (no SFTP). It reads image filenames from a folder and:
- inserts missing rows into img_results with result='OK'
- updates existing rows to result='OK' (enabled by default)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paths_config import HISTORIC_LOCAL_DIR

IMAGE_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg", ".bmp"}


def _collect_image_names(folder: Path, recursive: bool) -> List[str]:
    if recursive:
        paths: Iterable[Path] = folder.rglob("*")
    else:
        paths = folder.iterdir()

    names = {
        path.name
        for path in paths
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    return sorted(names)


import re


def _upsert_piece_result(db, img_name: str, operator_result: str) -> None:
    """Mirror of MainController._upsert_classification for standalone use."""
    m = re.search(r"_(OK|NOK)\.\w+$", img_name, re.IGNORECASE)
    model_result = m.group(1).upper() if m else "OK"
    jsn = img_name.split("_")[0] if "_" in img_name else img_name
    with db.get_cursor() as cursor:
        cursor.execute(
            "INSERT INTO piece_result (jsn, operator_result, model_result) "
            "VALUES (%s, %s, %s) "
            "ON CONFLICT (jsn) DO UPDATE SET "
            "operator_result = CASE WHEN piece_result.operator_result = 'NOK' "
            "  OR EXCLUDED.operator_result = 'NOK' THEN 'NOK' ELSE 'OK' END, "
            "model_result = CASE WHEN piece_result.model_result = 'NOK' "
            "  OR EXCLUDED.model_result = 'NOK' THEN 'NOK' ELSE 'OK' END "
            "RETURNING id",
            (jsn, operator_result, model_result),
        )
        piece_id = cursor.fetchone()["id"]
        cursor.execute(
            "INSERT INTO classified_images "
            "(img_name, operator_result, model_result, piece_id) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (img_name) DO UPDATE SET "
            "operator_result = EXCLUDED.operator_result, "
            "model_result = EXCLUDED.model_result, "
            "piece_id = EXCLUDED.piece_id",
            (img_name, operator_result, model_result, piece_id),
        )


def _import_images(folder: Path, recursive: bool, dry_run: bool, update_existing: bool) -> int:
    try:
        from db import get_db_connection
    except ModuleNotFoundError as exc:
        if exc.name in {"psycopg2", "psycopg2_binary"}:
            print("Missing dependency: psycopg2.")
            print("Run with project venv and install dependency:")
            print(r".\venv\Scripts\python.exe -m pip install psycopg2-binary==2.9.9")
            print(r".\venv\Scripts\python.exe utilities\import_local_images_to_db.py")
            return 1
        raise

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder does not exist or is not a directory: {folder}")

    image_names = _collect_image_names(folder, recursive=recursive)
    if not image_names:
        print("No image files found.")
        return 0

    db = get_db_connection()
    try:
        inserted = 0
        updated = 0
        skipped = 0

        query_check = "SELECT img_name, result FROM img_results WHERE img_name = %s"
        query_insert = "INSERT INTO img_results (img_name, result) VALUES (%s, %s)"
        query_update = "UPDATE img_results SET result = %s WHERE img_name = %s"

        for img_name in image_names:
            rows = db.fetch(query_check, (img_name,))

            if not rows:
                if not dry_run:
                    db.execute(query_insert, (img_name, "OK"))
                    try:
                        _upsert_piece_result(db, img_name, "OK")
                    except Exception as exc:
                        print(f"Warning: could not upsert piece_result for {img_name}: {exc}")
                inserted += 1
                continue

            if update_existing:
                current_has_ok = any(str(row.get("result") or "").strip().upper() == "OK" for row in rows)
                if current_has_ok:
                    skipped += 1
                    continue
                if not dry_run:
                    db.execute(query_update, ("OK", img_name))
                    try:
                        _upsert_piece_result(db, img_name, "OK")
                    except Exception as exc:
                        print(f"Warning: could not upsert piece_result for {img_name}: {exc}")
                updated += 1
            else:
                skipped += 1

        print("=" * 72)
        print("LOCAL IMAGE IMPORT SUMMARY")
        print("=" * 72)
        print(f"Folder: {folder.resolve()}")
        print(f"Recursive: {recursive}")
        print(f"Mode: {'DRY-RUN' if dry_run else 'WRITE'}")
        print(f"Update existing: {update_existing}")
        print(f"Images scanned: {len(image_names)}")
        print(f"Inserted (OK): {inserted}")
        print(f"Updated to OK: {updated}")
        print(f"Skipped: {skipped}")
        print("=" * 72)
        return 0
    finally:
        db.close()


def main() -> int:
    default_folder = (
        HISTORIC_LOCAL_DIR
        if HISTORIC_LOCAL_DIR.is_absolute()
        else REPO_ROOT / HISTORIC_LOCAL_DIR
    )

    parser = argparse.ArgumentParser(
        description="Read local images from a folder and insert/update img_results with result='OK'."
    )
    parser.add_argument(
        "--folder",
        default=str(default_folder),
        help=f"Folder containing local images. Default: {default_folder}",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include images in subfolders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to the database.",
    )
    parser.add_argument(
        "--no-update-existing",
        action="store_true",
        help="Only insert missing rows; do not update existing rows to OK.",
    )

    args = parser.parse_args()
    return _import_images(
        folder=Path(args.folder),
        recursive=args.recursive,
        dry_run=args.dry_run,
        update_existing=not args.no_update_existing,
    )


if __name__ == "__main__":
    raise SystemExit(main())

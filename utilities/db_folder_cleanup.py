#!/usr/bin/env python3
"""
Delete classified images by DB image names, with a hard delete cap.

Behavior:
- Reads image names from `img_results` in PostgreSQL.
- Scans classification folders:
  - side_ok, side_nok
  - front_ok, front_nok
  - diag_ok, diag_nok
- Matches files by exact filename.
- Enforces a lock: max deleted files defaults to DB row count (N).
- Prints a full summary.

Default mode is dry-run. Use --delete to actually remove files.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

from db import get_db_connection
from paths_config import STATUS_SYNC_DIRS


def _status_folders(base_dir: Path) -> List[Path]:
    folders: List[Path] = []
    seen: Set[Path] = set()
    for statuses in STATUS_SYNC_DIRS.values():
        for folder_name in statuses.values():
            folder_path = base_dir / folder_name
            if folder_path not in seen:
                folders.append(folder_path)
                seen.add(folder_path)
    return folders


def _fetch_db_image_names() -> List[str]:
    db = get_db_connection()
    try:
        query_options = [
            "SELECT img_name FROM img_results WHERE img_name IS NOT NULL AND BTRIM(img_name) <> ''",
            "SELECT name AS img_name FROM img_results WHERE name IS NOT NULL AND BTRIM(name) <> ''",
        ]

        rows = None
        last_error = None
        for query in query_options:
            try:
                rows = db.fetch(query)
                break
            except Exception as exc:  # pragma: no cover - environment dependent
                last_error = exc

        if rows is None:
            raise RuntimeError(f"Could not read image names from DB: {last_error}")

        image_names: List[str] = []
        for row in rows:
            raw_name = row.get("img_name")
            if raw_name is None:
                continue
            name = str(raw_name).strip()
            if name:
                image_names.append(name)
        return image_names
    finally:
        db.close()


def _find_matches(
    folders: Sequence[Path],
    db_names_lower: Set[str],
) -> Tuple[List[Path], Dict[str, int], List[Path]]:
    matches: List[Path] = []
    found_by_folder: Dict[str, int] = defaultdict(int)
    missing_folders: List[Path] = []

    for folder in folders:
        if not folder.exists():
            missing_folders.append(folder)
            continue

        for path in folder.iterdir():
            if not path.is_file():
                continue
            if path.name.lower() in db_names_lower:
                matches.append(path)
                found_by_folder[folder.name] += 1

    matches.sort(key=lambda p: (p.parent.name.lower(), p.name.lower()))
    return matches, found_by_folder, missing_folders


def _run_cleanup(base_dir: Path, do_delete: bool, limit: int | None) -> int:
    db_names = _fetch_db_image_names()
    db_rows_count = len(db_names)
    db_unique_lower = {name.lower() for name in db_names}

    max_deletions = db_rows_count if limit is None else limit
    if max_deletions < 0:
        raise ValueError("--limit must be >= 0")

    folders = _status_folders(base_dir)
    matches, found_by_folder, missing_folders = _find_matches(folders, db_unique_lower)

    target_count = min(len(matches), max_deletions)
    skipped_by_lock = max(0, len(matches) - max_deletions)

    deleted_count = 0
    deleted_by_folder: Dict[str, int] = defaultdict(int)
    errors: List[str] = []

    if do_delete:
        for match in matches:
            if deleted_count >= max_deletions:
                break
            try:
                match.unlink()
                deleted_count += 1
                deleted_by_folder[match.parent.name] += 1
            except Exception as exc:  # pragma: no cover - filesystem dependent
                errors.append(f"{match}: {exc}")

    print("=" * 72)
    print("DB FOLDER CLEANUP SUMMARY")
    print("=" * 72)
    print(f"Mode: {'DELETE' if do_delete else 'DRY-RUN'}")
    print(f"Base dir: {base_dir.resolve()}")
    print(f"DB rows (N): {db_rows_count}")
    print(f"DB unique names: {len(db_unique_lower)}")
    print(f"Delete cap (lock): {max_deletions}")
    print(f"Matching files found: {len(matches)}")
    print(f"Locked out by cap: {skipped_by_lock}")
    if do_delete:
        print(f"Deleted files: {deleted_count}")
    else:
        print(f"Would delete files: {target_count}")
    print("-" * 72)

    if missing_folders:
        print("Missing folders:")
        for folder in missing_folders:
            print(f"  - {folder}")
        print("-" * 72)

    print("Matches by folder:")
    for folder in folders:
        print(f"  - {folder.name}: {found_by_folder.get(folder.name, 0)}")
    print("-" * 72)

    if do_delete:
        print("Deleted by folder:")
        for folder in folders:
            print(f"  - {folder.name}: {deleted_by_folder.get(folder.name, 0)}")
        print("-" * 72)

    if errors:
        print("Errors:")
        for err in errors:
            print(f"  - {err}")
        print("-" * 72)

    if not matches:
        print("No matching files found in target folders.")
    elif do_delete and deleted_count >= max_deletions:
        print("Stopped because delete cap was reached.")
    elif do_delete:
        print("Finished deleting matching files (within cap).")
    else:
        print("Dry-run complete. Use --delete to apply.")

    print("=" * 72)

    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Delete files in side/front/diag OK/NOK folders using image names from DB. "
            "Hard cap defaults to DB row count."
        )
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory that contains side_ok/side_nok/front_ok/front_nok/diag_ok/diag_nok.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete files. Without this flag, script runs in dry-run mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional custom delete cap. Default is DB row count (N).",
    )

    args = parser.parse_args()
    return _run_cleanup(
        base_dir=Path(args.base_dir),
        do_delete=args.delete,
        limit=args.limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())

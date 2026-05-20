"""
Count OK/NOK pieces from an exported display manifest.

Usage:
    python utilities/count_manifest_piece_status.py
    python utilities/count_manifest_piece_status.py exports/display_state_20260515_055731/manifest.json

A piece is identified by the JSN prefix before "_Cam".
If any image for a JSN is NOK, the whole piece is counted as NOK.
Only JSNs whose images are all OK are counted as OK.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


DEFAULT_MANIFEST = Path("tmp_display/manifest_tesla3_left/manifest_19.json")
STATUS_RE = re.compile(r"^(?P<jsn>.+?)_Cam\d+_.*_(?P<status>OK|NOK)\.png$", re.IGNORECASE)


def parse_image_name(image_name: str) -> tuple[str, str]:
    match = STATUS_RE.match(image_name)
    if not match:
        raise ValueError(f"Invalid image name format: {image_name}")

    return match.group("jsn"), match.group("status").upper()


def parse_jsn_date(jsn: str) -> str:
    raw_date = jsn[5:11]
    try:
        parsed = datetime.strptime(raw_date, "%m%d%y")
    except ValueError as exc:
        raise ValueError(f"Invalid JSN date '{raw_date}' in JSN: {jsn}") from exc

    return parsed.strftime("%Y-%m-%d")


def count_piece_status(historic_images: list[str]) -> tuple[Counter[str], Counter[str]]:
    image_counts: Counter[str] = Counter()
    statuses_by_jsn: dict[str, Counter[str]] = defaultdict(Counter)

    for image_name in historic_images:
        jsn, status = parse_image_name(image_name)
        image_counts[status] += 1
        statuses_by_jsn[jsn][status] += 1

    piece_counts: Counter[str] = Counter()
    for status_counts in statuses_by_jsn.values():
        if status_counts["NOK"] > 0:
            piece_counts["NOK"] += 1
        else:
            piece_counts["OK"] += 1

    return piece_counts, image_counts


def count_piece_status_by_day(historic_images: list[str]) -> dict[str, Counter[str]]:
    statuses_by_jsn: dict[str, Counter[str]] = defaultdict(Counter)

    for image_name in historic_images:
        jsn, status = parse_image_name(image_name)
        statuses_by_jsn[jsn][status] += 1

    piece_counts_by_day: dict[str, Counter[str]] = defaultdict(Counter)
    for jsn, status_counts in statuses_by_jsn.items():
        day = parse_jsn_date(jsn)
        if status_counts["NOK"] > 0:
            piece_counts_by_day[day]["NOK"] += 1
        else:
            piece_counts_by_day[day]["OK"] += 1

    return piece_counts_by_day


def load_historic_images(manifest_path: Path) -> list[str]:
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    historic_images = manifest.get("historic_images")
    if not isinstance(historic_images, list):
        raise ValueError("manifest.json does not contain a valid 'historic_images' list")

    return historic_images


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count OK/NOK unique JSN pieces from manifest['historic_images']."
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        default=DEFAULT_MANIFEST,
        type=Path,
        help=f"Path to manifest.json. Default: {DEFAULT_MANIFEST}",
    )
    args = parser.parse_args()

    manifest_path = args.manifest
    historic_images = load_historic_images(manifest_path)
    piece_counts, image_counts = count_piece_status(historic_images)
    piece_counts_by_day = count_piece_status_by_day(historic_images)
    total_pieces = piece_counts["OK"] + piece_counts["NOK"]
    total_images = image_counts["OK"] + image_counts["NOK"]

    print(f"Manifest: {manifest_path}")
    print()
    print("Pieces by unique JSN:")
    print(f"  OK:  {piece_counts['OK']}")
    print(f"  NOK: {piece_counts['NOK']}")
    print(f"  Total: {total_pieces}")
    print()
    print("Pieces by day:")
    for day in sorted(piece_counts_by_day):
        day_counts = piece_counts_by_day[day]
        day_total = day_counts["OK"] + day_counts["NOK"]
        print(f"  {day}: OK={day_counts['OK']} NOK={day_counts['NOK']} Total={day_total}")
    print()
    print("Images in historic_images:")
    print(f"  OK:  {image_counts['OK']}")
    print(f"  NOK: {image_counts['NOK']}")
    print(f"  Total: {total_images}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

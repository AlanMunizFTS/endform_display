import argparse
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_status_folder(folder_name: str) -> tuple[str, str] | None:
    lower_name = folder_name.lower()
    if lower_name.endswith("_ok"):
        return lower_name[: -len("_ok")], "ok"
    if lower_name.endswith("_nok"):
        return lower_name[: -len("_nok")], "nok"
    return None


def list_subfolders(parent: Path) -> dict[str, Path]:
    if not parent.exists() or not parent.is_dir():
        return {}
    return {item.name: item for item in parent.iterdir() if item.is_dir()}


def count_images(folder: Path) -> int:
    return sum(
        1
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_status_map(parent: Path) -> dict[tuple[str, str], str]:
    status_map: dict[tuple[str, str], str] = {}
    for folder in parent.iterdir():
        if not folder.is_dir():
            continue
        parsed = parse_status_folder(folder.name)
        if parsed is None:
            continue
        position, status = parsed
        for file in folder.iterdir():
            if not file.is_file() or file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            key = (position, file.name.lower())
            status_map[key] = status
    return status_map


def ratio_percent(base_count: int, compare_count: int) -> str:
    if base_count == 0:
        return "100.00%" if compare_count == 0 else "N/A"
    return f"{(compare_count / base_count) * 100:.2f}%"


def match_percent(base_count: int, compare_count: int) -> str:
    bigger = max(base_count, compare_count)
    if bigger == 0:
        return "100.00%"
    return f"{(min(base_count, compare_count) / bigger) * 100:.2f}%"


def build_transition_rows(
    parent_a: Path, parent_b: Path
) -> tuple[list[list[str]], list[str]]:
    status_a = collect_status_map(parent_a)
    status_b = collect_status_map(parent_b)

    per_position = defaultdict(lambda: {"ok_to_nok": 0, "nok_to_ok": 0, "same": 0})
    in_both = set(status_a) & set(status_b)

    for position, image_name in in_both:
        a_value = status_a[(position, image_name)]
        b_value = status_b[(position, image_name)]
        if a_value == "ok" and b_value == "nok":
            per_position[position]["ok_to_nok"] += 1
        elif a_value == "nok" and b_value == "ok":
            per_position[position]["nok_to_ok"] += 1
        else:
            per_position[position]["same"] += 1

    rows: list[list[str]] = []
    total_ok_to_nok = 0
    total_nok_to_ok = 0
    total_same = 0
    total_in_both = 0

    for position in sorted(per_position):
        ok_to_nok = per_position[position]["ok_to_nok"]
        nok_to_ok = per_position[position]["nok_to_ok"]
        same = per_position[position]["same"]
        comparable = ok_to_nok + nok_to_ok + same

        total_ok_to_nok += ok_to_nok
        total_nok_to_ok += nok_to_ok
        total_same += same
        total_in_both += comparable

        rows.append(
            [
                position,
                str(comparable),
                str(ok_to_nok),
                str(nok_to_ok),
                f"{((ok_to_nok + nok_to_ok) / comparable * 100):.2f}%"
                if comparable
                else "0.00%",
            ]
        )

    rows.append(
        [
            "TOTAL",
            str(total_in_both),
            str(total_ok_to_nok),
            str(total_nok_to_ok),
            f"{((total_ok_to_nok + total_nok_to_ok) / total_in_both * 100):.2f}%"
            if total_in_both
            else "0.00%",
        ]
    )

    extra_info = [
        f"Only in {parent_a.name}: {len(set(status_a) - set(status_b))} images",
        f"Only in {parent_b.name}: {len(set(status_b) - set(status_a))} images",
    ]
    return rows, extra_info


def build_report(parent_a: Path, parent_b: Path) -> str:
    folders_a = list_subfolders(parent_a)
    folders_b = list_subfolders(parent_b)

    all_folder_names = sorted(set(folders_a) | set(folders_b))

    headers = [
        "Folder",
        f"{parent_a.name}_count",
        f"{parent_b.name}_count",
        "Diff(B-A)",
        "B_vs_A",
        "Match",
    ]

    rows = []
    total_a = 0
    total_b = 0

    for folder_name in all_folder_names:
        count_a = count_images(folders_a[folder_name]) if folder_name in folders_a else 0
        count_b = count_images(folders_b[folder_name]) if folder_name in folders_b else 0

        total_a += count_a
        total_b += count_b

        rows.append(
            [
                folder_name,
                str(count_a),
                str(count_b),
                str(count_b - count_a),
                ratio_percent(count_a, count_b),
                match_percent(count_a, count_b),
            ]
        )

    rows.append(
        [
            "TOTAL",
            str(total_a),
            str(total_b),
            str(total_b - total_a),
            ratio_percent(total_a, total_b),
            match_percent(total_a, total_b),
        ]
    )

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt(row: list[str]) -> str:
        return " | ".join(row[i].ljust(widths[i]) for i in range(len(row)))

    divider = "-+-".join("-" * width for width in widths)
    lines = [
        f"Image Folder Comparison",
        f"Parent A: {parent_a.resolve()}",
        f"Parent B: {parent_b.resolve()}",
        "",
        fmt(headers),
        divider,
    ]
    lines.extend(fmt(row) for row in rows)
    lines.append("")
    lines.append("B_vs_A = (count in parent B / count in parent A) * 100")
    lines.append("Match = (smaller count / larger count) * 100")

    transition_headers = [
        "Position",
        "Comparable",
        "OK->NOK",
        "NOK->OK",
        "Changed",
    ]
    transition_rows, extra_info = build_transition_rows(parent_a, parent_b)
    transition_widths = [
        max(len(transition_headers[i]), *(len(row[i]) for row in transition_rows))
        for i in range(len(transition_headers))
    ]

    def fmt_transition(row: list[str]) -> str:
        return " | ".join(
            row[i].ljust(transition_widths[i]) for i in range(len(row))
        )

    transition_divider = "-+-".join("-" * width for width in transition_widths)
    lines.append("")
    lines.append("Classification Changes (Parent A -> Parent B)")
    lines.append(fmt_transition(transition_headers))
    lines.append(transition_divider)
    lines.extend(fmt_transition(row) for row in transition_rows)
    lines.append("")
    lines.extend(extra_info)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare image counts in matching subfolders under two parent folders."
    )
    parser.add_argument(
        "--parent-a",
        default="./classified",
        help="First parent folder (base). Default: ./classified",
    )
    parser.add_argument(
        "--parent-b",
        default="./prueba2",
        help="Second parent folder (compare). Default: ./prueba2",
    )
    parser.add_argument(
        "--output",
        default="folder_comparison_report.txt",
        help="Output report file path. Default: folder_comparison_report.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parent_a = Path(args.parent_a)
    parent_b = Path(args.parent_b)

    report = build_report(parent_a, parent_b)
    print(report)

    output_path = Path(args.output)
    output_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()

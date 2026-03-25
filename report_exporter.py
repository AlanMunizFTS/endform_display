from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from paths_config import REPORTS_DIR


STATUS_COLUMNS = ("OK", "NOK", "FOK", "FNOK", "Total")


def _format_report_filename(start_at, end_at):
    if start_at is None or end_at is None:
        raise ValueError("Stats report requires a valid DB date range")

    return (
        f"reporte_{start_at:%Y%m%d}_{end_at:%Y%m%d}_{start_at:%H%M}_{end_at:%H%M}.xlsx"
    )


def export_stats_report(controller, db_client=None, output_dir=None):
    report_data = controller.build_piece_stats_report(db_client=db_client)
    matrix_rows = list(report_data.get("rows") or [])
    if not matrix_rows:
        raise ValueError("No piece stats available to export")

    report_dir = Path(output_dir or REPORTS_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Stats"

    header_fill = PatternFill(fill_type="solid", fgColor="404040")
    total_fill = PatternFill(fill_type="solid", fgColor="6E6E6E")
    header_font = Font(color="FFFFFF", bold=True)
    total_font = Font(color="FFFFFF", bold=True)
    center = Alignment(horizontal="center", vertical="center")
    left = Alignment(horizontal="left", vertical="center")

    headers = ["Class Name", *STATUS_COLUMNS]
    for col_idx, header in enumerate(headers, start=1):
        cell = sheet.cell(row=1, column=col_idx, value=header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center

    for row_idx, row in enumerate(matrix_rows, start=2):
        values = [
            row.get("class_name", ""),
            row.get("OK", 0),
            row.get("NOK", 0),
            row.get("FOK", 0),
            row.get("FNOK", 0),
            row.get("Total", 0),
        ]
        is_total = bool(row.get("is_total"))
        for col_idx, value in enumerate(values, start=1):
            cell = sheet.cell(row=row_idx, column=col_idx, value=value)
            cell.alignment = left if col_idx == 1 else center
            if is_total:
                cell.fill = total_fill
                cell.font = total_font

    sheet.freeze_panes = "A2"
    sheet.column_dimensions["A"].width = 28
    for column_letter in ("B", "C", "D", "E", "F"):
        sheet.column_dimensions[column_letter].width = 12

    filename = _format_report_filename(
        report_data.get("start_at"),
        report_data.get("end_at"),
    )
    output_path = report_dir / filename
    workbook.save(output_path)
    return str(output_path)

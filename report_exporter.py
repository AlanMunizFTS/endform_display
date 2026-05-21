from datetime import datetime, timedelta
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from paths_config import REPORTS_DIR


STATUS_COLUMNS = ("OK", "NOK", "FOK", "FNOK", "Total")
TRACEABILITY_STATUS_COLUMNS = ("OK", "NOK", "Total")
TRACEABILITY_REPORT_KIND = "ok_nok_by_jsn"


def parse_jsn_datetime(jsn):
    text = str(jsn or "").strip()
    if len(text) < 17 or not text[5:17].isdigit():
        raise ValueError("JSN does not contain MMDDYYHHMMSS at positions 6-17")

    date_token = text[5:11]
    time_token = text[11:17]
    month = int(date_token[0:2])
    day = int(date_token[2:4])
    year = 2000 + int(date_token[4:6])
    hour = int(time_token[0:2])
    minute = int(time_token[2:4])
    second = int(time_token[4:6])
    return datetime(year, month, day, hour, minute, second)


def _normalize_traceability_result(value):
    result = str(value or "").strip().upper()
    if result in {"OK", "FOK"}:
        return "OK"
    if result in {"NOK", "FNOK"}:
        return "NOK"
    return None


def _fetch_traceability_piece_rows(db_client):
    return list(
        db_client.fetch(
            "SELECT jsn, model_result FROM piece_result ORDER BY jsn"
        )
        or []
    )


def build_ok_nok_traceability_report(db_client):
    rows = _fetch_traceability_piece_rows(db_client)
    warnings = []
    valid_entries = []

    for row in rows:
        jsn = row.get("jsn") if hasattr(row, "get") else row[0]
        model_result = row.get("model_result") if hasattr(row, "get") else row[1]
        normalized_result = _normalize_traceability_result(model_result)
        if normalized_result is None:
            warnings.append(
                {
                    "jsn": jsn,
                    "reason": f"Unsupported model_result: {model_result}",
                }
            )
            continue

        try:
            captured_at = parse_jsn_datetime(jsn)
        except ValueError as exc:
            warnings.append({"jsn": jsn, "reason": str(exc)})
            continue

        valid_entries.append(
            {
                "jsn": str(jsn),
                "result": normalized_result,
                "captured_at": captured_at,
            }
        )

    day_map = {}
    hour_map = {}
    total_ok = 0
    total_nok = 0

    for entry in valid_entries:
        captured_at = entry["captured_at"]
        result = entry["result"]
        if result == "OK":
            total_ok += 1
        else:
            total_nok += 1

        day_key = captured_at.date()
        day_counts = day_map.setdefault(day_key, {"OK": 0, "NOK": 0})
        day_counts[result] += 1

        hour_key = captured_at.replace(minute=0, second=0, microsecond=0)
        hour_counts = hour_map.setdefault(hour_key, {"OK": 0, "NOK": 0})
        hour_counts[result] += 1

    start_at = min((entry["captured_at"] for entry in valid_entries), default=None)
    end_at = max((entry["captured_at"] for entry in valid_entries), default=None)

    day_rows = []
    for day_key in sorted(day_map):
        counts = day_map[day_key]
        total = counts["OK"] + counts["NOK"]
        day_rows.append(
            {
                "date": day_key,
                "OK": counts["OK"],
                "NOK": counts["NOK"],
                "Total": total,
                "pct_ok": counts["OK"] / total if total else 0,
                "pct_nok": counts["NOK"] / total if total else 0,
            }
        )

    hour_rows = []
    if start_at is not None and end_at is not None:
        current_hour = start_at.replace(minute=0, second=0, microsecond=0)
        final_hour = end_at.replace(minute=0, second=0, microsecond=0)
        while current_hour <= final_hour:
            counts = hour_map.get(current_hour, {"OK": 0, "NOK": 0})
            total = counts["OK"] + counts["NOK"]
            hour_rows.append(
                {
                    "date": current_hour.date(),
                    "hour": current_hour.hour,
                    "range": f"{current_hour:%H}:00 - {current_hour:%H}:59",
                    "OK": counts["OK"],
                    "NOK": counts["NOK"],
                    "Total": total,
                    "pct_ok": counts["OK"] / total if total else 0,
                    "pct_nok": counts["NOK"] / total if total else 0,
                }
            )
            current_hour += timedelta(hours=1)

    if not rows:
        warnings.append({"jsn": "", "reason": "No piece_result rows available"})

    total = total_ok + total_nok
    return {
        "kind": TRACEABILITY_REPORT_KIND,
        "start_at": start_at,
        "end_at": end_at,
        "summary": {
            "OK": total_ok,
            "NOK": total_nok,
            "Total": total,
            "pct_ok": total_ok / total if total else 0,
            "pct_nok": total_nok / total if total else 0,
        },
        "day_rows": day_rows,
        "hour_rows": hour_rows,
        "warnings": warnings,
    }


def _format_report_filename(start_at, end_at):
    if start_at is None or end_at is None:
        raise ValueError("Stats report requires a valid DB date range")

    return (
        f"reporte_{start_at:%Y%m%d}_{end_at:%Y%m%d}_{start_at:%H%M}_{end_at:%H%M}.xlsx"
    )


def _format_traceability_filename(created_at=None):
    timestamp = created_at or datetime.now()
    return f"desglose_ok_nok_{timestamp:%Y%m%d_%H%M%S}.xlsx"


def _apply_headers(sheet, headers, header_fill, header_font, center):
    for col_idx, header in enumerate(headers, start=1):
        cell = sheet.cell(row=1, column=col_idx, value=header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center


def _write_count_row(
    sheet,
    row_idx,
    values,
    total_fill,
    total_font,
    center,
    percent_columns,
):
    for col_idx, value in enumerate(values, start=1):
        cell = sheet.cell(row=row_idx, column=col_idx, value=value)
        cell.alignment = center
        if row_idx == 2 and sheet.title == "Resumen OK-NOK":
            cell.fill = total_fill
            cell.font = total_font
        if col_idx in percent_columns:
            cell.number_format = "0.00%"


def _build_traceability_workbook(db_client):
    report_data = build_ok_nok_traceability_report(db_client)
    workbook = Workbook()
    header_fill = PatternFill(fill_type="solid", fgColor="404040")
    total_fill = PatternFill(fill_type="solid", fgColor="6E6E6E")
    header_font = Font(color="FFFFFF", bold=True)
    total_font = Font(color="FFFFFF", bold=True)
    center = Alignment(horizontal="center", vertical="center")
    left = Alignment(horizontal="left", vertical="center")

    summary_sheet = workbook.active
    summary_sheet.title = "Resumen OK-NOK"
    summary_headers = [
        "Periodo inicio",
        "Periodo fin",
        "OK",
        "NOK",
        "Total",
        "% OK",
        "% NOK",
    ]
    _apply_headers(summary_sheet, summary_headers, header_fill, header_font, center)
    summary = report_data["summary"]
    _write_count_row(
        summary_sheet,
        2,
        [
            report_data["start_at"].strftime("%Y-%m-%d %H:%M:%S")
            if report_data["start_at"]
            else "",
            report_data["end_at"].strftime("%Y-%m-%d %H:%M:%S")
            if report_data["end_at"]
            else "",
            summary["OK"],
            summary["NOK"],
            summary["Total"],
            summary["pct_ok"],
            summary["pct_nok"],
        ],
        total_fill,
        total_font,
        center,
        percent_columns={6, 7},
    )

    day_sheet = workbook.create_sheet("Por dia")
    count_headers = ["Fecha", *TRACEABILITY_STATUS_COLUMNS, "% OK", "% NOK"]
    _apply_headers(day_sheet, count_headers, header_fill, header_font, center)
    for row_idx, row in enumerate(report_data["day_rows"], start=2):
        _write_count_row(
            day_sheet,
            row_idx,
            [
                row["date"].strftime("%Y-%m-%d"),
                row["OK"],
                row["NOK"],
                row["Total"],
                row["pct_ok"],
                row["pct_nok"],
            ],
            total_fill,
            total_font,
            center,
            percent_columns={5, 6},
        )

    hour_sheet = workbook.create_sheet("Por hora")
    hour_headers = ["Fecha", "Hora", "Rango", *TRACEABILITY_STATUS_COLUMNS, "% OK", "% NOK"]
    _apply_headers(hour_sheet, hour_headers, header_fill, header_font, center)
    for row_idx, row in enumerate(report_data["hour_rows"], start=2):
        _write_count_row(
            hour_sheet,
            row_idx,
            [
                row["date"].strftime("%Y-%m-%d"),
                f"{row['hour']:02d}",
                row["range"],
                row["OK"],
                row["NOK"],
                row["Total"],
                row["pct_ok"],
                row["pct_nok"],
            ],
            total_fill,
            total_font,
            center,
            percent_columns={7, 8},
        )

    warning_sheet = workbook.create_sheet("Advertencias")
    _apply_headers(warning_sheet, ["JSN", "Motivo"], header_fill, header_font, center)
    for row_idx, warning in enumerate(report_data["warnings"], start=2):
        jsn_cell = warning_sheet.cell(row=row_idx, column=1, value=warning.get("jsn", ""))
        reason_cell = warning_sheet.cell(row=row_idx, column=2, value=warning.get("reason", ""))
        jsn_cell.alignment = left
        reason_cell.alignment = left

    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        for column_cells in sheet.columns:
            column_letter = column_cells[0].column_letter
            max_width = max(
                len(str(cell.value or ""))
                for cell in column_cells
            )
            sheet.column_dimensions[column_letter].width = min(max(max_width + 2, 12), 42)

    return workbook


def _add_stats_matrix_sheet(workbook, matrix_rows, sheet=None):
    sheet = sheet or workbook.create_sheet("Stats")
    sheet.title = "Stats"
    header_fill = PatternFill(fill_type="solid", fgColor="404040")
    total_fill = PatternFill(fill_type="solid", fgColor="6E6E6E")
    header_font = Font(color="FFFFFF", bold=True)
    total_font = Font(color="FFFFFF", bold=True)
    center = Alignment(horizontal="center", vertical="center")
    left = Alignment(horizontal="left", vertical="center")

    headers = ["Class Name", *STATUS_COLUMNS]
    _apply_headers(sheet, headers, header_fill, header_font, center)

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
    return sheet


def export_ok_nok_traceability_report(db_client, output_dir=None, created_at=None):
    report_dir = Path(output_dir or REPORTS_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)
    workbook = _build_traceability_workbook(db_client)
    filename = _format_traceability_filename(created_at=created_at)
    output_path = report_dir / filename
    workbook.save(output_path)
    return str(output_path)


def export_combined_traceability_report(controller, db_client=None, output_dir=None, created_at=None):
    db = db_client or getattr(controller.display, "db", None)
    if db is None:
        raise ValueError("No database connection available")

    matrix_data = controller.build_piece_stats_report(db_client=db)
    matrix_rows = list(matrix_data.get("rows") or [])
    if not matrix_rows:
        raise ValueError("No piece stats available to export")

    report_dir = Path(output_dir or REPORTS_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)
    workbook = _build_traceability_workbook(db)
    _add_stats_matrix_sheet(workbook, matrix_rows)
    filename = _format_traceability_filename(created_at=created_at)
    output_path = report_dir / filename
    workbook.save(output_path)
    return str(output_path)


def export_stats_report(controller, db_client=None, output_dir=None):
    report_data = controller.build_piece_stats_report(db_client=db_client)
    matrix_rows = list(report_data.get("rows") or [])
    if not matrix_rows:
        raise ValueError("No piece stats available to export")

    report_dir = Path(output_dir or REPORTS_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    _add_stats_matrix_sheet(workbook, matrix_rows, sheet=workbook.active)

    filename = _format_report_filename(
        report_data.get("start_at"),
        report_data.get("end_at"),
    )
    output_path = report_dir / filename
    workbook.save(output_path)
    return str(output_path)

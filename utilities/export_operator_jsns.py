#!/usr/bin/env python3
"""Exporta a un TXT los JSN clasificados por el operador como OK o NOK.

El reporte usa ``piece_result.operator_result``, que es el resultado consolidado
por pieza: si alguna imagen de un JSN se marcó como NOK, la pieza se reporta
como NOK. No modifica registros de la base de datos.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Mapping

# Allow `python utilities/export_operator_jsns.py` from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db import get_db_connection
from paths_config import REPORTS_DIR


EXPORT_QUERY = """
    SELECT jsn, operator_result
    FROM piece_result
    WHERE jsn IS NOT NULL
      AND BTRIM(jsn) <> ''
      AND operator_result IN ('OK', 'NOK')
    ORDER BY operator_result ASC, jsn ASC
"""

JSNS_PER_PIECE = 4


def _jsn_sort_key(jsn: str) -> tuple[int, int | str, str]:
    """Sort numeric JSNs from oldest (lowest) to newest (highest)."""
    if jsn.isdigit():
        return (0, int(jsn), jsn)
    return (1, jsn, jsn)


def _build_piece_records(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Assign a piece/set number to every four JSNs in chronological order."""
    results_by_jsn: dict[str, str] = {}
    for row in rows:
        jsn = str(row.get("jsn") or "").strip()
        result = str(row.get("operator_result") or "").strip().upper()
        if jsn and result in {"OK", "NOK"}:
            results_by_jsn[jsn] = result

    records = []
    for index, jsn in enumerate(sorted(results_by_jsn, key=_jsn_sort_key)):
        records.append(
            {
                "jsn": jsn,
                "operator_result": results_by_jsn[jsn],
                # Four consecutive JSNs comprise one physical piece/set.
                "piece_number": (index // JSNS_PER_PIECE) + 1,
                "position_in_piece": (index % JSNS_PER_PIECE) + 1,
            }
        )
    piece_results = _piece_results(records)
    for record in records:
        record["piece_result"] = piece_results[int(record["piece_number"])]
    return records


def _piece_results(piece_records: Iterable[Mapping[str, object]]) -> dict[int, str]:
    """Return one result per four-JSN piece; NOK takes precedence within a set."""
    results: dict[int, str] = {}
    for record in piece_records:
        piece_number = int(record["piece_number"])
        operator_result = str(record.get("operator_result") or "").upper()
        if operator_result == "NOK":
            results[piece_number] = "NOK"
        elif piece_number not in results:
            results[piece_number] = "OK"
    return results


def _report_text(piece_records: Iterable[Mapping[str, object]]) -> str:
    """Format the operator results, including the derived four-JSN piece number."""
    piece_records = list(piece_records)
    piece_results = _piece_results(piece_records)
    grouped: dict[str, list[Mapping[str, object]]] = {"NOK": [], "OK": []}
    for record in piece_records:
        result = piece_results[int(record["piece_number"])]
        if result in grouped:
            grouped[result].append(record)

    lines = [
        "REPORTE DE JSN MARCADOS POR EL OPERADOR",
        "Cada bloque consecutivo de 4 JSN, del más antiguo al más reciente, es una pieza.",
        "Una pieza es NOK si al menos uno de sus 4 JSN está marcado como NOK.",
        "Formato: No. de pieza (set de 4) | Posición en el set | JSN | Resultado del JSN",
        "",
    ]
    for result in ("NOK", "OK"):
        records = grouped[result]
        piece_count = len({int(record["piece_number"]) for record in records})
        piece_label = "pieza" if piece_count == 1 else "piezas"
        lines.append(f"PIEZAS {result}: {piece_count} {piece_label} ({len(records)} JSN)")
        for record in records:
            lines.append(
                f"{record['piece_number']} | "
                f"{record['position_in_piece']}/{JSNS_PER_PIECE} | {record['jsn']} | "
                f"{record['operator_result']}"
            )
        lines.append("")
    return "\n".join(lines)


def export_operator_jsns(output_path: Path) -> dict[str, int]:
    """Read operator verdicts from PostgreSQL and write the TXT report."""
    db = get_db_connection()
    try:
        piece_records = _build_piece_records(db.fetch(EXPORT_QUERY))
    finally:
        db.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_report_text(piece_records), encoding="utf-8-sig")
    return {
        result: sum(1 for piece_result in _piece_results(piece_records).values() if piece_result == result)
        for result in ("NOK", "OK")
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genera un TXT con los JSN agrupados por resultado del operador."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "jsn_por_resultado_operador.txt",
        help="Ruta del TXT a generar (por defecto: reports/jsn_por_resultado_operador.txt).",
    )
    args = parser.parse_args()

    counts = export_operator_jsns(args.output)
    print(f"Reporte generado: {args.output.resolve()}")
    for result in ("NOK", "OK"):
        piece_label = "pieza" if counts[result] == 1 else "piezas"
        print(f"Piezas {result}: {counts[result]} {piece_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

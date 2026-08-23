import re


VALID_VERDICTS = ("OK", "NOK")
METRIC_KEYS = (
    "true_ok",
    "true_nok",
    "false_negative",
    "false_positive",
    "evaluated",
)


def normalize_verdict(value, allow_blank=True):
    """Normalize an operator verdict and reject unsupported values."""
    normalized = str(value or "").strip().upper()
    if not normalized and allow_blank:
        return ""
    if normalized not in VALID_VERDICTS:
        raise ValueError(f"Unsupported verdict: {value!r}. Use OK or NOK.")
    return normalized


def parse_actual_verdict_values(text):
    """Parse a pasted Excel-style column into normalized OK/NOK values."""
    raw_text = str(text or "").strip()
    if not raw_text:
        return []

    tokens = [token for token in re.split(r"[\s,;]+", raw_text) if token]
    return [normalize_verdict(token, allow_blank=False) for token in tokens]


def calculate_position_metrics(rows, positions=4):
    """Build per-position confusion counts using the application's legacy labels."""
    resolved_positions = max(1, int(positions or 4))
    metrics = {
        position: {metric_key: 0 for metric_key in METRIC_KEYS}
        for position in range(1, resolved_positions + 1)
    }

    for row in rows or []:
        try:
            actual = normalize_verdict(row.get("actual_result"), allow_blank=True)
        except (AttributeError, ValueError):
            continue
        if not actual:
            continue

        position_entries = row.get("positions") or []
        for fallback_position, entry in enumerate(position_entries, start=1):
            if not isinstance(entry, dict):
                continue
            try:
                position = int(entry.get("position") or fallback_position)
            except (TypeError, ValueError):
                continue
            if position not in metrics:
                continue

            inferred = str(entry.get("inferred_result") or "").strip().upper()
            if inferred not in VALID_VERDICTS:
                continue

            position_metrics = metrics[position]
            position_metrics["evaluated"] += 1
            if actual == "OK" and inferred == "OK":
                position_metrics["true_ok"] += 1
            elif actual == "NOK" and inferred == "NOK":
                position_metrics["true_nok"] += 1
            elif actual == "OK" and inferred == "NOK":
                position_metrics["false_negative"] += 1
            elif actual == "NOK" and inferred == "OK":
                position_metrics["false_positive"] += 1

    return metrics

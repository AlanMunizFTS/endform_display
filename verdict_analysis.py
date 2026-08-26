import re
import math
from bisect import bisect_left


VALID_VERDICTS = ("OK", "NOK")
METRIC_KEYS = (
    "true_ok",
    "true_nok",
    "false_negative",
    "false_positive",
    "evaluated",
)
DEFAULT_CONFIDENCE_THRESHOLD = 0.0
DEFAULT_REQUIRED_ANGLES = ("side", "diag")
CONFIDENCE_PRECISION = 4


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


def normalize_confidence_thresholds(thresholds=None, angles=DEFAULT_REQUIRED_ANGLES):
    """Return validated per-angle thresholds rounded to DB confidence precision."""
    source = thresholds if isinstance(thresholds, dict) else {}
    normalized = {}
    for raw_angle in angles or DEFAULT_REQUIRED_ANGLES:
        angle = str(raw_angle or "").strip().lower()
        if not angle:
            continue
        try:
            value = float(source.get(angle, DEFAULT_CONFIDENCE_THRESHOLD))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {angle.upper()} confidence threshold") from exc
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(
                f"{angle.upper()} confidence threshold must be between 0 and 1"
            )
        normalized[angle] = round(value, CONFIDENCE_PRECISION)
    return normalized


def infer_position_result(
    entry,
    thresholds=None,
    required_angles=DEFAULT_REQUIRED_ANGLES,
):
    """Infer OK/NOK from maximum confidences, or None when data is incomplete."""
    angles = tuple(
        str(angle or "").strip().lower()
        for angle in (required_angles or DEFAULT_REQUIRED_ANGLES)
        if str(angle or "").strip()
    )
    normalized_thresholds = normalize_confidence_thresholds(thresholds, angles)
    return _infer_position_result_normalized(entry, normalized_thresholds, angles)


def _infer_position_result_normalized(entry, thresholds, required_angles):
    if not isinstance(entry, dict) or not str(entry.get("jsn") or "").strip():
        return None
    if not entry.get("confidence_data_complete", False):
        return None

    confidence_by_angle = entry.get("max_confidence_by_angle") or {}
    for angle in required_angles:
        confidence = confidence_by_angle.get(angle)
        if confidence is None:
            continue
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            continue
        if confidence_value >= thresholds[angle]:
            return "NOK"
    return "OK"


def apply_confidence_thresholds(
    rows,
    thresholds=None,
    required_angles=DEFAULT_REQUIRED_ANGLES,
):
    """Recalculate inferred results in-place using one global threshold per angle."""
    angles = tuple(required_angles or DEFAULT_REQUIRED_ANGLES)
    normalized_thresholds = normalize_confidence_thresholds(thresholds, angles)
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        for entry in row.get("positions") or []:
            if not isinstance(entry, dict):
                continue
            entry["inferred_result"] = _infer_position_result_normalized(
                entry,
                normalized_thresholds,
                angles,
            )
    return normalized_thresholds


def _summarize_position_metrics(metrics):
    per_position = {}
    total_actual_ok = 0
    total_actual_nok = 0
    total_evaluated = 0
    total_false_positive = 0
    total_false_negative = 0

    for position, values in metrics.items():
        actual_ok = values["true_ok"] + values["false_negative"]
        actual_nok = values["true_nok"] + values["false_positive"]
        false_negative_rate = (
            values["false_negative"] / actual_ok if actual_ok else None
        )
        false_positive_rate = (
            values["false_positive"] / actual_nok if actual_nok else None
        )
        total_actual_ok += actual_ok
        total_actual_nok += actual_nok
        total_evaluated += values["evaluated"]
        total_false_positive += values["false_positive"]
        total_false_negative += values["false_negative"]
        per_position[position] = {
            **values,
            "actual_ok": actual_ok,
            "actual_nok": actual_nok,
            "false_negative_rate": false_negative_rate,
            "false_positive_rate": false_positive_rate,
        }

    return {
        "per_position": per_position,
        "average_false_negative_rate": (
            total_false_negative / total_evaluated
            if total_evaluated
            else None
        ),
        "average_false_positive_rate": (
            total_false_positive / total_evaluated
            if total_evaluated
            else None
        ),
        "total_actual_ok": total_actual_ok,
        "total_actual_nok": total_actual_nok,
        "total_evaluated": total_evaluated,
        "total_false_positive": total_false_positive,
        "total_false_negative": total_false_negative,
    }


def calculate_average_error_rates(rows, positions=4):
    """Calculate global legacy FP/FN rates over all evaluable pieces."""
    return _summarize_position_metrics(
        calculate_position_metrics(rows, positions=positions)
    )


def evaluate_confidence_thresholds(
    rows,
    thresholds=None,
    required_angles=DEFAULT_REQUIRED_ANGLES,
    positions=4,
):
    """Evaluate thresholds without mutating the analysis rows."""
    resolved_positions = max(1, int(positions or 4))
    metrics = {
        position: {metric_key: 0 for metric_key in METRIC_KEYS}
        for position in range(1, resolved_positions + 1)
    }
    angles = tuple(required_angles or DEFAULT_REQUIRED_ANGLES)
    normalized_thresholds = normalize_confidence_thresholds(thresholds, angles)

    for row in rows or []:
        try:
            actual = normalize_verdict(row.get("actual_result"), allow_blank=True)
        except (AttributeError, ValueError):
            continue
        if not actual:
            continue
        for fallback_position, entry in enumerate(row.get("positions") or [], start=1):
            if not isinstance(entry, dict):
                continue
            try:
                position = int(entry.get("position") or fallback_position)
            except (TypeError, ValueError):
                continue
            if position not in metrics:
                continue
            inferred = _infer_position_result_normalized(
                entry,
                normalized_thresholds,
                angles,
            )
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

    summary = _summarize_position_metrics(metrics)
    summary["thresholds"] = normalized_thresholds
    return summary


def _build_threshold_candidates(rows, angle):
    candidates = {0.0, 1.0}
    for row in rows or []:
        try:
            actual = normalize_verdict(row.get("actual_result"), allow_blank=True)
        except (AttributeError, ValueError):
            continue
        if not actual:
            continue
        for entry in row.get("positions") or []:
            if not isinstance(entry, dict) or not entry.get(
                "confidence_data_complete", False
            ):
                continue
            confidence = (entry.get("max_confidence_by_angle") or {}).get(angle)
            if confidence is None:
                continue
            try:
                value = round(float(confidence), CONFIDENCE_PRECISION)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                continue
            candidates.add(value)
            candidates.add(round(min(1.0, value + 0.0001), CONFIDENCE_PRECISION))
    return sorted(candidates)


def _first_true_index(size, predicate):
    low = 0
    high = size
    while low < high:
        middle = (low + high) // 2
        if predicate(middle):
            high = middle
        else:
            low = middle + 1
    return low if low < size else None


def _normalized_optimization_score(value):
    if value is None:
        return -1.0
    try:
        score = round(float(value), CONFIDENCE_PRECISION)
    except (TypeError, ValueError):
        return -1.0
    if not math.isfinite(score):
        return -1.0
    return score


def _collect_weighted_optimization_records(rows, positions, angles):
    resolved_positions = max(1, int(positions or 4))
    counts = {
        position: {"OK": 0, "NOK": 0}
        for position in range(1, resolved_positions + 1)
    }
    raw_records = []
    side_angle, diag_angle = angles

    for row in rows or []:
        try:
            actual = normalize_verdict(row.get("actual_result"), allow_blank=True)
        except (AttributeError, ValueError):
            continue
        if not actual:
            continue
        for fallback_position, entry in enumerate(row.get("positions") or [], start=1):
            if (
                not isinstance(entry, dict)
                or not str(entry.get("jsn") or "").strip()
                or not entry.get("confidence_data_complete", False)
            ):
                continue
            try:
                position = int(entry.get("position") or fallback_position)
            except (TypeError, ValueError):
                continue
            if position not in counts:
                continue
            confidence_by_angle = entry.get("max_confidence_by_angle") or {}
            raw_records.append(
                (
                    _normalized_optimization_score(
                        confidence_by_angle.get(side_angle)
                    ),
                    _normalized_optimization_score(
                        confidence_by_angle.get(diag_angle)
                    ),
                    actual,
                    position,
                )
            )
            counts[position][actual] += 1

    record_weight = 1.0 / len(raw_records) if raw_records else 0.0
    weighted_records = [
        (side_score, diag_score, actual, record_weight)
        for side_score, diag_score, actual, _position in raw_records
    ]
    return weighted_records, counts


def _collect_single_angle_optimization_records(rows, positions, angle):
    resolved_positions = max(1, int(positions or 4))
    counts = {
        position: {"OK": 0, "NOK": 0}
        for position in range(1, resolved_positions + 1)
    }
    raw_records = []

    for row in rows or []:
        try:
            actual = normalize_verdict(row.get("actual_result"), allow_blank=True)
        except (AttributeError, ValueError):
            continue
        if not actual:
            continue
        for fallback_position, entry in enumerate(row.get("positions") or [], start=1):
            if (
                not isinstance(entry, dict)
                or not str(entry.get("jsn") or "").strip()
                or not entry.get("confidence_data_complete", False)
            ):
                continue
            try:
                position = int(entry.get("position") or fallback_position)
            except (TypeError, ValueError):
                continue
            if position not in counts:
                continue
            confidence_by_angle = entry.get("max_confidence_by_angle") or {}
            raw_records.append(
                (
                    _normalized_optimization_score(confidence_by_angle.get(angle)),
                    actual,
                )
            )
            counts[position][actual] += 1

    record_weight = 1.0 / len(raw_records) if raw_records else 0.0
    return [
        (score, actual, record_weight)
        for score, actual in raw_records
    ], counts


def _optimize_single_confidence_threshold(
    rows,
    angle,
    false_negative_target,
    positions,
):
    candidates = _build_threshold_candidates(rows, angle)
    records, counts = _collect_single_angle_optimization_records(
        rows,
        positions,
        angle,
    )
    total_actual_ok = sum(values["OK"] for values in counts.values())
    total_actual_nok = sum(values["NOK"] for values in counts.values())
    if total_actual_ok <= 0 or total_actual_nok <= 0:
        raise ValueError(
            "Enter both OK and NOK actual values before finding the best point"
        )

    records.sort(key=lambda record: record[0])
    record_index = 0
    true_ok_rate = 0.0
    false_positive_rate = 0.0
    actual_ok_rate = total_actual_ok / (total_actual_ok + total_actual_nok)
    epsilon = 1e-12
    minimum_fp = None
    minimum_fp_candidates = []

    for threshold in candidates:
        while record_index < len(records) and records[record_index][0] < threshold:
            _score, actual, weight = records[record_index]
            if actual == "OK":
                true_ok_rate += weight
            else:
                false_positive_rate += weight
            record_index += 1

        candidate = {
            "threshold": threshold,
            "average_false_negative_rate": min(
                1.0,
                max(0.0, actual_ok_rate - true_ok_rate),
            ),
            "average_false_positive_rate": min(
                1.0,
                max(0.0, false_positive_rate),
            ),
        }
        if minimum_fp is None:
            minimum_fp = candidate["average_false_positive_rate"]
        elif candidate["average_false_positive_rate"] > minimum_fp + epsilon:
            break
        minimum_fp_candidates.append(candidate)

    best_candidate = min(
        minimum_fp_candidates,
        key=lambda candidate: (
            round(candidate["average_false_negative_rate"], 12),
            -candidate["threshold"],
        ),
    )
    best = evaluate_confidence_thresholds(
        rows,
        {angle: best_candidate["threshold"]},
        required_angles=(angle,),
        positions=positions,
    )
    target_met = best["average_false_negative_rate"] < false_negative_target
    return {
        **best,
        "target_met": target_met,
        "message": (
            "Best point minimizes false positives and meets the false-negative target."
            if target_met
            else "Best point minimizes false positives; the false-negative target is not met."
        ),
    }


def optimize_confidence_thresholds(
    rows,
    required_angles=DEFAULT_REQUIRED_ANGLES,
    false_negative_target=0.10,
    positions=4,
):
    """Find the best threshold for one angle or threshold pair for two angles."""
    angles = tuple(
        str(angle or "").strip().lower()
        for angle in (required_angles or DEFAULT_REQUIRED_ANGLES)
        if str(angle or "").strip()
    )
    if len(angles) not in (1, 2):
        raise ValueError("Confidence optimization requires one or two angles")
    try:
        resolved_target = float(false_negative_target)
    except (TypeError, ValueError) as exc:
        raise ValueError("False-negative target must be between 0 and 1") from exc
    if (
        not math.isfinite(resolved_target)
        or resolved_target <= 0.0
        or resolved_target > 1.0
    ):
        raise ValueError("False-negative target must be between 0 and 1")

    if len(angles) == 1:
        return _optimize_single_confidence_threshold(
            rows,
            angles[0],
            resolved_target,
            positions,
        )

    side_angle, diag_angle = angles
    side_candidates = _build_threshold_candidates(rows, side_angle)
    diag_candidates = _build_threshold_candidates(rows, diag_angle)
    records, counts = _collect_weighted_optimization_records(
        rows,
        positions,
        angles,
    )
    total_actual_ok = sum(values["OK"] for values in counts.values())
    total_actual_nok = sum(values["NOK"] for values in counts.values())
    if total_actual_ok <= 0 or total_actual_nok <= 0:
        raise ValueError(
            "Enter both OK and NOK actual values before finding the best point"
        )
    total_evaluated = total_actual_ok + total_actual_nok
    actual_ok_rate = total_actual_ok / total_evaluated

    diag_scores = sorted({record[1] for record in records})
    ok_tree = [0.0] * (len(diag_scores) + 1)
    nok_tree = [0.0] * (len(diag_scores) + 1)

    def add_weight(tree, score, weight):
        index = bisect_left(diag_scores, score) + 1
        while index < len(tree):
            tree[index] += weight
            index += index & -index

    def prefix_weight(tree, threshold):
        index = bisect_left(diag_scores, threshold)
        total = 0.0
        while index > 0:
            total += tree[index]
            index -= index & -index
        return total

    records.sort(key=lambda record: record[0])
    record_index = 0
    epsilon = 1e-12
    minimum_fp = None
    minimum_fp_candidates = []
    for side_threshold in side_candidates:
        while (
            record_index < len(records)
            and records[record_index][0] < side_threshold
        ):
            _side_score, diag_score, actual, weight = records[record_index]
            add_weight(
                ok_tree if actual == "OK" else nok_tree,
                diag_score,
                weight,
            )
            record_index += 1

        rate_cache = {}

        def rates_at(index):
            if index not in rate_cache:
                diag_threshold = diag_candidates[index]
                true_ok_rate = prefix_weight(ok_tree, diag_threshold)
                rate_cache[index] = {
                    "thresholds": {
                        side_angle: side_threshold,
                        diag_angle: diag_threshold,
                    },
                    "average_false_negative_rate": min(
                        1.0,
                        max(0.0, actual_ok_rate - true_ok_rate),
                    ),
                    "average_false_positive_rate": min(
                        1.0,
                        max(0.0, prefix_weight(nok_tree, diag_threshold)),
                    ),
                }
            return rate_cache[index]

        side_minimum_fp = rates_at(0)["average_false_positive_rate"]
        if minimum_fp is None:
            minimum_fp = side_minimum_fp
        elif side_minimum_fp > minimum_fp + epsilon:
            break

        first_higher_fp = _first_true_index(
            len(diag_candidates),
            lambda index: rates_at(index)["average_false_positive_rate"]
            > minimum_fp + epsilon,
        )
        best_diag_index = (
            len(diag_candidates) - 1
            if first_higher_fp is None
            else first_higher_fp - 1
        )
        minimum_fp_candidates.append(rates_at(best_diag_index))

    def best_point_key(result):
        thresholds = result["thresholds"]
        side_threshold = thresholds[side_angle]
        diag_threshold = thresholds[diag_angle]
        return (
            round(result["average_false_negative_rate"], 12),
            -(side_threshold + diag_threshold),
            -side_threshold,
            -diag_threshold,
        )

    best_thresholds = min(minimum_fp_candidates, key=best_point_key)["thresholds"]
    best = evaluate_confidence_thresholds(
        rows,
        best_thresholds,
        required_angles=angles,
        positions=positions,
    )
    target_met = best["average_false_negative_rate"] < resolved_target
    return {
        **best,
        "target_met": target_met,
        "message": (
            "Best point minimizes false positives and meets the false-negative target."
            if target_met
            else "Best point minimizes false positives; the false-negative target is not met."
        ),
    }

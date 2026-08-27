"""Inspect metadata and architecture embedded in an Ultralytics ``.pt`` model.

This utility is intentionally read-only with respect to the model.  It loads the
checkpoint and the model object, but it does not run inference, train, export, or
modify any files unless ``--output`` is explicitly supplied.

Examples::

    python utilities/inspect_model.py models/best_side_edge.pt
    python utilities/inspect_model.py models/best_side_edge.pt --json --output model_info.json

The JSON report contains the complete serializable training arguments, metrics,
training history, model YAML, and checkpoint inventory.  It does not serialize
weight tensors or the full Python object because those are both very large and
not useful as metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "best_side_edge.pt"


def _json_value(value: Any) -> Any:
    """Convert common Torch/NumPy/checkpoint values to JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]

    # Torch tensors and NumPy scalars/arrays expose one or both of these APIs.
    if hasattr(value, "detach"):
        try:
            return _json_value(value.detach().cpu().tolist())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_value(value.tolist())
        except Exception:
            pass

    # A few Ultralytics values are argparse namespaces or custom scalar types.
    if hasattr(value, "__dict__") and not isinstance(value, type):
        try:
            return _json_value(vars(value))
        except Exception:
            pass

    return repr(value)


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _iso_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().isoformat()


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> Any:
    """Load a checkpoint in a way that works across supported Torch versions."""
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Torch versions before the weights_only argument.
        return torch.load(path, map_location="cpu")


def _normalise_names(names: Any) -> dict[str, str]:
    if isinstance(names, Mapping):
        return {str(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {str(index): str(value) for index, value in enumerate(names)}
    return {}


def _detect_task(yolo: Any, model: Any, checkpoint: Any) -> tuple[str, str]:
    task = _safe_attr(yolo, "task")
    if isinstance(task, str) and task:
        return task, "Ultralytics YOLO task metadata"

    class_names = " ".join(
        type(candidate).__name__.lower()
        for candidate in (model, _safe_attr(model, "model"))
        if candidate is not None
    )
    for token, task_name in (
        ("obb", "obb"),
        ("segment", "segment"),
        ("pose", "pose"),
        ("classif", "classify"),
        ("detect", "detect"),
    ):
        if token in class_names:
            return task_name, "Model class name"

    if isinstance(checkpoint, Mapping):
        args = checkpoint.get("train_args")
        if isinstance(args, Mapping) and args.get("task"):
            return str(args["task"]), "Checkpoint train_args"
    return "unknown", "Not available in checkpoint"


def _resolve_reference(reference: Any, model_path: Path) -> dict[str, Any] | None:
    if not reference or not isinstance(reference, (str, Path)):
        return None

    raw = Path(str(reference))
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend(
            [
                Path.cwd() / raw,
                model_path.parent / raw,
                model_path.parent.parent / raw,
            ]
        )

    existing = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    return {
        "declared": str(reference),
        "resolved": str(existing) if existing else None,
        "exists": existing is not None,
    }


def _layer_table(model_yaml: Any) -> list[dict[str, Any]]:
    if not isinstance(model_yaml, Mapping):
        return []

    rows: list[dict[str, Any]] = []
    for section in ("backbone", "head"):
        layers = model_yaml.get(section)
        if not isinstance(layers, list):
            continue
        for index, layer in enumerate(layers):
            if not isinstance(layer, (list, tuple)) or len(layer) < 4:
                rows.append({"section": section, "index": index, "raw": _json_value(layer)})
                continue
            rows.append(
                {
                    "section": section,
                    "index": index,
                    "from": _json_value(layer[0]),
                    "repeats": _json_value(layer[1]),
                    "module": str(layer[2]),
                    "args": _json_value(layer[3]),
                }
            )
    return rows


def _model_statistics(model: Any) -> dict[str, Any]:
    parameters = list(model.parameters()) if hasattr(model, "parameters") else []
    buffers = list(model.buffers()) if hasattr(model, "buffers") else []
    modules = list(model.modules()) if hasattr(model, "modules") else []
    state_dict = model.state_dict() if hasattr(model, "state_dict") else {}

    return {
        "parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "buffers": sum(buffer.numel() for buffer in buffers),
        "state_dict_entries": len(state_dict),
        "module_count": len(modules),
        "module_types": dict(sorted(Counter(type(module).__name__ for module in modules).items())),
        "parameter_dtypes": dict(
            sorted(Counter(str(parameter.dtype) for parameter in parameters).items())
        ),
    }


def _history_summary(history: Any) -> dict[str, Any]:
    if not isinstance(history, Mapping):
        return {"available": False}

    summary: dict[str, Any] = {
        "available": True,
        "series": sorted(str(key) for key in history),
        "record_count": 0,
        "last_values": {},
    }
    for key, values in history.items():
        if isinstance(values, (list, tuple)):
            summary["record_count"] = max(summary["record_count"], len(values))
            if values:
                summary["last_values"][str(key)] = _json_value(values[-1])
    return summary


def inspect_model(path: str | Path, include_hash: bool = True) -> dict[str, Any]:
    """Return a JSON-serializable report for an Ultralytics checkpoint."""
    model_path = Path(path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"No existe el archivo del modelo: {model_path}")
    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"Se esperaba un archivo .pt: {model_path}")

    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Faltan dependencias. Instala requirements.txt o activa el venv del proyecto."
        ) from exc

    checkpoint = _load_checkpoint(model_path)
    if not isinstance(checkpoint, Mapping):
        checkpoint_keys: list[str] = []
        checkpoint_types = {"root": type(checkpoint).__name__}
    else:
        checkpoint_keys = [str(key) for key in checkpoint.keys()]
        checkpoint_types = {
            str(key): type(value).__name__ for key, value in checkpoint.items()
        }

    try:
        yolo = YOLO(str(model_path))
    except Exception as exc:
        raise RuntimeError(f"Ultralytics no pudo cargar el modelo: {exc}") from exc

    model = _safe_attr(yolo, "model")
    if model is None:
        raise RuntimeError("El archivo se cargó, pero no expuso un objeto model de Ultralytics.")

    task, task_source = _detect_task(yolo, model, checkpoint)
    train_args = checkpoint.get("train_args", {}) if isinstance(checkpoint, Mapping) else {}
    if not isinstance(train_args, Mapping):
        train_args = _json_value(train_args)
        train_args = train_args if isinstance(train_args, Mapping) else {}

    model_yaml = _safe_attr(model, "yaml", {})
    names = _normalise_names(_safe_attr(yolo, "names", _safe_attr(model, "names", {})))
    declared_model = train_args.get("model")
    pretrained = train_args.get("pretrained")
    if isinstance(pretrained, str):
        pretrained_value: Any = pretrained.lower() in {"1", "true", "yes", "on"}
    else:
        pretrained_value = pretrained

    references = {
        "training_dataset": _resolve_reference(train_args.get("data"), model_path),
        "initial_model": _resolve_reference(declared_model, model_path),
        "save_dir": _resolve_reference(train_args.get("save_dir"), model_path),
    }

    stat = model_path.stat()
    file_info: dict[str, Any] = {
        "path": str(model_path),
        "name": model_path.name,
        "size_bytes": stat.st_size,
        "size_mib": round(stat.st_size / (1024 * 1024), 3),
        "created": _iso_timestamp(stat.st_ctime),
        "modified": _iso_timestamp(stat.st_mtime),
        "accessed": _iso_timestamp(stat.st_atime),
    }
    if include_hash:
        file_info["sha256"] = _sha256(model_path)

    report: dict[str, Any] = {
        "report": {
            "generated_at": datetime.now().astimezone().isoformat(),
            "tool": "utilities/inspect_model.py",
            "read_only": True,
        },
        "file": file_info,
        "framework": {
            "framework": "Ultralytics YOLO",
            "ultralytics_installed": getattr(ultralytics, "__version__", None),
            "checkpoint_version": checkpoint.get("version") if isinstance(checkpoint, Mapping) else None,
            "torch": getattr(torch, "__version__", None),
            "python": platform.python_version(),
        },
        "task": {
            "type": task,
            "type_source": task_source,
            "geometry": {
                "obb": task == "obb",
                "segmentation": task in {"segment", "segmentation"},
                "axis_aligned_boxes": task == "detect",
                "pose": task == "pose",
                "classification": task in {"classify", "classification"},
            },
        },
        "model": {
            "class": type(model).__name__,
            "module": type(model).__module__,
            "yaml_file": _safe_attr(model, "yaml", {}).get("yaml_file")
            if isinstance(_safe_attr(model, "yaml", {}), Mapping)
            else None,
            "names": names,
            "number_of_classes": len(names) or _json_value(_safe_attr(model, "nc")),
            "channels": _json_value(
                _safe_attr(model, "yaml", {}).get("channels")
                if isinstance(_safe_attr(model, "yaml", {}), Mapping)
                else None
            ),
            "stride": _json_value(_safe_attr(model, "stride")),
            "statistics": _model_statistics(model),
            "yaml": _json_value(model_yaml),
            "layers": _layer_table(model_yaml),
        },
        "input_and_inference": {
            "training_imgsz": _json_value(train_args.get("imgsz")),
            "confidence_threshold": _json_value(train_args.get("conf")),
            "iou_threshold": _json_value(train_args.get("iou")),
            "max_detections": _json_value(train_args.get("max_det")),
            "device_used_for_training": _json_value(train_args.get("device")),
            "half_precision": _json_value(train_args.get("half")),
            "amp": _json_value(train_args.get("amp")),
        },
        "training": {
            "checkpoint_date": checkpoint.get("date") if isinstance(checkpoint, Mapping) else None,
            "epoch": checkpoint.get("epoch") if isinstance(checkpoint, Mapping) else None,
            "best_fitness": checkpoint.get("best_fitness") if isinstance(checkpoint, Mapping) else None,
            "train_args": _json_value(train_args),
            "metrics": _json_value(
                checkpoint.get("train_metrics", {}) if isinstance(checkpoint, Mapping) else {}
            ),
            "history_summary": _history_summary(
                checkpoint.get("train_results", {}) if isinstance(checkpoint, Mapping) else {}
            ),
            "history": _json_value(
                checkpoint.get("train_results", {}) if isinstance(checkpoint, Mapping) else {}
            ),
            "state": {
                "optimizer_embedded": bool(
                    isinstance(checkpoint, Mapping) and checkpoint.get("optimizer") is not None
                ),
                "ema_embedded": bool(
                    isinstance(checkpoint, Mapping) and checkpoint.get("ema") is not None
                ),
                "scaler_embedded": bool(
                    isinstance(checkpoint, Mapping) and checkpoint.get("scaler") is not None
                ),
                "updates": _json_value(
                    checkpoint.get("updates") if isinstance(checkpoint, Mapping) else None
                ),
            },
        },
        "pretraining_and_provenance": {
            "pretrained_flag": pretrained_value,
            "declared_initial_model": _json_value(declared_model),
            "initial_model_reference": references["initial_model"],
            "interpretation": (
                "El checkpoint declara que se entrenó con pesos preentrenados; "
                "el archivo inicial no está embebido, por lo que no se puede verificar "
                "su hash ni cuánto se conservaron esos pesos."
                if pretrained_value
                else "El checkpoint no declara uso de pesos preentrenados."
            ),
        },
        "references": references,
        "git": _json_value(checkpoint.get("git", {}) if isinstance(checkpoint, Mapping) else {}),
        "checkpoint": {
            "root_type": type(checkpoint).__name__,
            "keys": checkpoint_keys,
            "value_types": checkpoint_types,
            "license": checkpoint.get("license") if isinstance(checkpoint, Mapping) else None,
            "docs": checkpoint.get("docs") if isinstance(checkpoint, Mapping) else None,
        },
    }
    return _json_value(report)


def _format_value(value: Any) -> str:
    if value is None:
        return "no disponible"
    if isinstance(value, bool):
        return "sí" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _print_mapping(title: str, values: Mapping[str, Any], indent: str = "  ") -> None:
    print(title)
    for key, value in values.items():
        if isinstance(value, (Mapping, list)):
            continue
        print(f"{indent}{key}: {_format_value(value)}")


def print_human(report: Mapping[str, Any], full: bool = False) -> None:
    """Print a compact Spanish report; ``--full`` includes raw nested sections."""
    file_info = report["file"]
    framework = report["framework"]
    task = report["task"]
    model = report["model"]
    training = report["training"]
    provenance = report["pretraining_and_provenance"]

    print("=== INSPECCIÓN DE MODELO ===")
    print(f"Archivo: {file_info['path']}")
    print(f"Tamaño: {file_info['size_mib']} MiB")
    if file_info.get("sha256"):
        print(f"SHA-256: {file_info['sha256']}")
    print(f"Creado (metadato del archivo): {file_info['created']}")
    print(f"Modificado: {file_info['modified']}")

    print("\nFramework y tarea")
    print(f"  Framework: {framework['framework']}")
    print(f"  Ultralytics del entorno: {_format_value(framework['ultralytics_installed'])}")
    print(f"  Versión guardada en checkpoint: {_format_value(framework['checkpoint_version'])}")
    print(f"  Tarea: {task['type']} ({task['type_source']})")
    print(f"  Geometría: {_format_value(task['geometry'])}")

    print("\nModelo y arquitectura")
    print(f"  Clase: {model['module']}.{model['class']}")
    print(f"  YAML: {_format_value(model['yaml_file'])}")
    print(f"  Clases ({model['number_of_classes']}): {model['names']}")
    print(f"  Canales de entrada: {_format_value(model['channels'])}")
    print(f"  Stride: {_format_value(model['stride'])}")
    for key, value in model["statistics"].items():
        print(f"  {key}: {_format_value(value)}")

    print("\nEntrenamiento")
    print(f"  Fecha interna del checkpoint: {_format_value(training['checkpoint_date'])}")
    print(f"  Epoch guardado: {_format_value(training['epoch'])}")
    print(f"  Mejor fitness: {_format_value(training['best_fitness'])}")
    print(f"  Dataset declarado: {_format_value(report['references']['training_dataset'])}")
    print(f"  Parámetros: {len(training['train_args'])}")
    print(f"  Métricas finales guardadas: {training['metrics']}")
    print(f"  Historial: {training['history_summary']}")
    print(f"  Optimizer embebido: {_format_value(training['state']['optimizer_embedded'])}")

    print("\nPreentrenamiento y procedencia")
    print(f"  pretrained: {_format_value(provenance['pretrained_flag'])}")
    print(f"  Modelo inicial declarado: {_format_value(provenance['declared_initial_model'])}")
    print(f"  Git: {report['git']}")
    print(f"  Nota: {provenance['interpretation']}")

    print("\nPara ver todos los parámetros, YAML e historial: usa --json o --full.")
    if full:
        print("\n=== DATOS COMPLETOS ===")
        print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspecciona metadatos, arquitectura y entrenamiento de un modelo Ultralytics .pt."
    )
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Ruta al .pt (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--json", action="store_true", help="Imprime el informe completo en JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Guarda el informe JSON en este archivo (activa formato JSON).",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="No calcula SHA-256; útil para archivos muy grandes.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="En salida humana, agrega YAML, argumentos e historial completos.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = inspect_model(args.model, include_hash=not args.no_hash)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Informe JSON guardado en: {output_path}")

    if args.json or args.output:
        if not args.output:
            print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report, full=args.full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

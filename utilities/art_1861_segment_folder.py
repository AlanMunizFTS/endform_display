import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO  # type: ignore
from ultralytics.engine.results import Results  # type: ignore


SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
DEFAULT_IMAGE_SIZE = 640
DEFAULT_CONFIDENCE = 0.01
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "best_seg_double_flare.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Segmenta la pieza principal usando el modelo de art_1861_endform y "
            "guarda una imagen procesada por cada imagen encontrada."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Carpeta con imagenes crudas")
    parser.add_argument("output_dir", type=Path, help="Carpeta donde se guardaran las salidas")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Ruta al modelo de segmentacion. Default: %(default)s",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Tamano final cuadrado de salida. Default: %(default)s",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Confidence threshold para YOLO. Default: %(default)s",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Dispositivo para inferencia, por ejemplo cpu, 0, 0,1. Default: %(default)s",
    )
    return parser.parse_args()


def collect_images(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def pad_to_square(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    max_dim = max(h, w)

    padded = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)

    x_offset = (max_dim - w) // 2
    y_offset = (max_dim - h) // 2
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = image

    return cv2.resize(padded, (size, size))


def extract_main_segment(
    segmentation_results: List[Results],
    original_image: np.ndarray,
    img_size: int,
) -> Optional[np.ndarray]:
    best_crop: Optional[np.ndarray] = None
    best_confidence = -1.0

    for result in segmentation_results:
        if result.boxes is None or result.masks is None or result.boxes.conf is None:
            continue

        masks_xy = result.masks.xy
        confidences = result.boxes.conf.cpu().numpy()
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()

        for index, confidence in enumerate(confidences):
            if index >= len(masks_xy):
                continue

            contour = np.asarray(masks_xy[index], dtype=np.int32).reshape(-1, 1, 2)
            if contour.size == 0:
                continue

            b_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
            cv2.drawContours(b_mask, [contour], -1, 255, cv2.FILLED)

            mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
            isolated = cv2.bitwise_and(original_image, mask3ch)

            x1, y1, x2, y2 = boxes_xyxy[index].astype(np.int32)
            crop = isolated[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            if float(confidence) > best_confidence:
                best_confidence = float(confidence)
                best_crop = pad_to_square(crop, size=img_size)

    return best_crop


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    return image


def build_output_path(input_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative_path = input_path.relative_to(input_dir)
    return output_dir / relative_path.parent / f"{relative_path.stem}_segmented.png"


def process_image(
    model: YOLO,
    image_path: Path,
    input_dir: Path,
    output_dir: Path,
    img_size: int,
    conf: float,
    device: str,
) -> Tuple[bool, str]:
    image = load_image(image_path)
    results = model.predict(image, conf=conf, device=device, verbose=False)
    segmented = extract_main_segment(results, image, img_size=img_size)

    if segmented is None:
        return False, f"Sin segmentacion: {image_path}"

    output_path = build_output_path(image_path, input_dir, output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), segmented):
        raise IOError(f"No se pudo guardar la imagen de salida: {output_path}")

    return True, f"Guardada: {output_path}"


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_path = args.model.resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"La carpeta de entrada no existe o no es valida: {input_dir}")

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontro el modelo de segmentacion: {model_path}")

    image_paths = list(collect_images(input_dir))
    if not image_paths:
        raise FileNotFoundError(f"No se encontraron imagenes en: {input_dir}")

    print(f"Cargando modelo: {model_path}")
    model = YOLO(str(model_path))

    processed_count = 0
    skipped_count = 0

    for image_path in image_paths:
        try:
            ok, message = process_image(
                model=model,
                image_path=image_path,
                input_dir=input_dir,
                output_dir=output_dir,
                img_size=args.img_size,
                conf=args.conf,
                device=args.device,
            )
            print(message)
            if ok:
                processed_count += 1
            else:
                skipped_count += 1
        except Exception as exc:
            skipped_count += 1
            print(f"Error con {image_path}: {exc}")

    print(
        "Proceso terminado. "
        f"Segmentadas: {processed_count}. "
        f"Sin salida o con error: {skipped_count}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

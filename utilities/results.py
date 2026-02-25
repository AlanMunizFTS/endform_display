import os
import shutil
from ultralytics import YOLO
import cv2


classified = "./classified"
reclassified = "./prueba2"
models_folder = "./models"
results_file = "results.txt"
valid_image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_subfolders(base_dir):
    if not os.path.isdir(base_dir):
        return []

    return sorted(
        [
            os.path.join(base_dir, folder)
            for folder in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, folder))
        ]
    )


def reset_results_file(file_path):
    with open(file_path, "w"):
        pass


def write_inference_result(file_path, crop_name, result, position):
    lines = [
        "=== New Inference ===\n",
        f"Name: {crop_name}\n",
        f"Position: {position}\n",
    ]
    detection_count = 0

    if position == "front":
        has_front_outputs = (
            result.boxes is not None
            and result.boxes.conf is not None
            and len(result.boxes.conf) > 0
            and result.masks is not None
            and result.masks.xyn is not None
            and len(result.masks.xyn) > 0
        )
        if not has_front_outputs:
            return

        n = min(len(result.boxes.conf), len(result.masks.xyn))
        for i in range(n):
            class_id = int(result.boxes.cls[i])
            confidence = float(result.boxes.conf[i])
            class_name = result.names[class_id]
            mask_points = result.masks.xyn[i].tolist()

            lines.append(f"Detection {i+1}\n")
            lines.append(f"Class: {class_name}\n")
            lines.append(f"Confidence: {confidence:.6f}\n")
            lines.append(f"Mask points: {mask_points}\n")
            lines.append("-" * 40 + "\n")
            detection_count += 1
    else:
        if result.obb is None or len(result.obb.cls) == 0:
            return

        obb = result.obb
        for i in range(len(obb.cls)):
            class_id = int(obb.cls[i])
            confidence = float(obb.conf[i])
            coordinates = obb.xyxyxyxy[i].tolist()
            class_name = result.names[class_id]

            lines.append(f"Detection {i+1}\n")
            lines.append(f"Class: {class_name}\n")
            lines.append(f"Confidence: {confidence:.6f}\n")
            lines.append(f"Coordinates: {coordinates}\n")
            lines.append("-" * 40 + "\n")
            detection_count += 1

    if detection_count == 0:
        return

    lines.append("\n")
    with open(file_path, "a") as f:
        f.writelines(lines)


def update_progress(processed, total, bar_length=30):
    if total <= 0:
        return

    ratio = processed / total
    filled = int(bar_length * ratio)
    bar = "#" * filled + "-" * (bar_length - filled)
    print(
        f"\rProgress: [{bar}] {processed}/{total} images processed",
        end="",
        flush=True,
    )
    if processed == total:
        print()


def is_valid_image_file(image_path):
    if not os.path.isfile(image_path):
        return False
    if os.path.splitext(image_path)[1].lower() not in valid_image_extensions:
        return False
    if os.path.getsize(image_path) == 0:
        return False
    if cv2.imread(image_path) is None:
        return False
    return True


def has_high_confidence_detection(result, position, confidence_threshold):
    if position == "front":
        if (
            result.boxes is None
            or result.boxes.conf is None
            or len(result.boxes.conf) == 0
            or result.masks is None
            or result.masks.xyn is None
            or len(result.masks.xyn) == 0
        ):
            return False
        return any(float(conf) > confidence_threshold for conf in result.boxes.conf)

    if result.obb is None or len(result.obb.conf) == 0:
        return False
    return any(float(conf) > confidence_threshold for conf in result.obb.conf)

models = [YOLO(os.path.join(models_folder, file)) for file in os.listdir(models_folder) if file.endswith(".pt")]

confidence_threshold = 0.4

classified_folders = list_subfolders(classified)
reclassified_folders = list_subfolders(reclassified)
folders_to_process = classified_folders + reclassified_folders

positions = ["side", "front", "diag"]

classified_folders = [
    folder
    for folder in classified_folders
    if any(pos in os.path.basename(folder).lower() for pos in positions)
]

processable_folders = []
for folder in classified_folders:
    position = None
    for pos in positions:
        if pos in folder.lower():
            position = pos
            break

    model = None
    for m in models:
        if position in m.model_name.lower():
            model = m
            break

    if model is not None:
        processable_folders.append((folder, position, model))

existing_reclassified_images = {
    img
    for folder in reclassified_folders
    for img in os.listdir(folder)
    if os.path.isfile(os.path.join(folder, img))
}

seen_image_names = set(existing_reclassified_images)
folder_jobs = []
for folder, position, model in processable_folders:
    images = []
    for img in os.listdir(folder):
        image_path = os.path.join(folder, img)
        if img in seen_image_names:
            continue
        if not is_valid_image_file(image_path):
            print(f"Skipping invalid image: {image_path}")
            continue

        images.append(img)
        seen_image_names.add(img)

    if images:
        folder_jobs.append((folder, position, model, images))

reset_results_file(results_file)

total_images = sum(len(images) for _, _, _, images in folder_jobs)
processed_images = 0

print(f"Total images to process: {total_images}")



for index, (folder, position, model, images) in enumerate(folder_jobs):
    ok_folder = os.path.join(reclassified, f"{position}_ok")
    nok_folder = os.path.join(reclassified, f"{position}_nok")

    for img in images:
        image_path = os.path.join(folder, img)
        try:
            results = model(image_path, verbose=False)
            result = results[0]
            write_inference_result(results_file, img, result, position)
        except Exception as e:
            print(f"\nSkipping {image_path}: inference error ({e})")
            processed_images += 1
            update_progress(processed_images, total_images)
            continue

        has_high_conf = has_high_confidence_detection(
            result,
            position,
            confidence_threshold,
        )

        if has_high_conf:
            annotated = result.plot()
            cv2.imwrite(
                os.path.join(nok_folder, img),
                annotated
            )
        else:
            cv2.imwrite(
                os.path.join(ok_folder, img),
                result.orig_img
            )

        existing_reclassified_images.add(img)
        processed_images += 1
        update_progress(processed_images, total_images)

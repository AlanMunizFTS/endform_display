import time

from paths_config import TMP_DISPLAY_DIR
from utilities.log import get_logger, install_print_logger


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _display_sort_key(filename):
    lower_name = filename.lower()
    if "side" in lower_name:
        return (0, filename)
    if "front" in lower_name:
        return (1, filename)
    if "diag" in lower_name:
        return (2, filename)
    return (3, filename)


def _download_live_images_local(file_manager, local_path, rotation_state, logger, max_images=7):
    file_manager.makedirs(local_path, exist_ok=True)
    downloaded_files = []

    try:
        images = []
        for name in file_manager.listdir(local_path):
            path = file_manager.join(local_path, name)
            if not file_manager.is_file(path):
                continue
            if not name.lower().endswith(IMAGE_EXTENSIONS):
                continue
            images.append(name)

        # Most recent first by filename ordering used in production.
        images.sort(reverse=True)

        total_batches = (len(images) + max_images - 1) // max_images
        start_idx = rotation_state["current_offset"] * max_images
        end_idx = start_idx + max_images
        selected_images = images[start_idx:end_idx]

        rotation_state["current_offset"] = (
            (rotation_state["current_offset"] + 1) % total_batches if total_batches > 0 else 0
        )

        selected_images.sort(key=_display_sort_key)
        for img_name in selected_images:
            downloaded_files.append(file_manager.join(local_path, img_name))

        return downloaded_files
    except Exception as e:
        logger.error(f"[LOCAL] Error loading live images: {e}", allow_repeat=True)
        return []


def main():
    install_print_logger(reset=True)
    logger = get_logger()

    from display_window import DisplayWindow

    temp_dir = str(TMP_DISPLAY_DIR)
    live_rotation_state = {"current_offset": 0}
    logger.info("[LOCAL] Running in local-only mode (SFTP disabled)", allow_repeat=True)

    try:
        display = DisplayWindow(
            width=1920,
            height=1080,
            window_name="Display Imagenes",
            refresh_interval=0.25,
        )
        logger.info("[DB] DisplayWindow initialized successfully", allow_repeat=True)
    except Exception as e:
        logger.error(f"[DB] Failed to initialize DisplayWindow: {e}", allow_repeat=True)
        return

    # In local-only mode this just ensures the folder exists.
    display.start_historic_download_on_startup(temp_dir, check_interval=10)

    try:
        while True:
            if display.exit_requested:
                break

            if display.historic_mode:
                images = display.download_historic_batch(temp_dir, max_images=7)
            else:
                images = _download_live_images_local(
                    display.file_manager,
                    temp_dir,
                    live_rotation_state,
                    logger,
                    max_images=7,
                )

            display.image_paths = images
            display.show_image_grid(images, cols=4, rows=2)
    except KeyboardInterrupt:
        pass
    finally:
        display.close()


if __name__ == "__main__":
    main()

import time

from paths_config import TMP_DISPLAY_DIR
from utilities.log import get_logger, install_print_logger


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
LIVE_RESCAN_INTERVAL_SEC = 2.0
LIVE_BATCH_ROTATION_INTERVAL_SEC = 1.0


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

    try:
        now = time.monotonic()
        cached_images = rotation_state.get("cached_images")
        last_scan_ts = rotation_state.get("last_scan_ts", 0.0)
        last_dir_mtime = rotation_state.get("last_dir_mtime")

        # Rescan directory only periodically (or when directory mtime changes).
        should_rescan = cached_images is None or (now - last_scan_ts) >= LIVE_RESCAN_INTERVAL_SEC
        dir_mtime = None
        try:
            dir_mtime = file_manager.getmtime(local_path)
        except Exception:
            pass

        if not should_rescan and dir_mtime is not None and dir_mtime != last_dir_mtime:
            should_rescan = True

        if should_rescan:
            images = []
            for name in file_manager.listdir(local_path):
                if not name.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                path = file_manager.join(local_path, name)
                if file_manager.is_file(path):
                    images.append(name)

            # Most recent first by filename ordering used in production.
            images.sort(reverse=True)

            if images != cached_images:
                rotation_state["catalog_version"] = rotation_state.get("catalog_version", 0) + 1
                rotation_state["cached_images"] = images

            rotation_state["last_scan_ts"] = now
            rotation_state["last_dir_mtime"] = dir_mtime

        images = rotation_state.get("cached_images") or []
        if not images:
            rotation_state["current_batch"] = []
            return []

        catalog_version = rotation_state.get("catalog_version", 0)
        current_batch_catalog_version = rotation_state.get("current_batch_catalog_version", -1)
        last_rotation_ts = rotation_state.get("last_rotation_ts", 0.0)

        should_rotate = (
            not rotation_state.get("current_batch")
            or (now - last_rotation_ts) >= LIVE_BATCH_ROTATION_INTERVAL_SEC
            or current_batch_catalog_version != catalog_version
        )

        if should_rotate:
            total_batches = (len(images) + max_images - 1) // max_images
            current_offset = rotation_state.get("current_offset", 0)
            if total_batches > 0:
                current_offset %= total_batches
            else:
                current_offset = 0

            start_idx = current_offset * max_images
            end_idx = start_idx + max_images
            selected_images = images[start_idx:end_idx]
            selected_images.sort(key=_display_sort_key)

            rotation_state["current_batch"] = selected_images
            rotation_state["current_offset"] = (
                (current_offset + 1) % total_batches if total_batches > 0 else 0
            )
            rotation_state["last_rotation_ts"] = now
            rotation_state["current_batch_catalog_version"] = catalog_version

        return [
            file_manager.join(local_path, img_name)
            for img_name in (rotation_state.get("current_batch") or [])
        ]
    except Exception as e:
        logger.error(f"[LOCAL] Error loading live images: {e}", allow_repeat=True)
        return []


def main():
    install_print_logger(reset=True)
    logger = get_logger()

    from display_window import DisplayWindow

    temp_dir = str(TMP_DISPLAY_DIR)
    live_rotation_state = {
        "current_offset": 0,
        "cached_images": None,
        "last_scan_ts": 0.0,
        "last_dir_mtime": None,
        "catalog_version": 0,
        "current_batch": [],
        "last_rotation_ts": 0.0,
        "current_batch_catalog_version": -1,
    }
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

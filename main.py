import time

from paths_config import (
    REMOTE_HIST_DISPLAY_DIR,
    REMOTE_TEST_DISPLAY_DIR,
    TMP_DISPLAY_DIR,
)
from settings import get_optional_sftp_settings
from utilities.log import get_logger, install_print_logger


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
LIVE_RESCAN_INTERVAL_SEC = 2.0
LIVE_BATCH_ROTATION_INTERVAL_SEC = 1.0
SFTP_RECONNECT_INTERVAL_SEC = 10.0
CAMERA_IDS = {
    "25430027",
    "25384186",
    "25430026",
    "25384190",
    "25324823",
    "25324824",
    "25371186",
}


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


def _download_live_images_remote(
    app,
    remote_path,
    local_path,
    remote_hist_dir,
    rotation_state,
    logger,
    max_images=7,
):
    if not app or not app.sftp_client:
        return []

    app.file_manager.makedirs(local_path, exist_ok=True)
    downloaded_files = []

    try:
        files = app.list_remote_files(remote_path)
        images = [f for f in files if f.lower().endswith(IMAGE_EXTENSIONS)]

        images.sort(reverse=True)

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

        rotation_state["current_offset"] = (
            (current_offset + 1) % total_batches if total_batches > 0 else 0
        )

        app.ensure_remote_dir(remote_hist_dir)

        for img_name in selected_images:
            local_file = app.file_manager.join(local_path, img_name)
            remote_img_path = app.join_remote_path(remote_path, img_name)
            remote_hist_path = app.join_remote_path(remote_hist_dir, img_name)
            app.download_file(remote_img_path, local_file)
            app.upload_file(local_file, remote_hist_path)
            downloaded_files.append(local_file)

        return downloaded_files

    except Exception as e:
        logger.error(f"[SSH] Error downloading live images: {e}", allow_repeat=True)
        try:
            app.disconnect_sftp()
        except Exception:
            pass
        return []


def _process_remote_event(msg, display, logger):
    if not isinstance(msg, dict):
        return

    msg_type = msg.get("type")
    if msg_type == "stdout":
        line = str(msg.get("line", ""))
        lower_line = line.lower()

        if (not display.trigger_active) and ("waiting for trigger" in lower_line):
            display.trigger_active = True
            logger.info(
                "[REMOTE] Trigger status: ACTIVATED (found 'Waiting for Trigger')",
                allow_repeat=True,
            )

        if "configured successfully" in lower_line:
            for cam_id in CAMERA_IDS:
                if cam_id in line and cam_id not in display.connected_cameras:
                    display.connected_cameras.add(cam_id)
                    logger.info(
                        f"[REMOTE] Camera {cam_id} configured successfully",
                        allow_repeat=True,
                    )
                    break


def main():
    install_print_logger(reset=True)
    logger = get_logger()

    from display_window import DisplayWindow
    from sftp_app import SFTPApp

    temp_dir = str(TMP_DISPLAY_DIR)
    remote_path = REMOTE_TEST_DISPLAY_DIR
    remote_hist_dir = REMOTE_HIST_DISPLAY_DIR
    remote_command = (
        "sh -lc 'echo $$; "
        "cd ~/Vision-Standard 2>/dev/null || cd ~/vision-standard; "
        "stdbuf -oL -eL python3 -u main.py -f art_1861_endform -p omron -d teledyne 2>&1'"
    )
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
    live_rotation_state_remote = {"current_offset": 0}
    sftp_app = None
    sftp_credentials = None
    sftp_connected = False
    next_reconnect_ts = 0.0
    remote_process = None
    remote_pid = None
    stop_event = None
    pid_queue = None
    event_queue = None

    try:
        sftp_credentials = get_optional_sftp_settings()
    except Exception as e:
        logger.error(
            f"[SSH] Invalid SFTP settings, running local-only: {e}",
            allow_repeat=True,
        )
        sftp_credentials = None

    if sftp_credentials is not None:
        sftp_app = SFTPApp(
            sftp_credentials["hostname"],
            sftp_credentials["port"],
            sftp_credentials["username"],
            sftp_credentials["password"],
        )
        sftp_connected = sftp_app.connect_sftp()
        if sftp_connected:
            logger.info(
                "[SSH] Running with SFTP enabled (remote + local fallback)",
                allow_repeat=True,
            )
        else:
            logger.warn(
                "[SSH] Initial SFTP connection failed, running local-only fallback",
                allow_repeat=True,
            )
            next_reconnect_ts = time.monotonic() + SFTP_RECONNECT_INTERVAL_SEC
    else:
        logger.info("[LOCAL] Running in local-only mode (SFTP disabled)", allow_repeat=True)

    try:
        display = DisplayWindow(
            width=1920,
            height=1080,
            window_name="Display Imagenes",
            refresh_interval=0.25,
            sftp_client=sftp_app.sftp_client if sftp_connected and sftp_app else None,
            sftp_credentials=sftp_credentials,
        )
        logger.info("[DB] DisplayWindow initialized successfully", allow_repeat=True)
    except Exception as e:
        logger.error(f"[DB] Failed to initialize DisplayWindow: {e}", allow_repeat=True)
        if sftp_app:
            sftp_app.disconnect_sftp()
        return

    display.start_historic_download_on_startup(temp_dir, check_interval=10)

    def handle_disconnect(reason):
        nonlocal sftp_connected, next_reconnect_ts
        logger.warn(f"[SSH] Disconnected ({reason}), switching to local fallback", allow_repeat=True)
        sftp_connected = False
        if sftp_app:
            try:
                sftp_app.disconnect_sftp()
            except Exception:
                pass
        display.set_sftp_client(None)
        next_reconnect_ts = time.monotonic() + SFTP_RECONNECT_INTERVAL_SEC

    def try_connect(reason):
        nonlocal sftp_connected, next_reconnect_ts
        if not sftp_app:
            return False
        if sftp_connected and sftp_app.sftp_client:
            return True

        logger.info(f"[SSH] Connect attempt ({reason})", allow_repeat=True)
        connected = sftp_app.connect_sftp()
        if connected and sftp_app.sftp_client:
            sftp_connected = True
            display.set_sftp_client(sftp_app.sftp_client)
            logger.info("[SSH] Reconnected successfully", allow_repeat=True)
            return True

        sftp_connected = False
        display.set_sftp_client(None)
        try:
            sftp_app.disconnect_sftp()
        except Exception:
            pass
        next_reconnect_ts = time.monotonic() + SFTP_RECONNECT_INTERVAL_SEC
        logger.warn("[SSH] Reconnect failed, keeping local fallback", allow_repeat=True)
        return False

    def start_remote_process():
        nonlocal remote_process, remote_pid, stop_event, pid_queue, event_queue
        if remote_process and remote_process.is_alive():
            display.remote_requested = True
            return
        if not sftp_app:
            logger.warn("[REMOTE] Start requested but SFTP is disabled", allow_repeat=True)
            display.remote_requested = False
            return
        if not sftp_connected:
            if not try_connect("remote-start"):
                logger.warn("[REMOTE] Cannot start remote process while disconnected", allow_repeat=True)
                display.remote_requested = False
                return

        logger.info("[REMOTE] Start requested", allow_repeat=True)
        from multiprocessing import Event, Queue

        stop_event = Event()
        pid_queue = Queue()
        event_queue = Queue()
        remote_pid = None
        remote_process = sftp_app.start_remote_process_multiprocess(
            remote_command,
            pid_queue=pid_queue,
            stop_event=stop_event,
            status_queue=event_queue,
        )
        display.remote_requested = True
        display.trigger_active = False
        display.connected_cameras = set()
        try:
            remote_pid = pid_queue.get(timeout=5)
            logger.info(f"[REMOTE] PID: {remote_pid}", allow_repeat=True)
        except Exception:
            remote_pid = None

    def stop_remote_process(reason="user"):
        nonlocal remote_process, remote_pid, stop_event, pid_queue, event_queue
        if stop_event is None and remote_process is None and remote_pid is None:
            display.remote_requested = False
            display.trigger_active = False
            display.connected_cameras = set()
            return
        logger.info(f"[REMOTE] Stop requested ({reason})", allow_repeat=True)
        if stop_event is not None:
            stop_event.set()
        if remote_process and remote_process.is_alive():
            remote_process.join(timeout=5)
        if remote_process and remote_process.is_alive():
            remote_process.terminate()
            remote_process.join(timeout=2)
        if remote_pid and sftp_app and sftp_connected and sftp_app.ssh_client:
            try:
                sftp_app.ssh_client.exec_command(f"kill {remote_pid}")
            except Exception:
                pass
        logger.info("[REMOTE] Stop sequence completed", allow_repeat=True)
        remote_process = None
        remote_pid = None
        stop_event = None
        pid_queue = None
        event_queue = None
        display.remote_requested = False
        display.trigger_active = False
        display.connected_cameras = set()

    try:
        while True:
            if display.remote_action_request:
                action = display.remote_action_request
                display.remote_action_request = None
                if action == "start":
                    start_remote_process()
                elif action == "stop":
                    stop_remote_process("button")

            if event_queue is not None:
                try:
                    while True:
                        msg = event_queue.get_nowait()
                        _process_remote_event(msg, display, logger)
                except Exception:
                    pass

            if sftp_app and not sftp_connected and time.monotonic() >= next_reconnect_ts:
                try_connect("periodic-retry")

            if display.exit_requested:
                break

            if display.historic_mode:
                images = display.download_historic_batch(temp_dir, max_images=7)
            else:
                images = []
                if sftp_connected and sftp_app:
                    images = _download_live_images_remote(
                        sftp_app,
                        remote_path,
                        temp_dir,
                        remote_hist_dir,
                        live_rotation_state_remote,
                        logger,
                        max_images=7,
                    )
                    if not sftp_app.sftp_client:
                        stop_remote_process("sftp-disconnect")
                        handle_disconnect("live-download-failure")
                    elif not images:
                        logger.info(
                            "[LOCAL] Falling back to local live images",
                            allow_repeat=True,
                        )

                if not images:
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
        stop_remote_process("exit")
        display.close()
        if sftp_app:
            sftp_app.disconnect_sftp()


if __name__ == "__main__":
    main()

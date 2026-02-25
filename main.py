from utilities.log import get_logger, install_print_logger
from paths_config import TMP_DISPLAY_DIR, REMOTE_HIST_DISPLAY_DIR, REMOTE_TEST_DISPLAY_DIR
from settings import get_sftp_settings


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
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


def _clear_local_display_dir(file_manager, local_path):
    file_manager.makedirs(local_path, exist_ok=True)
    for file_name in file_manager.listdir(local_path):
        file_path = file_manager.join(local_path, file_name)
        try:
            if file_manager.is_file(file_path):
                file_manager.remove(file_path)
        except Exception:
            pass


def _download_live_images(app, remote_path, local_path, remote_hist_dir, rotation_state, logger, max_images=7):
    if not app.sftp_client:
        return []

    downloaded_files = []
    _clear_local_display_dir(app.file_manager, local_path)

    try:
        files = app.list_remote_files(remote_path)
        images = [f for f in files if f.lower().endswith(IMAGE_EXTENSIONS)]

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

    try:
        sftp_settings = get_sftp_settings()
        hostname = sftp_settings["hostname"]
        port = sftp_settings["port"]
        username = sftp_settings["username"]
        password = sftp_settings["password"]
    except Exception as e:
        logger.error(f"[SSH] Failed to load SFTP settings: {e}", allow_repeat=True)
        return

    app = SFTPApp(hostname, port, username, password)
    remote_process = None
    remote_pid = None
    stop_event = None
    pid_queue = None
    event_queue = None
    live_rotation_state = {"current_offset": 0}

    if app.connect_sftp():
        remote_path = REMOTE_TEST_DISPLAY_DIR
        remote_hist_dir = REMOTE_HIST_DISPLAY_DIR
        temp_dir = str(TMP_DISPLAY_DIR)
        remote_command = (
            "sh -lc 'echo $$; "
            "cd ~/Vision-Standard 2>/dev/null || cd ~/vision-standard; "
            "stdbuf -oL -eL python3 -u main.py -f art_1861_endform -p omron -d teledyne 2>&1'"
        )

        sftp_credentials = {
            "hostname": hostname,
            "port": port,
            "username": username,
            "password": password,
        }

        try:
            display = DisplayWindow(
                width=1920,
                height=1080,
                window_name="Display Imagenes",
                refresh_interval=0.25,
                sftp_client=app.sftp_client,
                sftp_credentials=sftp_credentials,
            )
            logger.info("[DB] DisplayWindow initialized successfully", allow_repeat=True)
        except Exception as e:
            logger.error(f"[DB] Failed to initialize DisplayWindow: {e}", allow_repeat=True)
            app.disconnect_sftp()
            return

        display.start_historic_download_on_startup(temp_dir, check_interval=10)

        def start_remote_process():
            nonlocal remote_process, remote_pid, stop_event, pid_queue, event_queue
            if remote_process and remote_process.is_alive():
                display.remote_requested = True
                return
            logger.info("[REMOTE] Start requested", allow_repeat=True)
            from multiprocessing import Event, Queue

            stop_event = Event()
            pid_queue = Queue()
            event_queue = Queue()
            remote_pid = None
            remote_process = app.start_remote_process_multiprocess(
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
            if remote_pid and app.ssh_client:
                try:
                    app.ssh_client.exec_command(f"kill {remote_pid}")
                except Exception:
                    pass
                try:
                    running = app.is_pid_running(remote_pid)
                    if running:
                        logger.info(
                            f"[REMOTE] Process {remote_pid} still running",
                            allow_repeat=True,
                        )
                    else:
                        logger.info(
                            f"[REMOTE] Process {remote_pid} confirmed terminated",
                            allow_repeat=True,
                        )
                except Exception:
                    logger.info(
                        f"[REMOTE] Process {remote_pid} termination check failed",
                        allow_repeat=True,
                    )
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

                if display.exit_requested:
                    break

                if display.historic_mode:
                    images = display.download_historic_batch(temp_dir, max_images=7)
                else:
                    images = _download_live_images(
                        app,
                        remote_path,
                        temp_dir,
                        remote_hist_dir,
                        live_rotation_state,
                        logger,
                        max_images=7,
                    )

                if images:
                    display.image_paths = images
                    display.show_image_grid(images, cols=4, rows=2)
                else:
                    import time

                    time.sleep(2)
        except KeyboardInterrupt:
            pass
        finally:
            stop_remote_process("exit")
            display.close()
            app.disconnect_sftp()
    else:
        logger.error(
            "[SSH] Initial SSH/SFTP connection failed. App startup aborted.",
            allow_repeat=True,
        )


if __name__ == "__main__":
    main()

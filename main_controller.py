import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Event, Process, Queue
from threading import Thread

from file_manager import FileManager
from paths_config import (
    HISTORIC_LOCAL_DIR,
    HISTORIC_SUBDIR_NAME,
    REMOTE_HIST_DISPLAY_DIR,
    REMOTE_TEST_DISPLAY_DIR,
    STATUS_SYNC_DIRS,
    SYNC_IMAGES_BASE_DIR,
    TMP_DISPLAY_DIR,
)
from sftp_app import SFTPApp
from utilities.log import get_logger, install_print_logger


def _display_sort_key(filename):
    lower_name = filename.lower()
    if "side" in lower_name:
        return (0, filename)
    if "front" in lower_name:
        return (1, filename)
    if "diag" in lower_name:
        return (2, filename)
    return (3, filename)


def _sleep_with_stop(stop_event, seconds):
    if seconds <= 0:
        return

    end_time = time.monotonic() + seconds
    while time.monotonic() < end_time:
        if stop_event is not None and stop_event.is_set():
            return
        time.sleep(0.2)


def _cleanup_zero_byte_images(file_manager, target_dir, image_extensions):
    removed_count = 0
    if not file_manager.exists(target_dir):
        return removed_count

    for name in file_manager.listdir(target_dir):
        if not name.lower().endswith(image_extensions):
            continue
        file_path = file_manager.join(target_dir, name)
        if not file_manager.is_file(file_path):
            continue
        try:
            if file_manager.getsize(file_path) > 0:
                continue
        except Exception:
            continue

        try:
            file_manager.remove(file_path)
            removed_count += 1
        except Exception:
            pass

    return removed_count


def _sftp_get_with_cleanup_retry(
    file_manager,
    sftp_client,
    remote_path,
    local_path,
    max_attempts=2,
):
    attempts = max(1, int(max_attempts))
    last_exc = None

    for _ in range(attempts):
        try:
            file_manager.sftp_get(sftp_client, remote_path, local_path)
            if file_manager.getsize(local_path) <= 0:
                raise IOError(f"Downloaded file is empty: {remote_path}")
            return
        except Exception as exc:
            last_exc = exc
            try:
                if file_manager.exists(local_path):
                    file_manager.remove(local_path)
            except Exception:
                pass

    if last_exc is not None:
        raise last_exc


def _download_images_background_worker(
    hostname,
    port,
    username,
    password,
    remote_dir,
    historic_temp_dir,
    check_interval=30,
    reconnect_interval=10,
    stop_event=None,
    verbose=False,
):
    import paramiko

    install_print_logger(reset=False)
    logger = get_logger()
    file_manager = FileManager()

    image_extensions = (".png", ".jpg", ".jpeg", ".bmp")
    ssh_client = None
    sftp_client = None
    sync_error_count = 0
    size_mismatch_count = 0
    total_downloaded_since_start = 0
    counter_interval_sec = 15 * 60
    next_counter_ts = time.monotonic() + counter_interval_sec

    file_manager.makedirs(historic_temp_dir, exist_ok=True)

    def close_connections():
        nonlocal sftp_client, ssh_client
        try:
            if sftp_client:
                sftp_client.close()
        except Exception:
            pass
        try:
            if ssh_client:
                ssh_client.close()
        except Exception:
            pass
        sftp_client = None
        ssh_client = None

    logger.info(
        "[HIST_SYNC_SSH] Historic download started (counter logged every 15 minutes)",
        allow_repeat=True,
    )
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            if sftp_client is None:
                try:
                    logger.info(
                        f"[HIST_SYNC_SSH] Connecting to {hostname}:{port} as {username}",
                        allow_repeat=True,
                    )
                    ssh_client = paramiko.SSHClient()
                    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    ssh_client.connect(
                        hostname=hostname,
                        port=port,
                        username=username,
                        password=password,
                        timeout=10,
                    )
                    sftp_client = ssh_client.open_sftp()
                    logger.info("[HIST_SYNC_SSH] Connection successful", allow_repeat=True)
                except Exception as exc:
                    logger.error(
                        f"[HIST_SYNC_SSH] Connection failed: {exc}",
                        allow_repeat=True,
                    )
                    close_connections()
                    _sleep_with_stop(stop_event, reconnect_interval)
                    continue

            try:
                removed_zero_files = _cleanup_zero_byte_images(
                    file_manager=file_manager,
                    target_dir=historic_temp_dir,
                    image_extensions=image_extensions,
                )
                if removed_zero_files:
                    logger.warn(
                        (
                            "[HIST_SYNC_SSH] Removed "
                            f"{removed_zero_files} zero-byte local historic images"
                        ),
                        allow_repeat=True,
                    )

                existing_local = (
                    set(file_manager.listdir(historic_temp_dir))
                    if file_manager.exists(historic_temp_dir)
                    else set()
                )

                file_manager.sftp_chdir(sftp_client, remote_dir)
                remote_files = file_manager.sftp_listdir(sftp_client)

                all_remote_images = [
                    f for f in remote_files if f.lower().endswith(image_extensions)
                ]
                all_remote_images = [f for f in all_remote_images if f.startswith("11861")]

                jsn_groups = defaultdict(list)
                for img in all_remote_images:
                    jsn = img.split("_")[0] if "_" in img else img
                    jsn_groups[jsn].append(img)

                sorted_jsns = sorted(jsn_groups.keys(), reverse=True)
                excluded_jsns = set(sorted_jsns[:2]) if len(sorted_jsns) >= 2 else set()

                filtered_images = [
                    img
                    for img in all_remote_images
                    if (img.split("_")[0] if "_" in img else img) not in excluded_jsns
                ]

                images_to_download = [
                    img for img in filtered_images if img not in existing_local
                ]

                downloaded_count = 0
                for img in images_to_download:
                    if stop_event is not None and stop_event.is_set():
                        break
                    local_file = file_manager.join(historic_temp_dir, img)
                    _sftp_get_with_cleanup_retry(
                        file_manager=file_manager,
                        sftp_client=sftp_client,
                        remote_path=img,
                        local_path=local_file,
                        max_attempts=2,
                    )
                    downloaded_count += 1

                if downloaded_count:
                    total_downloaded_since_start += downloaded_count

                now = time.monotonic()
                while now >= next_counter_ts:
                    logger.info(
                        (
                            "[HIST_SYNC_SSH] 15-minute counter: "
                            f"total_downloaded_since_start={total_downloaded_since_start}"
                        ),
                        allow_repeat=True,
                    )
                    next_counter_ts += counter_interval_sec

                _sleep_with_stop(stop_event, check_interval)

            except FileNotFoundError:
                logger.warn(
                    f"[HIST_SYNC_SSH] Remote historic folder not found: {remote_dir}",
                    allow_repeat=True,
                )
                _sleep_with_stop(stop_event, check_interval)
            except Exception as exc:
                sync_error_count += 1
                lower_exc = str(exc).lower()
                if "size mismatch in get" in lower_exc:
                    size_mismatch_count += 1
                logger.error(
                    (
                        f"[HIST_SYNC_SSH] Sync error: {exc} | "
                        f"sync_error_count={sync_error_count}"
                        + (
                            f" | size_mismatch_count={size_mismatch_count}"
                            if "size mismatch in get" in lower_exc
                            else ""
                        )
                    ),
                    allow_repeat=True,
                )
                close_connections()
                _sleep_with_stop(stop_event, reconnect_interval)
    finally:
        close_connections()
        logger.info("[HIST_SYNC_SSH] Historic worker stopped", allow_repeat=True)


def _download_live_images_local_impl(
    file_manager,
    local_path,
    rotation_state,
    logger,
    image_extensions,
    live_rescan_interval_sec,
    live_batch_rotation_interval_sec,
    max_images=7,
):
    file_manager.makedirs(local_path, exist_ok=True)

    try:
        now = time.monotonic()
        cached_images = rotation_state.get("cached_images")
        last_scan_ts = rotation_state.get("last_scan_ts", 0.0)
        last_dir_mtime = rotation_state.get("last_dir_mtime")

        should_rescan = cached_images is None or (now - last_scan_ts) >= live_rescan_interval_sec
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
                if not name.lower().endswith(image_extensions):
                    continue
                path = file_manager.join(local_path, name)
                if not file_manager.is_file(path):
                    continue
                try:
                    if file_manager.getsize(path) <= 0:
                        continue
                except Exception:
                    continue
                images.append(name)

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
            or (now - last_rotation_ts) >= live_batch_rotation_interval_sec
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
    except Exception as exc:
        logger.error(f"[LOCAL] Error loading live images: {exc}", allow_repeat=True)
        return []


def _download_live_images_remote_impl(
    app,
    remote_path,
    local_path,
    remote_hist_dir,
    rotation_state,
    logger,
    image_extensions,
    max_images=7,
):
    if not app or not app.sftp_client:
        return []

    app.file_manager.makedirs(local_path, exist_ok=True)
    downloaded_files = []
    transfer_error_count = int(rotation_state.get("transfer_error_count", 0) or 0)

    try:
        files = app.list_remote_files(remote_path)
        images = [f for f in files if f.lower().endswith(image_extensions)]
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
            try:
                _sftp_get_with_cleanup_retry(
                    file_manager=app.file_manager,
                    sftp_client=app.sftp_client,
                    remote_path=remote_img_path,
                    local_path=local_file,
                    max_attempts=2,
                )
                app.upload_file(local_file, remote_hist_path)
                downloaded_files.append(local_file)
            except Exception as exc:
                transfer_error_count += 1
                rotation_state["transfer_error_count"] = transfer_error_count
                logger.warn(
                    (
                        f"[SSH] Skipping live image {img_name} after transfer error: {exc} | "
                        f"transfer_error_count={transfer_error_count}"
                    ),
                    allow_repeat=True,
                )

        return downloaded_files

    except Exception as exc:
        logger.error(f"[SSH] Error downloading live images: {exc}", allow_repeat=True)
        try:
            app.disconnect_sftp()
        except Exception:
            pass
        return []


def _process_remote_event_impl(msg, display, logger, camera_ids):
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
            for cam_id in camera_ids:
                if cam_id in line and cam_id not in display.connected_cameras:
                    display.connected_cameras.add(cam_id)
                    logger.info(
                        f"[REMOTE] Camera {cam_id} configured successfully",
                        allow_repeat=True,
                    )
                    break


def download_live_images_local(file_manager, local_path, rotation_state, logger, max_images=7):
    return _download_live_images_local_impl(
        file_manager=file_manager,
        local_path=local_path,
        rotation_state=rotation_state,
        logger=logger,
        image_extensions=(".png", ".jpg", ".jpeg", ".bmp"),
        live_rescan_interval_sec=2.0,
        live_batch_rotation_interval_sec=1.0,
        max_images=max_images,
    )


def download_live_images_remote(
    app,
    remote_path,
    local_path,
    remote_hist_dir,
    rotation_state,
    logger,
    max_images=7,
):
    return _download_live_images_remote_impl(
        app=app,
        remote_path=remote_path,
        local_path=local_path,
        remote_hist_dir=remote_hist_dir,
        rotation_state=rotation_state,
        logger=logger,
        image_extensions=(".png", ".jpg", ".jpeg", ".bmp"),
        max_images=max_images,
    )


def process_remote_event(msg, display, logger, camera_ids=None):
    _process_remote_event_impl(
        msg=msg,
        display=display,
        logger=logger,
        camera_ids=camera_ids
        or {
            "25430027",
            "25384186",
            "25430026",
            "25384190",
            "25324823",
            "25324824",
            "25371186",
        },
    )


@dataclass
class ControllerConfig:
    image_extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp")
    live_rescan_interval_sec: float = 2.0
    live_batch_rotation_interval_sec: float = 1.0
    sftp_reconnect_interval_sec: float = 10.0
    db_reconnect_interval_sec: float = 3.0
    max_images: int = 7
    remote_command: str = (
        "sh -lc 'echo $$; "
        "cd ~/Vision-Standard 2>/dev/null || cd ~/vision-standard; "
        "stdbuf -oL -eL python3 -u main.py -f art_1861_endform -p omron -d teledyne 2>&1'"
    )
    camera_ids: set = field(
        default_factory=lambda: {
            "25430027",
            "25384186",
            "25430026",
            "25384190",
            "25324823",
            "25324824",
            "25371186",
        }
    )
    temp_dir: str = field(default_factory=lambda: str(TMP_DISPLAY_DIR))
    remote_live_dir: str = REMOTE_TEST_DISPLAY_DIR
    remote_hist_dir: str = REMOTE_HIST_DISPLAY_DIR
    display_cols: int = 4
    display_rows: int = 2
    historic_download_check_interval: int = 10

class MainController:
    def __init__(
        self,
        display,
        logger=None,
        sftp_credentials=None,
        sftp_app=None,
        config=None,
        file_manager=None,
    ):
        self.display = display
        self.logger = logger or get_logger()
        self.config = config or ControllerConfig()
        self.file_manager = file_manager or getattr(display, "file_manager", None) or FileManager()
        self.sftp_credentials = sftp_credentials or getattr(display, "sftp_credentials", None)
        self.sftp_app = sftp_app

        self.sftp_connected = False
        self.next_reconnect_ts = 0.0
        self.db_connected = False
        self.next_db_reconnect_ts = 0.0
        self.remote_process = None
        self.remote_pid = None
        self.stop_event = None
        self.pid_queue = None
        self.event_queue = None

        self.live_rotation_state = {
            "current_offset": 0,
            "cached_images": None,
            "last_scan_ts": 0.0,
            "last_dir_mtime": None,
            "catalog_version": 0,
            "current_batch": [],
            "last_rotation_ts": 0.0,
            "current_batch_catalog_version": -1,
        }
        self.live_rotation_state_remote = {"current_offset": 0}
        self.historic_bootstrap_loading = False
        self.historic_bootstrap_complete = False
        self.historic_bootstrap_thread = None
        self.sync_worker_thread = None
        self.reset_worker_thread = None

        if hasattr(self.display, "set_controller"):
            self.display.set_controller(self)
        else:
            self.display.controller = self

        if getattr(self.display, "db", None) is None:
            self._mark_db_unavailable("startup-no-connection")
        else:
            self.db_connected = True
            if hasattr(self.display, "set_db_connection"):
                self.display.set_db_connection(self.display.db)

    def initialize(self):
        if not self.db_connected:
            self.try_connect_db("startup")
        if self.db_connected:
            self._register_historic_local_dir_on_startup()

        if self.sftp_credentials is not None and self.sftp_app is None:
            self.sftp_app = SFTPApp(
                self.sftp_credentials["hostname"],
                self.sftp_credentials["port"],
                self.sftp_credentials["username"],
                self.sftp_credentials["password"],
            )

        if self.sftp_app:
            self.sftp_connected = self.sftp_app.connect_sftp()
            if self.sftp_connected:
                self.logger.info(
                    "[SSH] Running with SFTP enabled (remote + local fallback)",
                    allow_repeat=True,
                )
                self.display.set_sftp_client(self.sftp_app.sftp_client)
            else:
                self.logger.warn(
                    "[SSH] Initial SFTP connection failed, running local-only fallback",
                    allow_repeat=True,
                )
                self.display.set_sftp_client(None)
                self.next_reconnect_ts = time.monotonic() + self.config.sftp_reconnect_interval_sec
        else:
            self.logger.info("[LOCAL] Running in local-only mode (SFTP disabled)", allow_repeat=True)
            self.display.set_sftp_client(None)

    def _db_block_message(self):
        return "PostgreSQL is disconnected. Start postgres and wait for automatic reconnect."

    def _mark_db_unavailable(self, reason, exc=None):
        self.db_connected = False
        if getattr(self.display, "db", None) is not None:
            try:
                self.display.db.close()
            except Exception:
                pass
        if hasattr(self.display, "set_db_blocked"):
            self.display.set_db_blocked(self._db_block_message())
        else:
            self.display.db = None

        self.next_db_reconnect_ts = time.monotonic() + self.config.db_reconnect_interval_sec
        if exc is not None:
            self.logger.error(f"[DB] Connection unavailable ({reason}): {exc}")
        else:
            self.logger.warn(f"[DB] Connection unavailable ({reason})")

    def try_connect_db(self, reason):
        if self.db_connected and getattr(self.display, "db", None) is not None:
            return True
        if time.monotonic() < self.next_db_reconnect_ts:
            return False

        try:
            from db import get_db_connection

            db_client = get_db_connection()
            if hasattr(self.display, "set_db_connection"):
                self.display.set_db_connection(db_client)
            else:
                self.display.db = db_client
            self.db_connected = True
            self.next_db_reconnect_ts = 0.0
            self.logger.info("[DB] Reconnected successfully", allow_repeat=True)
            return True
        except Exception as exc:
            self._mark_db_unavailable(reason, exc=exc)
            return False

    def _register_historic_local_dir_on_startup(self):
        if self.historic_bootstrap_loading or self.historic_bootstrap_complete:
            return
        if not self.db_connected:
            return

        historic_dir = str(HISTORIC_LOCAL_DIR)
        self.historic_bootstrap_loading = True
        self.logger.info("[DB] Historic startup bootstrap started", allow_repeat=True)

        def _bootstrap_worker():
            worker_db = None
            completed = False
            try:
                from db import get_db_connection

                worker_db = get_db_connection()
                self._register_local_images_in_db(
                    historic_dir,
                    db_client=worker_db,
                    track_registered=False,
                )
                self.logger.info("[DB] Historic startup bootstrap completed", allow_repeat=True)
                completed = True
            except Exception as exc:
                self.logger.error(
                    f"[DB] Historic startup bootstrap failed: {exc}",
                    allow_repeat=True,
                )
            finally:
                if worker_db is not None:
                    try:
                        worker_db.close()
                    except Exception:
                        pass
                self.historic_bootstrap_loading = False
                if completed:
                    self.historic_bootstrap_complete = True

        self.historic_bootstrap_thread = Thread(
            target=_bootstrap_worker,
            name="historic-db-bootstrap",
            daemon=True,
        )
        self.historic_bootstrap_thread.start()

    def _show_no_images_dialog(self, message):
        d = self.display
        d.no_images_dialog_message = message
        d.show_no_images_dialog = True

    def _set_sync_progress(self, stage, percent):
        d = self.display
        d.sync_stage = str(stage)
        d.sync_progress = max(0, min(100, int(percent)))

    def _set_reset_progress(self, stage, percent):
        d = self.display
        d.reset_stage = str(stage)
        d.reset_progress = max(0, min(100, int(percent)))

    def start_sync_images_by_status_async(self, historic_dir=None, base_dir=None):
        d = self.display
        if getattr(d, "sync_in_progress", False) or getattr(d, "reset_in_progress", False):
            return

        d.sync_in_progress = True
        d.sync_progress = 0
        d.sync_stage = "Preparing dataset sync..."
        d.sync_message = ""
        d.sync_message_is_error = False
        d.sync_message_time = 0

        def _sync_worker():
            worker_db = None
            try:
                from db import get_db_connection

                worker_db = get_db_connection()

                def _sync_progress_cb(done, total, stage):
                    if total <= 0:
                        phase_percent = 0
                        stage_text = stage
                    else:
                        phase_percent = int((done / total) * 85)
                        stage_text = f"{stage} ({done}/{total})"
                    self._set_sync_progress(stage_text, phase_percent)

                sync_result = self.sync_images_by_status(
                    historic_dir=historic_dir,
                    base_dir=base_dir,
                    db_client=worker_db,
                    progress_callback=_sync_progress_cb,
                )

                if not sync_result.get("ok", False):
                    raise RuntimeError(sync_result.get("error", "Dataset sync failed"))

                def _verify_progress_cb(done, total, stage):
                    if total <= 0:
                        phase_percent = 85
                        stage_text = stage
                    else:
                        phase_percent = 85 + int((done / total) * 15)
                        stage_text = f"{stage} ({done}/{total})"
                    self._set_sync_progress(stage_text, phase_percent)

                verify_result = self.verify_sync_images_by_status(
                    historic_dir=historic_dir,
                    base_dir=base_dir,
                    db_client=worker_db,
                    progress_callback=_verify_progress_cb,
                )

                self._set_sync_progress("Completed", 100)
                if verify_result.get("verified"):
                    d.sync_message = "Dataset completed and verified"
                    d.sync_message_is_error = False
                else:
                    issue_count = verify_result.get("issue_count", 0)
                    d.sync_message = (
                        f"Dataset completed but verification failed ({issue_count} issues)"
                    )
                    d.sync_message_is_error = True
                    self.logger.warn(
                        f"[SYNC] Verification failed with {issue_count} issues",
                        allow_repeat=True,
                    )
            except Exception as exc:
                d.sync_message = f"Dataset sync failed: {exc}"
                d.sync_message_is_error = True
                self.logger.error(f"[SYNC] Dataset sync failed: {exc}", allow_repeat=True)
            finally:
                d.sync_in_progress = False
                d.sync_message_time = time.time()
                if worker_db is not None:
                    try:
                        worker_db.close()
                    except Exception:
                        pass

        self.sync_worker_thread = Thread(
            target=_sync_worker,
            name="dataset-sync-worker",
            daemon=True,
        )
        self.sync_worker_thread.start()

    def start_reset_async(self):
        d = self.display
        if getattr(d, "reset_in_progress", False) or getattr(d, "sync_in_progress", False):
            return

        d.reset_in_progress = True
        d.reset_progress = 0
        d.reset_stage = "Preparing reset..."
        d.sync_message = ""
        d.sync_message_is_error = False
        d.sync_message_time = 0

        def _reset_worker():
            worker_db = None
            try:
                from db import get_db_connection

                worker_db = get_db_connection()

                def _reset_progress_cb(done, total, stage):
                    if total <= 0:
                        phase_percent = 0
                        stage_text = stage
                    else:
                        phase_percent = int((done / total) * 100)
                        stage_text = f"{stage} ({done}/{total})"
                    self._set_reset_progress(stage_text, phase_percent)

                result = self.perform_reset(
                    db_client=worker_db,
                    progress_callback=_reset_progress_cb,
                )

                if result.get("ok", False):
                    d.sync_message = "Reset completed successfully"
                    d.sync_message_is_error = False
                else:
                    error_text = result.get("error", "Reset failed")
                    d.sync_message = f"Reset completed with issues: {error_text}"
                    d.sync_message_is_error = True
            except Exception as exc:
                d.sync_message = f"Reset failed: {exc}"
                d.sync_message_is_error = True
                self.logger.error(f"[RESET] Reset failed: {exc}", allow_repeat=True)
            finally:
                d.reset_in_progress = False
                d.sync_message_time = time.time()
                if worker_db is not None:
                    try:
                        worker_db.close()
                    except Exception:
                        pass

        self.reset_worker_thread = Thread(
            target=_reset_worker,
            name="dataset-reset-worker",
            daemon=True,
        )
        self.reset_worker_thread.start()

    def handle_disconnect(self, reason):
        self.logger.warn(f"[SSH] Disconnected ({reason}), switching to local fallback", allow_repeat=True)
        self.sftp_connected = False
        if self.sftp_app:
            try:
                self.sftp_app.disconnect_sftp()
            except Exception:
                pass
        self.display.set_sftp_client(None)
        self.next_reconnect_ts = time.monotonic() + self.config.sftp_reconnect_interval_sec

    def try_connect(self, reason):
        if not self.sftp_app:
            return False
        if self.sftp_connected and self.sftp_app.sftp_client:
            return True

        self.logger.info(f"[SSH] Connect attempt ({reason})", allow_repeat=True)
        connected = self.sftp_app.connect_sftp()
        if connected and self.sftp_app.sftp_client:
            self.sftp_connected = True
            self.display.set_sftp_client(self.sftp_app.sftp_client)
            self.logger.info("[SSH] Reconnected successfully", allow_repeat=True)
            return True

        self.sftp_connected = False
        self.display.set_sftp_client(None)
        try:
            self.sftp_app.disconnect_sftp()
        except Exception:
            pass
        self.next_reconnect_ts = time.monotonic() + self.config.sftp_reconnect_interval_sec
        self.logger.warn("[SSH] Reconnect failed, keeping local fallback", allow_repeat=True)
        return False

    def start_remote_process(self):
        if self.remote_process and self.remote_process.is_alive():
            self.display.remote_requested = True
            return
        if not self.sftp_app:
            self.logger.warn("[REMOTE] Start requested but SFTP is disabled", allow_repeat=True)
            self.display.remote_requested = False
            return
        if not self.sftp_connected and not self.try_connect("remote-start"):
            self.logger.warn("[REMOTE] Cannot start remote process while disconnected", allow_repeat=True)
            self.display.remote_requested = False
            return

        self.logger.info("[REMOTE] Start requested", allow_repeat=True)
        self.stop_event = Event()
        self.pid_queue = Queue()
        self.event_queue = Queue()
        self.remote_pid = None
        self.remote_process = self.sftp_app.start_remote_process_multiprocess(
            self.config.remote_command,
            pid_queue=self.pid_queue,
            stop_event=self.stop_event,
            status_queue=self.event_queue,
        )
        self.display.remote_requested = True
        self.display.trigger_active = False
        self.display.connected_cameras = set()
        try:
            self.remote_pid = self.pid_queue.get(timeout=5)
            self.logger.info(f"[REMOTE] PID: {self.remote_pid}", allow_repeat=True)
        except Exception:
            self.remote_pid = None

    def stop_remote_process(self, reason="user"):
        if self.stop_event is None and self.remote_process is None and self.remote_pid is None:
            self.display.remote_requested = False
            self.display.trigger_active = False
            self.display.connected_cameras = set()
            return
        self.logger.info(f"[REMOTE] Stop requested ({reason})", allow_repeat=True)
        if self.stop_event is not None:
            self.stop_event.set()
        if self.remote_process and self.remote_process.is_alive():
            self.remote_process.join(timeout=5)
        if self.remote_process and self.remote_process.is_alive():
            self.remote_process.terminate()
            self.remote_process.join(timeout=2)
        if self.remote_pid and self.sftp_app and self.sftp_connected and self.sftp_app.ssh_client:
            try:
                self.sftp_app.ssh_client.exec_command(f"kill {self.remote_pid}")
            except Exception:
                pass
        self.logger.info("[REMOTE] Stop sequence completed", allow_repeat=True)
        self.remote_process = None
        self.remote_pid = None
        self.stop_event = None
        self.pid_queue = None
        self.event_queue = None
        self.display.remote_requested = False
        self.display.trigger_active = False
        self.display.connected_cameras = set()

    def _process_remote_events(self):
        if self.event_queue is not None:
            try:
                while True:
                    msg = self.event_queue.get_nowait()
                    _process_remote_event_impl(
                        msg=msg,
                        display=self.display,
                        logger=self.logger,
                        camera_ids=self.config.camera_ids,
                    )
            except Exception:
                pass

    def _download_live_images_local(self):
        return _download_live_images_local_impl(
            file_manager=self.file_manager,
            local_path=self.config.temp_dir,
            rotation_state=self.live_rotation_state,
            logger=self.logger,
            image_extensions=self.config.image_extensions,
            live_rescan_interval_sec=self.config.live_rescan_interval_sec,
            live_batch_rotation_interval_sec=self.config.live_batch_rotation_interval_sec,
            max_images=self.config.max_images,
        )

    def _download_live_images_remote(self):
        return _download_live_images_remote_impl(
            app=self.sftp_app,
            remote_path=self.config.remote_live_dir,
            local_path=self.config.temp_dir,
            remote_hist_dir=self.config.remote_hist_dir,
            rotation_state=self.live_rotation_state_remote,
            logger=self.logger,
            image_extensions=self.config.image_extensions,
            max_images=self.config.max_images,
        )

    def _load_historic_index(self, force_rescan=False):
        d = self.display

        local_historic_dir = self.file_manager.join(self.config.temp_dir, HISTORIC_SUBDIR_NAME)
        if not self.file_manager.exists(local_historic_dir):
            d._historic_index_cache = []
            d._historic_jsn_cache = []
            d._historic_index_mtime = None
            d._historic_index_last_scan = time.monotonic()
            return []

        current_mtime = None
        try:
            current_mtime = self.file_manager.getmtime(local_historic_dir)
        except Exception:
            pass

        use_cache = False
        if not force_rescan and d._historic_index_cache is not None:
            if current_mtime is not None and current_mtime == d._historic_index_mtime:
                use_cache = True
            elif (
                current_mtime is None
                and (time.monotonic() - d._historic_index_last_scan) < d.historic_index_rescan_interval
            ):
                use_cache = True

        if use_cache:
            return d._historic_index_cache

        files = self.file_manager.listdir(local_historic_dir)
        images_with_jsn = [
            name
            for name in files
            if name.lower().endswith(self.config.image_extensions) and name.startswith("11861")
        ]
        valid_images = []
        for name in images_with_jsn:
            image_path = self.file_manager.join(local_historic_dir, name)
            if not self.file_manager.is_file(image_path):
                continue
            try:
                if self.file_manager.getsize(image_path) <= 0:
                    continue
            except Exception:
                continue
            valid_images.append(name)
        images_with_jsn = valid_images

        jsn_groups = defaultdict(list)
        for img in images_with_jsn:
            jsn = img.split("_")[0]
            jsn_groups[jsn].append(img)

        sorted_jsns = sorted(jsn_groups.keys(), reverse=True)
        historic_images = []
        for jsn in sorted_jsns:
            group_images = jsn_groups[jsn]
            group_images.sort(key=_display_sort_key)
            historic_images.append(group_images)

        d._historic_index_cache = historic_images
        d._historic_jsn_cache = sorted_jsns
        d._historic_index_mtime = current_mtime
        d._historic_index_last_scan = time.monotonic()
        d.historic_db_registered = False
        return historic_images

    def enter_historic_mode(self):
        d = self.display
        if self.historic_bootstrap_loading:
            self._show_no_images_dialog("Historic loading in progress")
            return

        current_jsn = None
        fallback_offset = d.historic_offset
        if d.historic_mode and d.historic_images:
            try:
                current_batch = d.historic_images[d.historic_offset]
                if current_batch:
                    current_jsn = (
                        current_batch[0].split("_")[0]
                        if "_" in current_batch[0]
                        else current_batch[0]
                    )
            except Exception:
                current_jsn = None

        try:
            d.historic_images = self._load_historic_index(force_rescan=False)

            if not d.historic_images:
                if not d.historic_mode:
                    self._show_no_images_dialog("No images available")
                return

            if not d.historic_mode:
                d.historic_mode = True
                d.historic_offset = 0
            else:
                if current_jsn:
                    found_idx = None
                    for idx, batch in enumerate(d.historic_images):
                        if not batch:
                            continue
                        batch_jsn = batch[0].split("_")[0] if "_" in batch[0] else batch[0]
                        if batch_jsn == current_jsn:
                            found_idx = idx
                            break
                    if found_idx is not None:
                        d.historic_offset = found_idx
                    else:
                        d.historic_offset = min(fallback_offset, len(d.historic_images) - 1)
                else:
                    d.historic_offset = min(fallback_offset, len(d.historic_images) - 1)
        except Exception as exc:
            print(f"Error entering historic: {exc}")

    def exit_historic_mode(self):
        d = self.display
        d.historic_mode = False
        d.historic_offset = 0
        d.historic_images = []
        d.search_jsn = ""
        d.search_active = False
        d.filtered_suggestions = []
        d.selected_suggestion_idx = -1
        d.show_reset_confirm = False
        d.show_delete_confirm = False
        d.show_piece_date_dialog = False

    def next_historic_batch(self):
        d = self.display
        if not d.historic_images:
            return
        total_batches = len(d.historic_images)
        d.historic_offset = (d.historic_offset + 1) % total_batches

    def prev_historic_batch(self):
        d = self.display
        if not d.historic_images:
            return
        if d.historic_offset == 0:
            return
        d.historic_offset = d.historic_offset - 1

    def collect_available_jsns(self):
        d = self.display
        if not d.historic_images:
            d.available_jsns = []
            return

        if d._historic_jsn_cache:
            d.available_jsns = list(d._historic_jsn_cache)
            return

        jsn_set = set()
        for batch in d.historic_images:
            if batch and len(batch) > 0:
                jsn = batch[0].split("_")[0] if "_" in batch[0] else ""
                if jsn:
                    jsn_set.add(jsn)
        d.available_jsns = sorted(list(jsn_set), reverse=True)

    def update_suggestions(self):
        d = self.display
        if not d.search_jsn:
            d.filtered_suggestions = d.available_jsns[:10]
        else:
            d.filtered_suggestions = [jsn for jsn in d.available_jsns if d.search_jsn in jsn][:10]
        d.selected_suggestion_idx = -1

    def perform_jsn_search(self):
        d = self.display
        if not d.search_jsn.strip():
            print("No JSN entered for search")
            return

        search_term = d.search_jsn.strip()
        for idx, batch in enumerate(d.historic_images):
            jsn = batch[0].split("_")[0] if "_" in batch[0] else ""
            if jsn == search_term:
                d.historic_offset = idx
                print(f"JSN {search_term} found at position {idx}")
                d.search_active = False
                d.filtered_suggestions = []
                d.search_jsn = ""
                return

        print(f"JSN {search_term} not found in historic images")
        d.search_active = False
        d.filtered_suggestions = []
        d.search_jsn = ""

    def _get_current_historic_jsn(self):
        d = self.display
        if not d.historic_images:
            return None
        if d.historic_offset < 0 or d.historic_offset >= len(d.historic_images):
            return None
        batch = d.historic_images[d.historic_offset]
        if not batch:
            return None
        first = batch[0]
        return first.split("_")[0] if "_" in first else first

    def perform_delete_current_piece(self):
        d = self.display
        jsn = self._get_current_historic_jsn()
        if not jsn:
            print("No historic piece selected for deletion")
            return

        print("\n" + "=" * 70)
        print(f"STARTING PIECE DELETE (JSN {jsn})")
        print("=" * 70)

        local_historic_dir = self.file_manager.join(self.config.temp_dir, HISTORIC_SUBDIR_NAME)

        local_deleted = 0
        local_candidates = []
        if self.file_manager.exists(local_historic_dir):
            try:
                for name in self.file_manager.listdir(local_historic_dir):
                    if name.startswith(jsn) and name.lower().endswith(self.config.image_extensions):
                        local_candidates.append(self.file_manager.join(local_historic_dir, name))
                for path in local_candidates:
                    try:
                        self.file_manager.remove(path)
                        local_deleted += 1
                    except Exception as exc:
                        print(f"Error deleting local file {path}: {exc}")
                print(f"Local delete: {local_deleted}/{len(local_candidates)}")
            except Exception as exc:
                print(f"Error reading local historic folder: {exc}")
        else:
            print("Local historic folder does not exist")

        remote_deleted = 0
        if d.sftp_client:
            try:
                self.file_manager.sftp_chdir(d.sftp_client, self.config.remote_hist_dir)
                remote_files = self.file_manager.sftp_listdir(d.sftp_client)
                remote_candidates = [
                    f for f in remote_files if f.startswith(jsn) and f.lower().endswith(self.config.image_extensions)
                ]
                for remote_file in remote_candidates:
                    try:
                        file_path = f"{self.config.remote_hist_dir}/{remote_file}"
                        self.file_manager.sftp_remove(d.sftp_client, file_path)
                        remote_deleted += 1
                    except Exception as exc:
                        print(f"Error deleting remote file {remote_file}: {exc}")
                print(f"Remote delete: {remote_deleted}/{len(remote_candidates)}")
            except Exception as exc:
                print(f"Error accessing remote historic folder: {exc}")
        else:
            print("No SFTP connection available")

        if d.db:
            try:
                query_delete = "DELETE FROM img_results WHERE img_name LIKE %s"
                affected_rows = d.db.execute(query_delete, (f"{jsn}%",))
                print(f"Deleted {affected_rows} database records")
            except Exception as exc:
                print(f"Error clearing database records: {exc}")
        else:
            print("No database connection available")

        if d.temp_results:
            d.temp_results = {k: v for k, v in d.temp_results.items() if not k.startswith(jsn)}
        d._db_registered_images = {name for name in d._db_registered_images if not name.startswith(jsn)}
        if d._db_result_cache:
            d._db_result_cache = {
                k: v for k, v in d._db_result_cache.items() if not k.startswith(jsn)
            }
        for path in local_candidates:
            d._image_cache.pop(path, None)

        d.historic_db_registered = False
        d._historic_index_cache = None
        d._historic_index_mtime = None
        d._historic_jsn_cache = []

        remaining_images = []
        if self.file_manager.exists(local_historic_dir):
            remaining_images = [
                f
                for f in self.file_manager.listdir(local_historic_dir)
                if f.lower().endswith(self.config.image_extensions) and f.startswith("11861")
            ]

        if not remaining_images:
            d.historic_images = []
            d.historic_offset = 0
            d.available_jsns = []
            d.filtered_suggestions = []
            self.exit_historic_mode()
            self._show_no_images_dialog("No images available")
        else:
            self.enter_historic_mode()

        print("=" * 70)
        print("PIECE DELETE COMPLETED")
        print("=" * 70 + "\n")

    def perform_reset(self, db_client=None, progress_callback=None):
        d = self.display
        db = db_client or d.db
        print("\n" + "=" * 70)
        print("STARTING COMPLETE RESET")
        print("=" * 70)

        local_historic_dir = self.file_manager.join(self.config.temp_dir, HISTORIC_SUBDIR_NAME)
        errors = []

        local_entries = []
        if self.file_manager.exists(local_historic_dir):
            try:
                local_entries = list(self.file_manager.listdir(local_historic_dir))
            except Exception as exc:
                errors.append(f"Unable to scan local historic folder: {exc}")
                print(f"Error scanning local historic folder: {exc}")
                local_entries = []

        remote_files = []
        if d.sftp_client:
            try:
                self.file_manager.sftp_chdir(d.sftp_client, self.config.remote_hist_dir)
                remote_files = list(self.file_manager.sftp_listdir(d.sftp_client))
            except Exception as exc:
                errors.append(f"Unable to access remote folder: {exc}")
                print(f"Error accessing remote folder: {exc}")
                remote_files = []

        local_steps = max(1, len(local_entries))
        remote_steps = max(1, len(remote_files))
        db_steps = 1
        final_steps = 1
        total_steps = local_steps + remote_steps + db_steps + final_steps
        completed_steps = 0

        def _advance(stage):
            nonlocal completed_steps
            completed_steps += 1
            if callable(progress_callback):
                progress_callback(completed_steps, total_steps, stage)

        if callable(progress_callback):
            progress_callback(0, total_steps, "Preparing reset")

        if self.file_manager.exists(local_historic_dir):
            if local_entries:
                for idx, entry_name in enumerate(local_entries, start=1):
                    entry_path = self.file_manager.join(local_historic_dir, entry_name)
                    try:
                        if self.file_manager.is_dir(entry_path):
                            self.file_manager.rmtree(entry_path)
                        else:
                            self.file_manager.remove(entry_path)
                    except Exception as exc:
                        errors.append(f"Error removing local file '{entry_name}': {exc}")
                        print(f"Error removing local entry {entry_name}: {exc}")
                    _advance(f"Clearing local historic folder ({idx}/{len(local_entries)})")
                print("Local historic folder cleared")
            else:
                print("Local historic folder is already empty")
                _advance("Local historic folder is already empty")
            try:
                self.file_manager.makedirs(local_historic_dir, exist_ok=True)
            except Exception as exc:
                errors.append(f"Error recreating local historic folder: {exc}")
                print(f"Error recreating local historic folder: {exc}")
        else:
            try:
                self.file_manager.makedirs(local_historic_dir, exist_ok=True)
                print("Local historic folder did not exist and was created")
            except Exception as exc:
                errors.append(f"Error creating local historic folder: {exc}")
                print(f"Error creating local historic folder: {exc}")
            _advance("Preparing local historic folder")

        if d.sftp_client:
            if remote_files:
                print(f"Deleting {len(remote_files)} files from remote server...")
                deleted_count = 0
                for idx, remote_file in enumerate(remote_files, start=1):
                    try:
                        file_path = f"{self.config.remote_hist_dir}/{remote_file}"
                        self.file_manager.sftp_remove(d.sftp_client, file_path)
                        deleted_count += 1
                    except Exception as exc:
                        errors.append(f"Error deleting remote file '{remote_file}': {exc}")
                        print(f"Error deleting {remote_file}: {exc}")
                    _advance(f"Clearing remote historic folder ({idx}/{len(remote_files)})")
                print(f"Deleted {deleted_count}/{len(remote_files)} remote files")
            else:
                print("Remote folder is already empty")
                _advance("Remote historic folder is already empty")
        else:
            print("No SFTP connection available")
            _advance("Remote reset skipped (no SFTP connection)")

        if db:
            try:
                query_delete = "DELETE FROM img_results"
                affected_rows = db.execute(query_delete)
                print(f"Deleted {affected_rows} records from database")
            except Exception as exc:
                errors.append(f"Error clearing database: {exc}")
                print(f"Error clearing database: {exc}")
        else:
            message = "No database connection available"
            errors.append(message)
            print(message)
        _advance("Resetting database")

        d.historic_images = []
        d.historic_offset = 0
        d.temp_results = {}
        d.available_jsns = []
        d.filtered_suggestions = []
        d.historic_db_registered = False
        d._db_registered_images.clear()
        d._historic_index_cache = None
        d._historic_index_mtime = None
        d._historic_jsn_cache = []
        d._db_result_cache.clear()
        d._image_cache.clear()
        _advance("Finalizing reset")

        print("=" * 70)
        if errors:
            print("RESET COMPLETED WITH ISSUES")
        else:
            print("RESET COMPLETED SUCCESSFULLY")
        print("=" * 70 + "\n")

        self.exit_historic_mode()
        if callable(progress_callback):
            progress_callback(total_steps, total_steps, "Completed")
        if errors:
            return {"ok": False, "error": errors[0], "errors": errors}
        return {"ok": True}

    def start_historic_download_on_startup(self, local_path, check_interval=30):
        d = self.display
        historic_temp_dir = self.file_manager.join(local_path, HISTORIC_SUBDIR_NAME)
        self.file_manager.makedirs(historic_temp_dir, exist_ok=True)

        creds = self.sftp_credentials or d.sftp_credentials
        if not creds:
            print("SFTP historic downloader disabled: missing credentials")
            return

        hostname = creds.get("hostname")
        port = creds.get("port")
        username = creds.get("username")
        password = creds.get("password")
        if not all([hostname, port, username, password]):
            print("SFTP historic downloader disabled: incomplete credentials")
            return

        if d.download_process and d.download_process.is_alive():
            print("Background download process is already running")
            return

        try:
            d.download_stop_event = Event()
            d.download_process = Process(
                target=_download_images_background_worker,
                args=(
                    hostname,
                    port,
                    username,
                    password,
                    self.config.remote_hist_dir,
                    historic_temp_dir,
                    check_interval,
                    10,
                    d.download_stop_event,
                ),
            )
            d.download_process.daemon = True
            d.download_process.start()
        except Exception as exc:
            print(f"Error starting background download: {exc}")
            d.download_process = None
            d.download_stop_event = None

    def stop_historic_download_worker(self):
        d = self.display
        if d.download_stop_event is not None:
            try:
                d.download_stop_event.set()
            except Exception:
                pass

        if d.download_process is not None:
            try:
                d.download_process.join(timeout=2)
            except Exception:
                pass

            try:
                if d.download_process.is_alive():
                    d.download_process.terminate()
                    d.download_process.join(timeout=1)
            except Exception:
                pass

        d.download_process = None
        d.download_stop_event = None

    def download_historic_batch(self, local_path, max_images=7):
        d = self.display
        if not d.historic_images:
            return []

        try:
            historic_temp_dir = self.file_manager.join(local_path, HISTORIC_SUBDIR_NAME)
            batch_images = d.historic_images[d.historic_offset]

            downloaded_files = []
            for img in batch_images:
                local_file = self.file_manager.join(historic_temp_dir, img)
                if self.file_manager.exists(local_file):
                    downloaded_files.append(local_file)

            self._register_local_images_in_db(historic_temp_dir, image_names=batch_images)
            return downloaded_files

        except Exception as exc:
            print(f"Error reading historic batch: {exc}")
            return []

    def _register_local_images_in_db(
        self,
        historic_dir,
        image_names=None,
        db_client=None,
        track_registered=True,
    ):
        d = self.display
        db = db_client or d.db
        try:
            if not db:
                return
            if not self.file_manager.exists(historic_dir):
                return

            if image_names is None:
                local_images = [
                    f
                    for f in self.file_manager.listdir(historic_dir)
                    if f.lower().endswith(self.config.image_extensions)
                ]
            else:
                local_images = list(image_names)

            if not local_images:
                return

            if track_registered:
                pending = [img for img in local_images if img not in d._db_registered_images]
            else:
                pending = local_images
            if not pending:
                return

            existing_rows = db.fetch(
                "SELECT img_name FROM img_results WHERE img_name = ANY(%s)",
                (pending,),
            )
            existing = {row["img_name"] for row in existing_rows} if existing_rows else set()

            images_to_insert = [img for img in pending if img not in existing]
            if images_to_insert:
                query_insert = "INSERT INTO img_results (img_name, result) VALUES (%s, %s)"
                for img_name in images_to_insert:
                    try:
                        db.execute(query_insert, (img_name, "OK"))
                    except Exception as exc:
                        print(f"Error inserting {img_name}: {exc}")

            if track_registered:
                d._db_registered_images.update(pending)
            d.historic_db_registered = True

        except Exception as exc:
            print(f"General error registering images in DB: {exc}")

    def _update_result_in_db(self, img_name, new_value):
        d = self.display
        try:
            query_update = "UPDATE img_results SET result = %s WHERE img_name = %s"
            d.db.execute(query_update, (new_value, img_name))
            d._db_result_cache[img_name] = new_value
        except Exception as exc:
            print(f"Error updating result: {exc}")

    def save_temp_results_to_db(self):
        d = self.display
        if not d.temp_results:
            print("No changes to save")
            return

        print(f"\n{'=' * 60}")
        print("SAVING CHANGES TO DATABASE")
        print(f"{'=' * 60}")
        print(f"Total changes: {len(d.temp_results)}")

        success_count = 0
        failed_count = 0

        for img_name, new_value in d.temp_results.items():
            try:
                self._update_result_in_db(img_name, new_value)
                success_count += 1
            except Exception as exc:
                failed_count += 1
                print(f"Error saving {img_name}: {exc}")

        print(f"{'=' * 60}")
        print(f"{success_count} changes saved successfully")
        if failed_count > 0:
            print(f"{failed_count} changes failed")
        print(f"{'=' * 60}\n")

        d.temp_results.clear()
        print("Temporary changes cleared")

    def sync_images_by_status(
        self,
        historic_dir=None,
        base_dir=None,
        db_client=None,
        progress_callback=None,
    ):
        d = self.display
        db = db_client or d.db
        historic_dir = historic_dir or self.file_manager.join(self.config.temp_dir, HISTORIC_SUBDIR_NAME)
        base_dir = base_dir or str(SYNC_IMAGES_BASE_DIR)

        position_dirs = {
            position: {
                status: self.file_manager.join(base_dir, folder_name)
                for status, folder_name in statuses.items()
            }
            for position, statuses in STATUS_SYNC_DIRS.items()
        }

        for dirs in position_dirs.values():
            for path in dirs.values():
                self.file_manager.makedirs(path, exist_ok=True)

        if not db:
            message = "No database connection available"
            print(message)
            return {"ok": False, "error": message}

        if not self.file_manager.exists(historic_dir):
            message = f"Historic folder not found: {historic_dir}"
            print(message)
            return {"ok": False, "error": message}

        try:
            rows = db.fetch("SELECT img_name, result FROM img_results")
        except Exception as exc:
            message = f"Error fetching image results: {exc}"
            print(message)
            return {"ok": False, "error": message}

        if not rows:
            message = "No image results found in database"
            print(message)
            return {"ok": False, "error": message}

        total_rows = len(rows)
        copied_count = 0
        removed_count = 0
        error_count = 0

        if callable(progress_callback):
            progress_callback(0, total_rows, "Saving dataset")

        for idx, row in enumerate(rows, start=1):
            img_name = row.get("img_name") or row.get("name")
            status = row.get("result")

            if not img_name or status is None:
                if callable(progress_callback):
                    progress_callback(idx, total_rows, "Saving dataset")
                continue

            status = str(status).strip().upper()
            if status not in ("OK", "NOK"):
                if callable(progress_callback):
                    progress_callback(idx, total_rows, "Saving dataset")
                continue

            match = re.search(r"(side|front|diag)", img_name, re.IGNORECASE)
            if not match:
                if callable(progress_callback):
                    progress_callback(idx, total_rows, "Saving dataset")
                continue
            position = match.group(1).lower()

            source_path = self.file_manager.join(historic_dir, img_name)
            if not self.file_manager.exists(source_path):
                if callable(progress_callback):
                    progress_callback(idx, total_rows, "Saving dataset")
                continue

            target_dir = position_dirs[position][status]
            other_status = "NOK" if status == "OK" else "OK"
            other_dir = position_dirs[position][other_status]

            target_path = self.file_manager.join(target_dir, img_name)
            other_path = self.file_manager.join(other_dir, img_name)

            if self.file_manager.exists(other_path):
                try:
                    self.file_manager.remove(other_path)
                    removed_count += 1
                except Exception as exc:
                    error_count += 1
                    print(f"Error removing from wrong folder: {other_path} -> {exc}")

            if not self.file_manager.exists(target_path):
                try:
                    self.file_manager.copy2(source_path, target_path)
                    copied_count += 1
                except Exception as exc:
                    error_count += 1
                    print(f"Error copying {img_name} to {target_dir}: {exc}")

            if callable(progress_callback):
                progress_callback(idx, total_rows, "Saving dataset")

        return {
            "ok": True,
            "rows": total_rows,
            "copied": copied_count,
            "removed": removed_count,
            "errors": error_count,
        }

    def verify_sync_images_by_status(
        self,
        historic_dir=None,
        base_dir=None,
        db_client=None,
        progress_callback=None,
    ):
        db = db_client or self.display.db
        historic_dir = historic_dir or self.file_manager.join(self.config.temp_dir, HISTORIC_SUBDIR_NAME)
        base_dir = base_dir or str(SYNC_IMAGES_BASE_DIR)

        if not db:
            return {"verified": False, "issue_count": 1, "issues": {"db": ["No database connection"]}}

        if not self.file_manager.exists(historic_dir):
            return {
                "verified": False,
                "issue_count": 1,
                "issues": {"historic": [f"Historic folder not found: {historic_dir}"]},
            }

        rows = db.fetch("SELECT img_name, result FROM img_results ORDER BY img_name")
        if not rows:
            return {
                "verified": False,
                "issue_count": 1,
                "issues": {"db_rows": ["img_results returned no rows"]},
            }

        image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        historic_images = sorted(
            name
            for name in self.file_manager.listdir(historic_dir)
            if self.file_manager.is_file(self.file_manager.join(historic_dir, name))
            and any(name.lower().endswith(ext) for ext in image_extensions)
        )
        if not historic_images:
            return {
                "verified": False,
                "issue_count": 1,
                "issues": {"historic_images": ["No image files found in historic folder"]},
            }

        db_status_by_image = defaultdict(set)
        for row in rows:
            img_name = row.get("img_name") or row.get("name")
            result = row.get("result")
            status = "" if result is None else str(result).strip().upper()
            if not img_name or status not in ("OK", "NOK"):
                continue
            db_status_by_image[img_name].add(status)

        if not db_status_by_image:
            return {
                "verified": False,
                "issue_count": 1,
                "issues": {"db_status": ["No valid DB rows with status OK/NOK were found"]},
            }

        total_steps = max(1, len(historic_images) * 2)
        done = 0
        if callable(progress_callback):
            progress_callback(done, total_steps, "Verifying classification")

        expected_folder_by_image = {}
        missing_db_status = []
        conflicting_db_status = []
        invalid_position = []

        for img_name in historic_images:
            statuses = db_status_by_image.get(img_name, set())
            if not statuses:
                missing_db_status.append(img_name)
                done += 1
                if callable(progress_callback):
                    progress_callback(done, total_steps, "Verifying classification")
                continue
            if len(statuses) > 1:
                conflicting_db_status.append(f"{img_name}: {sorted(statuses)}")
                done += 1
                if callable(progress_callback):
                    progress_callback(done, total_steps, "Verifying classification")
                continue

            match = re.search(r"(side|front|diag)", img_name, re.IGNORECASE)
            if not match:
                invalid_position.append(img_name)
                done += 1
                if callable(progress_callback):
                    progress_callback(done, total_steps, "Verifying classification")
                continue

            position = match.group(1).lower()
            status = next(iter(statuses))
            expected_folder_by_image[img_name] = STATUS_SYNC_DIRS[position][status]
            done += 1
            if callable(progress_callback):
                progress_callback(done, total_steps, "Verifying classification")

        status_dirs = [
            self.file_manager.join(base_dir, folder_name)
            for statuses in STATUS_SYNC_DIRS.values()
            for folder_name in statuses.values()
        ]
        actual_locations = defaultdict(list)
        for folder_path in status_dirs:
            if not self.file_manager.exists(folder_path):
                continue
            for name in self.file_manager.listdir(folder_path):
                file_path = self.file_manager.join(folder_path, name)
                if self.file_manager.is_file(file_path):
                    actual_locations[name].append(self.file_manager.basename(folder_path))

        duplicates = {
            img_name: sorted(folder_names)
            for img_name, folder_names in actual_locations.items()
            if len(folder_names) > 1
        }

        missing = []
        wrong_folder = []
        for img_name, expected_folder in expected_folder_by_image.items():
            actual = actual_locations.get(img_name)
            if not actual:
                missing.append(f"{img_name} (expected in {expected_folder})")
            elif actual[0] != expected_folder:
                wrong_folder.append(
                    f"{img_name} (expected {expected_folder}, found {actual[0]})"
                )
            done += 1
            if callable(progress_callback):
                progress_callback(done, total_steps, "Verifying classification")

        issues = {
            "missing_db_status": missing_db_status,
            "conflicting_db_status": conflicting_db_status,
            "invalid_position": invalid_position,
            "duplicates": list(duplicates.keys()),
            "missing": missing,
            "wrong_folder": wrong_folder,
        }
        issue_count = sum(len(v) for v in issues.values())
        verified = issue_count == 0 and len(expected_folder_by_image) == len(historic_images)

        if callable(progress_callback):
            progress_callback(total_steps, total_steps, "Verifying classification")

        return {
            "verified": verified,
            "issue_count": issue_count,
            "issues": issues,
            "historic_images": len(historic_images),
            "mapped_images": len(expected_folder_by_image),
        }

    def get_piece_date(self):
        d = self.display
        if not d.historic_images or d.historic_offset >= len(d.historic_images):
            return "N/A"

        try:
            batch = d.historic_images[d.historic_offset]
            if not batch:
                return "N/A"

            first_image = batch[0]
            historic_dir = self.file_manager.join(self.config.temp_dir, HISTORIC_SUBDIR_NAME)
            image_path = self.file_manager.join(historic_dir, first_image)
            if self.file_manager.exists(image_path):
                import datetime

                mtime = self.file_manager.getmtime(image_path)
                dt = datetime.datetime.fromtimestamp(mtime)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            return "N/A"
        except Exception as exc:
            print(f"Error getting piece date: {exc}")
            return "N/A"

    def get_result_for_image(self, img_name):
        d = self.display
        if img_name in d.temp_results:
            return d.temp_results[img_name]

        cached_result = d._db_result_cache.get(img_name)
        if cached_result is not None:
            return cached_result

        result_text = "N/A"
        if d.db:
            try:
                query = "SELECT result FROM img_results WHERE img_name = %s"
                result = d.db.fetch(query, (img_name,))
                if result and len(result) > 0:
                    result_value = result[0]["result"]
                    result_text = str(result_value) if result_value is not None else "N/A"
            except Exception as exc:
                result_text = "Error"
                print(f"Error querying result for {img_name}: {exc}")

        d._db_result_cache[img_name] = result_text
        return result_text

    def toggle_result(self, img_name, current_value=None):
        if not img_name:
            return
        base_value = current_value if current_value in ("OK", "NOK") else self.get_result_for_image(img_name)
        new_value = "NOK" if base_value == "OK" else "OK"
        self._update_result_in_db(img_name, new_value)
        self.display.temp_results[img_name] = new_value

    def handle_ui_action(self, action, **payload):
        d = self.display

        if not self.db_connected:
            return

        if action == "enter_historic_mode":
            self.enter_historic_mode()
        elif action == "exit_historic_mode":
            self.exit_historic_mode()
        elif action == "request_exit":
            d.exit_requested = True
        elif action == "request_remote_start":
            self.start_remote_process()
        elif action == "request_remote_stop":
            self.stop_remote_process("button")
        elif action == "next_historic_batch":
            self.next_historic_batch()
        elif action == "prev_historic_batch":
            self.prev_historic_batch()
        elif action == "open_piece_date_dialog":
            d.show_piece_date_dialog = True
        elif action == "close_piece_date_dialog":
            d.show_piece_date_dialog = False
        elif action == "open_reset_confirm":
            d.show_reset_confirm = True
            d.show_delete_confirm = False
        elif action == "cancel_reset_confirm":
            d.show_reset_confirm = False
        elif action == "confirm_reset":
            d.show_reset_confirm = False
            self.start_reset_async()
        elif action == "open_delete_confirm":
            d.show_delete_confirm = True
            d.show_reset_confirm = False
        elif action == "cancel_delete_confirm":
            d.show_delete_confirm = False
        elif action == "confirm_delete":
            d.show_delete_confirm = False
            self.perform_delete_current_piece()
        elif action == "sync_images_by_status":
            self.start_sync_images_by_status_async()
        elif action == "toggle_result":
            self.toggle_result(payload.get("img_name"), payload.get("result_value"))
        elif action == "dismiss_no_images_dialog":
            d.show_no_images_dialog = False
            d.no_images_dialog_message = "No images available"
        elif action == "search_focus":
            d.search_active = True
            self.collect_available_jsns()
            self.update_suggestions()
        elif action == "search_blur":
            d.search_active = False
            d.filtered_suggestions = []
        elif action == "search_append_digit":
            digit = payload.get("digit")
            if digit is not None and len(d.search_jsn) < 21 and str(digit).isdigit():
                d.search_jsn += str(digit)
                self.update_suggestions()
        elif action == "search_backspace":
            d.search_jsn = d.search_jsn[:-1]
            self.update_suggestions()
        elif action == "search_move_up":
            if d.filtered_suggestions:
                d.selected_suggestion_idx = max(-1, d.selected_suggestion_idx - 1)
        elif action == "search_move_down":
            if d.filtered_suggestions:
                d.selected_suggestion_idx = min(
                    len(d.filtered_suggestions) - 1,
                    d.selected_suggestion_idx + 1,
                )
        elif action == "search_select_suggestion":
            jsn_value = payload.get("jsn")
            if jsn_value:
                d.search_jsn = str(jsn_value)[:21]
            self.perform_jsn_search()
            d.search_active = False
            d.filtered_suggestions = []
        elif action == "search_submit":
            if d.selected_suggestion_idx >= 0 and d.selected_suggestion_idx < len(d.filtered_suggestions):
                d.search_jsn = d.filtered_suggestions[d.selected_suggestion_idx][:21]
            self.perform_jsn_search()
        elif action == "search_cancel":
            d.search_active = False
            d.filtered_suggestions = []

    def run(self):
        self.initialize()
        try:
            while True:
                if not self.db_connected:
                    self.try_connect_db("runtime-loop")
                    self.display.image_paths = []
                    self.display.show_image_grid(
                        [],
                        cols=self.config.display_cols,
                        rows=self.config.display_rows,
                    )
                    continue

                if not self.historic_bootstrap_loading and not self.historic_bootstrap_complete:
                    self._register_historic_local_dir_on_startup()

                if self.display.remote_action_request:
                    action = self.display.remote_action_request
                    self.display.remote_action_request = None
                    if action == "start":
                        self.start_remote_process()
                    elif action == "stop":
                        self.stop_remote_process("button")

                self._process_remote_events()

                if self.sftp_app and not self.sftp_connected and time.monotonic() >= self.next_reconnect_ts:
                    self.try_connect("periodic-retry")

                if self.display.exit_requested:
                    break

                if self.display.historic_mode:
                    images = self.download_historic_batch(
                        self.config.temp_dir,
                        max_images=self.config.max_images,
                    )
                else:
                    images = []
                    if self.sftp_connected and self.sftp_app:
                        images = self._download_live_images_remote()
                        if not self.sftp_app.sftp_client:
                            self.stop_remote_process("sftp-disconnect")
                            self.handle_disconnect("live-download-failure")
                        elif not images:
                            self.logger.info(
                                "[LOCAL] Falling back to local live images",
                                allow_repeat=True,
                            )

                    if not images:
                        images = self._download_live_images_local()

                self.display.image_paths = images
                self.display.show_image_grid(
                    images,
                    cols=self.config.display_cols,
                    rows=self.config.display_rows,
                )
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        self.stop_remote_process("exit")
        if self.sftp_app:
            self.sftp_app.disconnect_sftp()
        self.stop_historic_download_worker()
        self.display.close()


def check_historic_images():
    import paramiko
    from settings import get_sftp_settings

    file_manager = FileManager()
    sftp_settings = get_sftp_settings()
    hostname = sftp_settings["hostname"]
    port = sftp_settings["port"]
    username = sftp_settings["username"]
    password = sftp_settings["password"]
    remote_hist_dir = REMOTE_HIST_DISPLAY_DIR
    local_hist_dir = file_manager.join(str(TMP_DISPLAY_DIR), HISTORIC_SUBDIR_NAME)
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp")

    print("\n" + "=" * 70)
    print("HISTORIC IMAGES VERIFICATION")
    print("=" * 70)

    try:
        print("Connecting to SFTP server...")
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(
            hostname=hostname,
            port=port,
            username=username,
            password=password,
            timeout=10,
        )
        sftp_client = ssh_client.open_sftp()
        print("Connection successful\n")

        try:
            file_manager.sftp_chdir(sftp_client, remote_hist_dir)
            remote_files = file_manager.sftp_listdir(sftp_client)
            remote_images = [f for f in remote_files if f.lower().endswith(image_extensions)]
            remote_count = len(remote_images)
        except FileNotFoundError:
            remote_count = 0
            print(f"Remote folder {remote_hist_dir} does not exist")

        if file_manager.exists(local_hist_dir):
            local_files = file_manager.listdir(local_hist_dir)
            local_images = [f for f in local_files if f.lower().endswith(image_extensions)]
            local_count = len(local_images)
        else:
            local_count = 0
            print(f"Local folder {local_hist_dir} does not exist")

        print("RESULTS:")
        print("=" * 70)
        print(f"Images on remote server ({remote_hist_dir}):")
        print(f"   Total: {remote_count} files")
        print(f"\nImages in local folder ({local_hist_dir}):")
        print(f"   Total: {local_count} files")
        print(f"\nPending images to download: {max(0, remote_count - local_count)}")

        if local_count == remote_count and remote_count > 0:
            print("\nSYNCHRONIZED - All images are downloaded")
        elif local_count > remote_count:
            print("\nATTENTION - More local images than remote")
        elif remote_count > local_count:
            print("\nNEW IMAGES AVAILABLE - Open historic mode to download them")
        else:
            print("\nNo images in any location")

        print("=" * 70)

        sftp_client.close()
        ssh_client.close()

    except paramiko.AuthenticationException:
        print("Error: Authentication failed")
    except paramiko.SSHException as exc:
        print(f"SSH Error: {str(exc)}")
    except Exception as exc:
        print(f"Error: {str(exc)}")

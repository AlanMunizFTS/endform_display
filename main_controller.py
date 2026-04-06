import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Event, Process, Queue
from threading import Thread

from file_manager import FileManager
from paths_config import (
    ANNOTATED_LOCAL_DIR,
    ANNOTATED_SUBDIR_NAME,
    FINAL_CLASSIFICATION_DIR,
    FINAL_CLASSIFICATION_DIRS,
    HISTORIC_LOCAL_DIR,
    HISTORIC_SUBDIR_NAME,
    REMOTE_ANNOTATED_DIR,
    REMOTE_HIST_DISPLAY_DIR,
    REMOTE_TEST_DISPLAY_DIR,
    STATUS_SYNC_DIRS,
    SYNC_IMAGES_BASE_DIR,
    TMP_DISPLAY_DIR,
)
from sftp_app import SFTPApp
from settings import is_remote_db_enabled
from utilities.log import get_logger, install_print_logger
from state_package import export_display_state, import_display_state


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


def _download_images_background_worker(
    hostname,
    port,
    username,
    password,
    remote_dir,
    local_temp_dir,
    check_interval=30,
    reconnect_interval=10,
    stop_event=None,
    worker_label="HIST_SYNC_SSH",
    verbose=False,
):
    import paramiko

    install_print_logger(reset=False)
    logger = get_logger()
    file_manager = FileManager()

    image_extensions = (".png", ".jpg", ".jpeg", ".bmp")
    ssh_client = None
    sftp_client = None

    file_manager.makedirs(local_temp_dir, exist_ok=True)

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

    logger.info(f"[{worker_label}] Worker started", allow_repeat=True)
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            if sftp_client is None:
                try:
                    logger.info(
                        f"[{worker_label}] Connecting to {hostname}:{port} as {username}",
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
                    logger.info(f"[{worker_label}] Connection successful", allow_repeat=True)
                except Exception as exc:
                    logger.error(
                        f"[{worker_label}] Connection failed: {exc}",
                        allow_repeat=True,
                    )
                    close_connections()
                    _sleep_with_stop(stop_event, reconnect_interval)
                    continue

            try:
                existing_local = (
                    set(file_manager.listdir(local_temp_dir))
                    if file_manager.exists(local_temp_dir)
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
                    local_file = file_manager.join(local_temp_dir, img)
                    file_manager.sftp_get(sftp_client, img, local_file)
                    downloaded_count += 1

                if downloaded_count:
                    logger.info(
                        f"[{worker_label}] Downloaded {downloaded_count} new images",
                        allow_repeat=True,
                    )

                _sleep_with_stop(stop_event, check_interval)

            except FileNotFoundError:
                logger.warn(
                    f"[{worker_label}] Remote folder not found: {remote_dir}",
                    allow_repeat=True,
                )
                _sleep_with_stop(stop_event, check_interval)
            except Exception as exc:
                logger.error(
                    f"[{worker_label}] Sync error: {exc}",
                    allow_repeat=True,
                )
                close_connections()
                _sleep_with_stop(stop_event, reconnect_interval)
    finally:
        close_connections()
        logger.info(f"[{worker_label}] Worker stopped", allow_repeat=True)


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
                if file_manager.is_file(path):
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
    rotation_state,
    logger,
    image_extensions,
    max_images=7,
):
    if not app or not app.sftp_client:
        return []

    app.file_manager.makedirs(local_path, exist_ok=True)
    downloaded_files = []

    try:
        files = app.list_remote_files(remote_path)
        images = [f for f in files if f.lower().endswith(image_extensions)]
        images.sort(reverse=True)

        if not images:
            return []

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

        # Clear existing images from local_path before downloading new batch
        # so tmp_display never exceeds max_images files at any point
        try:
            for fname in app.file_manager.listdir(local_path):
                if fname.lower().endswith(image_extensions):
                    try:
                        app.file_manager.remove(app.file_manager.join(local_path, fname))
                    except Exception:
                        pass
        except Exception:
            pass

        for img_name in selected_images:
            local_file = app.file_manager.join(local_path, img_name)
            remote_img_path = app.join_remote_path(remote_path, img_name)
            try:
                app.download_file(remote_img_path, local_file)
                downloaded_files.append(local_file)
            except FileNotFoundError:
                return []

        return downloaded_files

    except Exception as exc:
        logger.error(f"[SSH] Error downloading live images: {exc}", allow_repeat=True)
        try:
            app.disconnect_sftp()
        except Exception:
            pass
        return []


def _process_remote_event_impl(msg, display, logger):
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
        if "done signal sent to plc" in lower_line:
            if getattr(display, "manual_capture_pending", False):
                display.manual_capture_pending = False
                _show_capture_modal(display, "captured", (0, 150, 0), duration_sec=0.5)

    elif msg_type == "stderr":
        return


def _show_capture_modal(display, message, color, duration_sec=None):
    if display is None:
        return

    modal_fn = getattr(display, "show_capture_modal", None)
    if callable(modal_fn):
        modal_fn(message, color, duration_sec=duration_sec)
        return

    display.capture_modal_text = str(message or "").strip()
    display.capture_modal_color = tuple(color) if color is not None else (0, 0, 200)
    display.capture_modal_visible = bool(display.capture_modal_text)
    if duration_sec is None:
        display.capture_modal_expires_at = 0.0
    else:
        display.capture_modal_expires_at = time.time() + max(0.05, float(duration_sec))


def _clear_capture_modal(display):
    if display is None:
        return

    clear_fn = getattr(display, "clear_capture_modal", None)
    if callable(clear_fn):
        clear_fn()
        return

    display.capture_modal_visible = False
    display.capture_modal_text = ""
    display.capture_modal_expires_at = 0.0


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
    rotation_state,
    logger,
    max_images=7,
):
    return _download_live_images_remote_impl(
        app=app,
        remote_path=remote_path,
        local_path=local_path,
        rotation_state=rotation_state,
        logger=logger,
        image_extensions=(".png", ".jpg", ".jpeg", ".bmp"),
        max_images=max_images,
    )


def process_remote_event(msg, display, logger):
    _process_remote_event_impl(
        msg=msg,
        display=display,
        logger=logger,
    )


@dataclass
class ControllerConfig:
    image_extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp")
    live_rescan_interval_sec: float = 2.0
    live_batch_rotation_interval_sec: float = 1.0
    sftp_reconnect_interval_sec: float = 10.0
    db_reconnect_interval_sec: float = 3.0
    remote_db_polling_enabled: bool = field(default_factory=is_remote_db_enabled)
    local_model_results_polling_enabled: bool = True
    local_model_results_table: str = "model_results_local"
    remote_db_table: str = "model_results"
    remote_db_columns: tuple = ("img_name", "class_name", "confidence")
    remote_db_query_limit: int = 25
    remote_db_target_sync_batch: int = 25
    remote_db_max_scan_pages: int = 20
    remote_db_forward_scan_ratio: float = 0.7
    remote_db_success_interval_sec: float = 0.0
    remote_db_idle_backoff_sec: float = 2.0
    remote_db_error_backoff_sec: float = 5.0
    max_images: int = 7
    remote_command: str = (
        "sh -lc 'echo $$; "
        "cd ~/Vision-Standard 2>/dev/null || cd ~/vision-standard; "
        "stdbuf -oL -eL python3 -u main.py -f art_1861_endform -p omron -d teledyne 2>&1'"
    )
    temp_dir: str = field(default_factory=lambda: str(TMP_DISPLAY_DIR))
    remote_live_dir: str = REMOTE_TEST_DISPLAY_DIR
    remote_hist_dir: str = REMOTE_HIST_DISPLAY_DIR
    remote_annotated_dir: str = REMOTE_ANNOTATED_DIR
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
        self.stdin_queue = None

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
        self._pending_remote_images = None
        self.historic_bootstrap_loading = False
        self.historic_bootstrap_complete = False
        self.historic_bootstrap_thread = None
        self.sync_worker_thread = None
        self.reset_worker_thread = None
        self.rebuild_worker_thread = None
        self.export_worker_thread = None
        self.import_worker_thread = None
        self.remote_db_poll_thread = None
        self.remote_db_stop_event = None
        self._remote_db_placeholder_warned = False
        self._remote_db_idle_logged = False
        self.remote_db_forward_cursor_id = 0
        self.remote_db_backfill_cursor_id = 0
        self.last_historic_check = 0.0
        self.dataset_transfer_active = False

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

    def _resolve_temp_subdir(self, subdir_name, default_path):
        if self.config.temp_dir == str(TMP_DISPLAY_DIR):
            return str(default_path)
        return self.file_manager.join(self.config.temp_dir, subdir_name)

    def _get_visible_historic_dir(self):
        return self._resolve_temp_subdir(ANNOTATED_SUBDIR_NAME, ANNOTATED_LOCAL_DIR)

    def _get_export_historic_dir(self):
        return self._resolve_temp_subdir(HISTORIC_SUBDIR_NAME, HISTORIC_LOCAL_DIR)

    def _list_local_image_names(self, directory, require_jsn_prefix=False):
        if not self.file_manager.exists(directory):
            return []
        images = [
            name
            for name in self.file_manager.listdir(directory)
            if name.lower().endswith(self.config.image_extensions)
        ]
        if require_jsn_prefix:
            images = [name for name in images if name.startswith("11861")]
        return images

    def _get_visible_historic_image_set(self):
        return set(self._list_local_image_names(self._get_visible_historic_dir()))

    def _get_visible_historic_image_snapshot(self):
        return sorted(self._get_visible_historic_image_set())

    def _clear_tmp_display(self):
        tmp_dir = self.config.temp_dir
        if not self.file_manager.exists(tmp_dir):
            return
        for fname in self.file_manager.listdir(tmp_dir):
            if fname.lower().endswith(self.config.image_extensions):
                try:
                    self.file_manager.remove(self.file_manager.join(tmp_dir, fname))
                except Exception:
                    pass

    def initialize(self):
        self._clear_tmp_display()
        if not self.db_connected:
            self.try_connect_db("startup")
        if self.db_connected:
            self._register_historic_local_dir_on_startup()
        self.start_remote_db_polling()

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

    def _resolve_remote_db_ssh_credentials(self):
        if self.sftp_credentials is not None:
            return dict(self.sftp_credentials)

        try:
            from settings import get_sftp_settings

            return get_sftp_settings()
        except Exception as exc:
            self.logger.warn(
                f"[REMOTE_DB] SSH credentials unavailable for remote DB polling: {exc}",
                allow_repeat=True,
            )
            return None

    def _is_remote_db_table_placeholder(self):
        table_name = str(getattr(self.config, "remote_db_table", "") or "").strip()
        return not table_name

    def _is_local_model_results_table_placeholder(self):
        table_name = str(getattr(self.config, "local_model_results_table", "") or "").strip()
        return not table_name

    def _quote_remote_db_identifier(self, identifier):
        parts = [part.strip() for part in str(identifier).split(".")]
        if not parts or any(not part for part in parts):
            raise ValueError(f"Invalid SQL identifier: {identifier!r}")
        return ".".join(f'"{part.replace(chr(34), chr(34) * 2)}"' for part in parts)

    def _build_model_result_query(self, table_name):
        normalized_table_name = str(table_name or "").strip()
        if not normalized_table_name:
            return None

        quoted_table_name = self._quote_remote_db_identifier(normalized_table_name)
        selected_columns = ", ".join(
            self._quote_remote_db_identifier(column_name)
            for column_name in self.config.remote_db_columns
        )
        return (
            f"SELECT \"id\", {selected_columns} FROM {quoted_table_name} "
            f"WHERE \"id\" > %s "
            f"ORDER BY \"id\" ASC LIMIT %s"
        )

    def _build_remote_db_query(self):
        if self._is_remote_db_table_placeholder():
            return None
        return self._build_model_result_query(self.config.remote_db_table)

    def _serialize_remote_db_row(self, row):
        return json.dumps(row, ensure_ascii=True, sort_keys=True, default=str)

    def _collect_remote_model_result_rows(self, remote_db, query, after_id, limit, scan_label):
        rows = remote_db.fetch(query, (after_id, limit))
        if rows:
            self.logger.info(
                f"[REMOTE_DB] Retrieved {len(rows)} rows for {scan_label} scan using LIMIT {limit}",
            )
        for idx, row in enumerate(rows, start=1):
            self.logger.info(
                f"[REMOTE_DB] {scan_label} row {idx}: {self._serialize_remote_db_row(row)}",
            )
        return rows

    def _fetch_existing_local_classified_images(self, local_db, image_names):
        if not image_names:
            return {}

        rows = local_db.fetch(
            "SELECT id, img_name FROM classified_images WHERE img_name = ANY(%s)",
            (image_names,),
        )
        return {
            row["img_name"]: {"id": row["id"]}
            for row in rows or []
            if row.get("img_name") and row.get("id") is not None
        }

    def _normalize_remote_defect_class_name(self, class_name):
        normalized = str(class_name or "").strip()
        if normalized.upper() == "NOK":
            return "STREAKED"
        return normalized

    def _insert_local_classification_defects(self, local_db, matched_rows):
        synced_img_names = []
        synced_remote_ids = []
        recalculated_jsns = set()
        for row in matched_rows:
            img_name = row["img_name"]
            class_name = self._normalize_remote_defect_class_name(row.get("class_name"))
            local_db.execute(
                "INSERT INTO classified_image_defects "
                "(classified_image_id, class_name, confidence) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (classified_image_id, class_name, confidence) DO NOTHING",
                (
                    row["classified_image_id"],
                    class_name,
                    row.get("confidence"),
                ),
            )
            synced_img_names.append(img_name)
            synced_remote_ids.append(row["remote_id"])
            recalculated_jsns.add(img_name.split("_")[0] if "_" in img_name else img_name)

        for jsn in sorted(recalculated_jsns):
            self._recalculate_piece_result(jsn, db_client=local_db)

        return synced_img_names, synced_remote_ids

    def _delete_model_result_rows(self, source_db, table_name, row_ids):
        if not row_ids:
            return 0

        normalized_table_name = str(table_name or "").strip()
        if not normalized_table_name:
            return 0

        quoted_table_name = self._quote_remote_db_identifier(normalized_table_name)
        return source_db.execute(
            f"DELETE FROM {quoted_table_name} WHERE id = ANY(%s)",
            (row_ids,),
        )

    def _delete_remote_model_results(self, remote_db, remote_ids):
        if not remote_ids:
            return 0

        return self._delete_model_result_rows(
            source_db=remote_db,
            table_name=self.config.remote_db_table,
            row_ids=remote_ids,
        )

    def _scan_remote_rows_for_sync(
        self,
        remote_db,
        local_db,
        query,
        start_after_id,
        max_pages,
        target_sync_batch,
        scan_label,
        excluded_remote_ids=None,
    ):
        page_limit = max(1, int(self.config.remote_db_query_limit))
        seen_remote_ids = set(excluded_remote_ids or [])
        matched_rows = []
        missing_local = []
        missing_local_seen = set()
        scanned_count = 0
        pages_scanned = 0
        current_after_id = max(0, int(start_after_id or 0))

        while pages_scanned < max_pages and len(matched_rows) < target_sync_batch:
            rows = self._collect_remote_model_result_rows(
                remote_db=remote_db,
                query=query,
                after_id=current_after_id,
                limit=page_limit,
                scan_label=scan_label,
            )
            pages_scanned += 1

            if not rows:
                current_after_id = 0
                break

            normalized_rows = []
            for row in rows:
                remote_id = row.get("id")
                img_name = str(row.get("img_name") or "").strip()
                if remote_id is None or not img_name:
                    continue
                normalized_rows.append(
                    {
                        "remote_id": int(remote_id),
                        "img_name": img_name,
                        "class_name": row.get("class_name"),
                        "confidence": row.get("confidence"),
                    }
                )

            if normalized_rows:
                current_after_id = max(row["remote_id"] for row in normalized_rows)

            scanned_count += len(normalized_rows)

            page_image_names = [
                row["img_name"]
                for row in normalized_rows
                if row["remote_id"] not in seen_remote_ids
            ]
            existing_images = self._fetch_existing_local_classified_images(local_db, page_image_names)

            for row in normalized_rows:
                if row["remote_id"] in seen_remote_ids:
                    continue
                seen_remote_ids.add(row["remote_id"])

                local_image = existing_images.get(row["img_name"])
                if local_image is None:
                    if row["img_name"] not in missing_local_seen:
                        missing_local_seen.add(row["img_name"])
                        missing_local.append(row["img_name"])
                    continue

                matched_rows.append(
                    {
                        **row,
                        "classified_image_id": local_image["id"],
                    }
                )

            if len(rows) < page_limit:
                current_after_id = 0
                break

        return {
            "matched_rows": matched_rows,
            "missing_local": missing_local,
            "scanned_count": scanned_count,
            "pages_scanned": pages_scanned,
            "next_after_id": current_after_id,
            "seen_remote_ids": seen_remote_ids,
        }

    def _sync_remote_rows_into_local_classified_images(
        self,
        matched_rows,
        missing_local,
        local_db,
        remote_db,
        source_table=None,
    ):
        candidate_rows = []
        for row in matched_rows:
            img_name = str(row.get("img_name") or "").strip()
            if not img_name:
                continue
            candidate_rows.append(
                {
                    "remote_id": row["remote_id"],
                    "classified_image_id": row["classified_image_id"],
                    "img_name": img_name,
                    "class_name": row.get("class_name"),
                    "confidence": row.get("confidence"),
                }
            )

        if not candidate_rows:
            return {
                "candidate_count": 0,
                "matched_count": 0,
                "synced_count": 0,
                "deleted_count": 0,
                "missing_local": list(missing_local or []),
                "synced_img_names": [],
            }

        synced_img_names, synced_remote_ids = self._insert_local_classification_defects(local_db, candidate_rows)
        deleted_count = self._delete_model_result_rows(
            source_db=remote_db,
            table_name=source_table or self.config.remote_db_table,
            row_ids=synced_remote_ids,
        )

        return {
            "candidate_count": len(candidate_rows),
            "matched_count": len(candidate_rows),
            "synced_count": len(synced_img_names),
            "deleted_count": deleted_count,
            "missing_local": list(missing_local or []),
            "synced_img_names": synced_img_names,
        }

    def _run_local_model_result_poll_iteration(self, local_db):
        if not getattr(self.config, "local_model_results_polling_enabled", False):
            return {"ran": False, "has_activity": False, "synced_count": 0}

        query = self._build_model_result_query(self.config.local_model_results_table)
        if query is None:
            return {"ran": False, "has_activity": False, "synced_count": 0}
        try:
            scan = self._scan_remote_rows_for_sync(
                remote_db=local_db,
                local_db=local_db,
                query=query,
                start_after_id=0,
                max_pages=max(1, int(self.config.remote_db_max_scan_pages)),
                target_sync_batch=max(1, int(self.config.remote_db_target_sync_batch)),
                scan_label="local",
            )
            sync_summary = self._sync_remote_rows_into_local_classified_images(
                scan["matched_rows"],
                missing_local=scan["missing_local"],
                local_db=local_db,
                remote_db=local_db,
                source_table=self.config.local_model_results_table,
            )
            has_activity = (
                scan["scanned_count"] > 0
                or sync_summary["candidate_count"] > 0
                or sync_summary["synced_count"] > 0
                or sync_summary["deleted_count"] > 0
                or bool(sync_summary["missing_local"])
            )
            if has_activity:
                self.logger.info(
                    "[LOCAL_MODEL_RESULTS] Sync summary: "
                    f"scanned={scan['scanned_count']}, "
                    f"pages={scan['pages_scanned']}, "
                    f"candidates={sync_summary['candidate_count']}, "
                    f"matched={sync_summary['matched_count']}, "
                    f"synced={sync_summary['synced_count']}, "
                    f"deleted_local={sync_summary['deleted_count']}",
                )
            if sync_summary["missing_local"]:
                self.logger.info(
                    "[LOCAL_MODEL_RESULTS] Missing local classified_images rows: "
                    + ", ".join(sync_summary["missing_local"][:10]),
                )
            return {
                "ran": True,
                "has_activity": has_activity,
                "synced_count": sync_summary["synced_count"],
                "deleted_count": sync_summary["deleted_count"],
            }
        except Exception as exc:
            self.logger.debug(f"[LOCAL_MODEL_RESULTS] Poll iteration skipped: {exc}")
            return {"ran": False, "has_activity": False, "synced_count": 0}

    def _run_remote_model_result_poll_iteration(self, local_db):
        query = self._build_remote_db_query()
        if query is None:
            if not self._remote_db_placeholder_warned:
                self.logger.warn(
                    "[REMOTE_DB] Polling skipped: configure ControllerConfig.remote_db_table before querying PostgreSQL.",
                    allow_repeat=True,
                )
                self._remote_db_placeholder_warned = True
            return {"skipped": True, "delay": max(0.0, float(self.config.remote_db_error_backoff_sec))}

        self._remote_db_placeholder_warned = False

        ssh_credentials = self._resolve_remote_db_ssh_credentials()
        if not ssh_credentials:
            return {
                "skipped": True,
                "delay": max(
                    float(self.config.remote_db_success_interval_sec),
                    float(self.config.remote_db_idle_backoff_sec),
                ),
            }

        remote_db = None
        try:
            from db import get_remote_db_connection_via_ssh

            total_page_budget = max(1, int(self.config.remote_db_max_scan_pages))
            forward_page_budget = min(
                total_page_budget,
                max(1, int(round(total_page_budget * float(self.config.remote_db_forward_scan_ratio)))),
            )
            backfill_page_budget = max(0, total_page_budget - forward_page_budget)
            target_sync_batch = max(1, int(self.config.remote_db_target_sync_batch))

            self.logger.debug(
                f"[REMOTE_DB] Opening dedicated SSH tunnel to {ssh_credentials['hostname']}:{ssh_credentials['port']}",
            )
            remote_db = get_remote_db_connection_via_ssh(
                ssh_host=ssh_credentials["hostname"],
                ssh_port=ssh_credentials["port"],
                ssh_username=ssh_credentials["username"],
                ssh_password=ssh_credentials["password"],
            )
            self.logger.debug("[REMOTE_DB] Remote PostgreSQL connection successful")

            forward_scan = self._scan_remote_rows_for_sync(
                remote_db=remote_db,
                local_db=local_db,
                query=query,
                start_after_id=self.remote_db_forward_cursor_id,
                max_pages=forward_page_budget,
                target_sync_batch=target_sync_batch,
                scan_label="forward",
            )
            self.remote_db_forward_cursor_id = forward_scan["next_after_id"]

            combined_matched_rows = list(forward_scan["matched_rows"])
            combined_missing_local = list(forward_scan["missing_local"])
            scanned_count = forward_scan["scanned_count"]
            pages_scanned = forward_scan["pages_scanned"]

            remaining_target = max(0, target_sync_batch - len(combined_matched_rows))
            if remaining_target > 0 and backfill_page_budget > 0:
                backfill_scan = self._scan_remote_rows_for_sync(
                    remote_db=remote_db,
                    local_db=local_db,
                    query=query,
                    start_after_id=self.remote_db_backfill_cursor_id,
                    max_pages=backfill_page_budget,
                    target_sync_batch=remaining_target,
                    scan_label="backfill",
                    excluded_remote_ids=forward_scan["seen_remote_ids"],
                )
                self.remote_db_backfill_cursor_id = backfill_scan["next_after_id"]
                combined_matched_rows.extend(backfill_scan["matched_rows"])
                combined_missing_local.extend(backfill_scan["missing_local"])
                scanned_count += backfill_scan["scanned_count"]
                pages_scanned += backfill_scan["pages_scanned"]

            sync_summary = self._sync_remote_rows_into_local_classified_images(
                combined_matched_rows,
                missing_local=combined_missing_local,
                local_db=local_db,
                remote_db=remote_db,
                source_table=self.config.remote_db_table,
            )
            has_activity = (
                scanned_count > 0
                or sync_summary["candidate_count"] > 0
                or sync_summary["synced_count"] > 0
                or sync_summary["deleted_count"] > 0
                or bool(sync_summary["missing_local"])
            )
            if has_activity:
                self._remote_db_idle_logged = False
                self.logger.info(
                    "[REMOTE_DB] Local sync summary: "
                    f"scanned={scanned_count}, "
                    f"pages={pages_scanned}, "
                    f"candidates={sync_summary['candidate_count']}, "
                    f"matched={sync_summary['matched_count']}, "
                    f"synced={sync_summary['synced_count']}, "
                    f"deleted_remote={sync_summary['deleted_count']}",
                )
            if sync_summary["missing_local"]:
                self.logger.info(
                    "[REMOTE_DB] Missing local classified_images rows: "
                    + ", ".join(sync_summary["missing_local"][:10]),
                )
            elif not has_activity and not self._remote_db_idle_logged:
                idle_backoff = max(0.0, float(self.config.remote_db_idle_backoff_sec))
                self.logger.info(
                    f"[REMOTE_DB] No remote metadata available to sync; retrying in {idle_backoff:.1f}s"
                )
                self._remote_db_idle_logged = True

            if sync_summary["synced_count"] > 0 or sync_summary["deleted_count"] > 0:
                delay = max(0.0, float(self.config.remote_db_success_interval_sec))
            else:
                delay = max(
                    float(self.config.remote_db_success_interval_sec),
                    float(self.config.remote_db_idle_backoff_sec),
                )
            return {"skipped": False, "delay": delay, "has_activity": has_activity}
        except Exception as exc:
            self._remote_db_idle_logged = False
            self.logger.error(f"[REMOTE_DB] Poll iteration failed: {exc}", allow_repeat=True)
            return {"skipped": False, "delay": max(0.0, float(self.config.remote_db_error_backoff_sec))}
        finally:
            if remote_db is not None:
                try:
                    remote_db.close()
                except Exception:
                    pass

    def _run_remote_db_poll_iteration(self):
        local_db = getattr(self.display, "db", None)
        if not self.db_connected or local_db is None:
            self.logger.warn(
                "[REMOTE_DB] Local PostgreSQL unavailable; skipping remote metadata sync.",
            )
            return max(0.0, float(self.config.remote_db_error_backoff_sec))

        local_result = self._run_local_model_result_poll_iteration(local_db)
        remote_result = self._run_remote_model_result_poll_iteration(local_db)

        if local_result.get("has_activity"):
            return max(0.0, float(self.config.remote_db_success_interval_sec))
        return max(0.0, float(remote_result.get("delay", self.config.remote_db_idle_backoff_sec)))

    def _remote_db_polling_worker(self):
        self.logger.info("[REMOTE_DB] Polling worker started", allow_repeat=True)
        try:
            while self.remote_db_stop_event is not None and not self.remote_db_stop_event.is_set():
                delay_sec = self._run_remote_db_poll_iteration()
                if self.remote_db_stop_event is not None and self.remote_db_stop_event.is_set():
                    break
                _sleep_with_stop(self.remote_db_stop_event, delay_sec)
        finally:
            self.logger.info("[REMOTE_DB] Polling worker stopped", allow_repeat=True)

    def start_remote_db_polling(self):
        if not getattr(self.config, "remote_db_polling_enabled", False):
            self.logger.info("[REMOTE_DB] Polling disabled by config", allow_repeat=True)
            return False
        if self.remote_db_poll_thread is not None and self.remote_db_poll_thread.is_alive():
            return False

        self.remote_db_stop_event = Event()
        self.remote_db_poll_thread = Thread(
            target=self._remote_db_polling_worker,
            name="remote-db-polling",
            daemon=True,
        )
        self.remote_db_poll_thread.start()
        return True

    def stop_remote_db_polling(self):
        if self.remote_db_stop_event is not None:
            self.remote_db_stop_event.set()
        if self.remote_db_poll_thread is not None:
            try:
                self.remote_db_poll_thread.join(timeout=5)
            except Exception:
                pass
        self.remote_db_poll_thread = None
        self.remote_db_stop_event = None

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
        """Register all visible historic images on startup if not already in img_results."""
        if self.dataset_transfer_active:
            return
        if not self.db_connected:
            return

        visible_dir = self._get_visible_historic_dir()
        if not self.file_manager.exists(visible_dir):
            self.historic_bootstrap_complete = True
            return

        self.historic_bootstrap_loading = True
        self.historic_bootstrap_thread = Thread(target=self._register_historic_local_dir_worker, daemon=True)
        self.historic_bootstrap_thread.start()

    def _register_historic_local_dir_worker(self):
        """Worker thread for registering visible historic images on startup."""
        try:
            if self.dataset_transfer_active:
                return
            visible_dir = self._get_visible_historic_dir()
            if not self.file_manager.exists(visible_dir):
                return

            local_images = self._list_local_image_names(visible_dir)

            if not local_images:
                return

            # Check existing in DB
            existing_rows = self.display.db.fetch(
                "SELECT img_name FROM img_results WHERE img_name = ANY(%s)",
                (local_images,),
            )
            existing = {row["img_name"] for row in existing_rows} if existing_rows else set()

            new_images = [img for img in local_images if img not in existing]
            if new_images:
                query = "INSERT INTO img_results (img_name, result) VALUES (%s, %s)"
                for img in new_images:
                    try:
                        self.display.db.execute(query, (img, "OK"))
                    except Exception as exc:
                        self.logger.error(f"Error registering {img}: {exc}")

        finally:
            self.historic_bootstrap_loading = False
            self.historic_bootstrap_complete = True

    def _check_and_register_new_historic_images(self):
        """Check for new visible historic images and register them in DB."""
        if self.dataset_transfer_active:
            return
        visible_dir = self._get_visible_historic_dir()
        if not self.file_manager.exists(visible_dir):
            return

        try:
            current_mtime = self.file_manager.getmtime(visible_dir)
        except Exception:
            return

        if not hasattr(self, 'last_historic_mtime'):
            self.last_historic_mtime = current_mtime
            return

        if current_mtime > self.last_historic_mtime:
            self.last_historic_mtime = current_mtime
            # Run heavy DB/file work in background to avoid freezing the main loop
            if not getattr(self, '_register_worker_running', False):
                self._register_worker_running = True
                captured_dir = visible_dir

                def _register_worker():
                    worker_db = None
                    try:
                        if self.dataset_transfer_active:
                            return
                        from db import get_db_connection
                        worker_db = get_db_connection()
                        self._register_local_images_in_db(captured_dir, db_client=worker_db)
                        self._backfill_piece_result(db_client=worker_db)
                    except Exception as exc:
                        print(f"Error in register worker: {exc}")
                    finally:
                        if worker_db:
                            try:
                                worker_db.close()
                            except Exception:
                                pass
                        self._register_worker_running = False

                Thread(target=_register_worker, name="historic-register-worker", daemon=True).start()
        if self.historic_bootstrap_loading or self.historic_bootstrap_complete:
            return
        if not self.db_connected:
            return

        visible_dir = self._get_visible_historic_dir()
        self.historic_bootstrap_loading = True
        self.logger.info("[DB] Historic startup bootstrap started", allow_repeat=True)

        def _bootstrap_worker():
            worker_db = None
            completed = False
            try:
                if self.dataset_transfer_active:
                    return
                from db import get_db_connection

                worker_db = get_db_connection()
                self._register_local_images_in_db(
                    visible_dir,
                    db_client=worker_db,
                    track_registered=False,
                )
                self._backfill_piece_result(db_client=worker_db)
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

    def _set_sync_progress(self, stage, percent, title=None, helper_text=None):
        d = self.display
        d.sync_stage = str(stage)
        d.sync_progress = max(0, min(100, int(percent)))
        if title is not None:
            d.sync_progress_title = str(title)
        if helper_text is not None:
            d.sync_progress_helper_text = str(helper_text)

    def _set_reset_progress(self, stage, percent, title=None, helper_text=None):
        d = self.display
        d.reset_stage = str(stage)
        d.reset_progress = max(0, min(100, int(percent)))
        if title is not None:
            d.reset_progress_title = str(title)
        if helper_text is not None:
            d.reset_progress_helper_text = str(helper_text)

    def start_sync_images_by_status_async(self, historic_dir=None, base_dir=None):
        d = self.display
        if getattr(d, "sync_in_progress", False) or getattr(d, "reset_in_progress", False):
            return

        d.sync_in_progress = True
        d.sync_progress = 0
        d.sync_stage = "Preparing dataset sync..."
        d.sync_progress_title = "Saving Dataset"
        d.sync_progress_helper_text = "Please wait until the process finishes."
        d.sync_message = ""
        d.sync_message_is_error = False
        d.sync_message_time = 0

        def _sync_worker():
            worker_db = None
            try:
                self.stop_historic_download_worker()
                visible_images_snapshot = self._get_visible_historic_image_snapshot()

                from db import get_db_connection

                worker_db = get_db_connection()

                def _sync_progress_cb(done, total, stage):
                    if total <= 0:
                        phase_percent = 0
                        stage_text = stage
                    else:
                        phase_percent = int((done / total) * 65)
                        stage_text = f"{stage} ({done}/{total})"
                    self._set_sync_progress(
                        stage_text,
                        phase_percent,
                        title="Saving Dataset",
                        helper_text="Please wait until the process finishes.",
                    )

                sync_result = self.sync_images_by_status(
                    historic_dir=historic_dir,
                    base_dir=base_dir,
                    db_client=worker_db,
                    progress_callback=_sync_progress_cb,
                    visible_images_snapshot=visible_images_snapshot,
                )

                if not sync_result.get("ok", False):
                    raise RuntimeError(sync_result.get("error", "Dataset sync failed"))

                def _classification_progress_cb(done, total, stage):
                    if total <= 0:
                        phase_percent = 65
                        stage_text = stage
                    else:
                        phase_percent = 65 + int((done / total) * 23)
                        stage_text = f"{stage} ({done}/{total})"
                    self._set_sync_progress(
                        stage_text,
                        phase_percent,
                        title="Saving Dataset",
                        helper_text="Please wait until the process finishes.",
                    )

                classification = self.save_classification_results(
                    db_client=worker_db,
                    historic_dir=historic_dir,
                    progress_callback=_classification_progress_cb,
                    visible_images_snapshot=visible_images_snapshot,
                    export_stats_report=True,
                )

                def _verify_progress_cb(done, total, stage):
                    if total <= 0:
                        phase_percent = 88
                        stage_text = stage
                    else:
                        phase_percent = 88 + int((done / total) * 12)
                        stage_text = f"{stage} ({done}/{total})"
                    self._set_sync_progress(
                        stage_text,
                        phase_percent,
                        title="Saving Dataset",
                        helper_text="Please wait until the process finishes.",
                    )

                verify_result = self.verify_sync_images_by_status(
                    historic_dir=historic_dir,
                    base_dir=base_dir,
                    db_client=worker_db,
                    progress_callback=_verify_progress_cb,
                    rows_snapshot=sync_result.get("rows_snapshot"),
                    visible_images_snapshot=visible_images_snapshot,
                )

                self._set_sync_progress("Completed", 100)
                classification = classification or {}
                folder_errors = classification.get("classification_folder_errors", [])
                report_path = classification.get("stats_report_path")
                report_error = classification.get("stats_report_error")
                report_name = os.path.basename(str(report_path)) if report_path else ""
                if (
                    verify_result.get("verified")
                    and classification.get("ok")
                    and not folder_errors
                    and not report_error
                ):
                    images = classification.get("images", 0)
                    copied = classification.get("files_copied", 0)
                    if report_name:
                        d.sync_message = (
                            f"Dataset saved: {images} images, {copied} files copied. "
                            f"Report: {report_name}"
                        )
                    else:
                        d.sync_message = (
                            f"Dataset saved: {images} images, {copied} files copied"
                        )
                    d.sync_message_is_error = False
                elif (
                    verify_result.get("verified")
                    and classification.get("ok")
                    and not folder_errors
                    and report_error
                ):
                    images = classification.get("images", 0)
                    copied = classification.get("files_copied", 0)
                    d.sync_message = (
                        f"Dataset saved: {images} images, {copied} files copied. "
                        f"Report warning: {report_error}"
                    )
                    d.sync_message_is_error = True
                    self.logger.warn(
                        f"[SYNC] Stats report warning: {report_error}",
                        allow_repeat=True,
                    )
                elif verify_result.get("verified") and classification.get("ok") and folder_errors:
                    n_errors = len(folder_errors)
                    d.sync_message = (
                        f"Dataset saved but {n_errors} file copy issues"
                    )
                    d.sync_message_is_error = True
                    self.logger.warn(
                        f"[SYNC] Classification folder errors: {'; '.join(folder_errors[:5])}",
                        allow_repeat=True,
                    )
                elif not verify_result.get("verified"):
                    issue_count = verify_result.get("issue_count", 0)
                    d.sync_message = (
                        f"Dataset completed but verification failed ({issue_count} issues)"
                    )
                    d.sync_message_is_error = True
                    self.logger.warn(
                        f"[SYNC] Verification failed with {issue_count} issues",
                        allow_repeat=True,
                    )
                else:
                    cls_error = classification.get("error", "unknown error")
                    d.sync_message = f"Dataset saved but classification folder copy failed: {cls_error}"
                    d.sync_message_is_error = True
                    self.logger.warn(
                        f"[SYNC] Classification folder copy failed: {cls_error}",
                        allow_repeat=True,
                )
            except Exception as exc:
                d.sync_message = f"Dataset sync failed: {exc}"
                d.sync_message_is_error = True
                self.logger.error(f"[SYNC] Dataset sync failed: {exc}", allow_repeat=True)
            finally:
                try:
                    self.start_historic_download_on_startup(
                        self.config.temp_dir,
                        check_interval=self.config.historic_download_check_interval,
                    )
                except Exception as exc:
                    self.logger.error(
                        f"[SYNC] Error restarting historic download workers: {exc}",
                        allow_repeat=True,
                    )
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

    def _pause_dataset_background_workers(self):
        remote_db_was_running = bool(
            self.remote_db_poll_thread is not None
            and self.remote_db_poll_thread.is_alive()
        )
        self.stop_historic_download_worker()
        self.stop_remote_db_polling()
        return {"remote_db_was_running": remote_db_was_running}

    def _resume_dataset_background_workers(self, worker_state):
        try:
            self.start_historic_download_on_startup(
                self.config.temp_dir,
                check_interval=self.config.historic_download_check_interval,
            )
        except Exception as exc:
            self.logger.error(
                f"[DATASET_TRANSFER] Error restarting historic download workers: {exc}",
                allow_repeat=True,
            )

        if worker_state and worker_state.get("remote_db_was_running"):
            try:
                self.start_remote_db_polling()
            except Exception as exc:
                self.logger.error(
                    f"[DATASET_TRANSFER] Error restarting remote DB polling: {exc}",
                    allow_repeat=True,
                )

    def start_export_display_state_async(self):
        d = self.display
        if getattr(d, "sync_in_progress", False) or getattr(d, "reset_in_progress", False):
            return

        self.dataset_transfer_active = True
        d.sync_in_progress = True
        d.sync_progress = 0
        d.sync_stage = "Preparing export..."
        d.sync_progress_title = "Exporting Dataset"
        d.sync_progress_helper_text = "Creating a portable package from the current local state."
        d.sync_message = ""
        d.sync_message_is_error = False
        d.sync_message_time = 0

        def _export_worker():
            worker_db = None
            worker_state = None
            try:
                worker_state = self._pause_dataset_background_workers()

                from db import get_db_connection

                worker_db = get_db_connection()

                def _export_progress_cb(done, total, stage):
                    if total <= 0:
                        percent = 0
                        stage_text = stage
                    else:
                        percent = int((done / total) * 100)
                        stage_text = stage
                    self._set_sync_progress(
                        stage_text,
                        percent,
                        title="Exporting Dataset",
                        helper_text="Creating a portable package from the current local state.",
                    )

                result = export_display_state(
                    self,
                    db_client=worker_db,
                    progress_callback=_export_progress_cb,
                )
                if not result.get("ok", False):
                    raise RuntimeError(result.get("error", "Dataset export failed"))

                package_name = result.get("package_name") or os.path.basename(
                    str(result.get("package_path") or "")
                )
                d.sync_message = f"Export completed: {package_name}"
                d.sync_message_is_error = False
                self.logger.info(
                    f"[EXPORT] Dataset export completed: {result.get('package_path')}",
                    allow_repeat=True,
                )
            except Exception as exc:
                d.sync_message = f"Export failed: {exc}"
                d.sync_message_is_error = True
                self.logger.error(f"[EXPORT] Dataset export failed: {exc}", allow_repeat=True)
            finally:
                if worker_state is not None:
                    self._resume_dataset_background_workers(worker_state)
                self.dataset_transfer_active = False
                d.sync_in_progress = False
                d.sync_message_time = time.time()
                if worker_db is not None:
                    try:
                        worker_db.close()
                    except Exception:
                        pass

        self.export_worker_thread = Thread(
            target=_export_worker,
            name="dataset-export-worker",
            daemon=True,
        )
        self.export_worker_thread.start()

    def start_import_display_state_async(self, package_path):
        d = self.display
        if getattr(d, "sync_in_progress", False) or getattr(d, "reset_in_progress", False):
            return
        if not package_path:
            return

        self.dataset_transfer_active = True
        d.sync_in_progress = True
        d.sync_progress = 0
        d.sync_stage = "Preparing import..."
        d.sync_progress_title = "Importing Dataset"
        d.sync_progress_helper_text = "Merging package contents into the local display state."
        d.sync_message = ""
        d.sync_message_is_error = False
        d.sync_message_time = 0

        def _import_worker():
            worker_db = None
            worker_state = None
            try:
                worker_state = self._pause_dataset_background_workers()

                from db import get_db_connection

                worker_db = get_db_connection()

                def _import_progress_cb(done, total, stage):
                    if total <= 0:
                        percent = 0
                        stage_text = stage
                    else:
                        percent = int((done / total) * 100)
                        stage_text = stage
                    self._set_sync_progress(
                        stage_text,
                        percent,
                        title="Importing Dataset",
                        helper_text="Merging package contents into the local display state.",
                    )

                result = import_display_state(
                    self,
                    package_path=package_path,
                    db_client=worker_db,
                    progress_callback=_import_progress_cb,
                )
                if not result.get("ok", False):
                    raise RuntimeError(result.get("error", "Dataset import failed"))

                annotated = result.get("annotated", {})
                historic = result.get("historic", {})
                db_stats = result.get("db", {})
                file_imported = int(annotated.get("copied", 0)) + int(historic.get("copied", 0))
                file_skipped = int(annotated.get("skipped", 0)) + int(historic.get("skipped", 0))
                db_inserted_total = sum((db_stats.get("inserted") or {}).values())
                db_skipped_total = sum((db_stats.get("skipped") or {}).values())
                d.sync_message = (
                    f"Import completed: {file_imported} files, {db_inserted_total} DB rows added, "
                    f"{file_skipped + db_skipped_total} duplicates skipped"
                )
                d.sync_message_is_error = False
                self.logger.info(
                    f"[IMPORT] Dataset import completed from {package_path}",
                    allow_repeat=True,
                )
            except Exception as exc:
                d.sync_message = f"Import failed: {exc}"
                d.sync_message_is_error = True
                self.logger.error(f"[IMPORT] Dataset import failed: {exc}", allow_repeat=True)
            finally:
                if worker_state is not None:
                    self._resume_dataset_background_workers(worker_state)
                self.dataset_transfer_active = False
                d.sync_in_progress = False
                d.sync_message_time = time.time()
                if worker_db is not None:
                    try:
                        worker_db.close()
                    except Exception:
                        pass

        self.import_worker_thread = Thread(
            target=_import_worker,
            name="dataset-import-worker",
            daemon=True,
        )
        self.import_worker_thread.start()

    def start_reset_async(self):
        d = self.display
        if getattr(d, "reset_in_progress", False) or getattr(d, "sync_in_progress", False):
            return

        d.reset_in_progress = True
        d.reset_progress = 0
        d.reset_stage = "Preparing reset..."
        d.reset_progress_title = "Resetting Dataset"
        d.reset_progress_helper_text = "Clearing historic, annotated, classified, and final folders."
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
                    self._set_reset_progress(
                        stage_text,
                        phase_percent,
                        title="Resetting Dataset",
                        helper_text="Clearing historic, annotated, classified, and final folders.",
                    )

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

    def start_rebuild_db_from_historic_async(self):
        d = self.display
        if getattr(d, "reset_in_progress", False) or getattr(d, "sync_in_progress", False):
            return

        d.reset_in_progress = True
        d.reset_progress = 0
        d.reset_stage = "Preparing rebuild..."
        d.reset_progress_title = "Rebuilding Database"
        d.reset_progress_helper_text = "Please wait until rebuild finishes."
        d.sync_message = ""
        d.sync_message_is_error = False
        d.sync_message_time = 0

        def _rebuild_worker():
            worker_db = None
            try:
                from db import get_db_connection

                worker_db = get_db_connection()

                def _rebuild_progress_cb(done, total, stage):
                    if total <= 0:
                        phase_percent = 0
                        stage_text = stage
                    else:
                        phase_percent = int((done / total) * 100)
                        stage_text = f"{stage} ({done}/{total})"
                    self._set_reset_progress(
                        stage_text,
                        phase_percent,
                        title="Rebuilding Database",
                        helper_text="Please wait until rebuild finishes.",
                    )

                result = self.perform_rebuild_db_from_historic(
                    db_client=worker_db,
                    progress_callback=_rebuild_progress_cb,
                )

                if result.get("ok", False):
                    d.sync_message = "Database rebuilt successfully"
                    d.sync_message_is_error = False
                else:
                    error_text = result.get("error", "Database rebuild failed")
                    d.sync_message = f"Database rebuild completed with issues: {error_text}"
                    d.sync_message_is_error = True
            except Exception as exc:
                d.sync_message = f"Database rebuild failed: {exc}"
                d.sync_message_is_error = True
                self.logger.error(f"[REBUILD] Database rebuild failed: {exc}", allow_repeat=True)
            finally:
                d.reset_in_progress = False
                d.sync_message_time = time.time()
                if worker_db is not None:
                    try:
                        worker_db.close()
                    except Exception:
                        pass

        self.rebuild_worker_thread = Thread(
            target=_rebuild_worker,
            name="dataset-rebuild-worker",
            daemon=True,
        )
        self.rebuild_worker_thread.start()

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
            _clear_capture_modal(self.display)
            return
        if not self.sftp_connected and not self.try_connect("remote-start"):
            self.logger.warn("[REMOTE] Cannot start remote process while disconnected", allow_repeat=True)
            self.display.remote_requested = False
            _clear_capture_modal(self.display)
            return

        self.logger.info("[REMOTE] Start requested", allow_repeat=True)
        self.stop_event = Event()
        self.pid_queue = Queue()
        self.event_queue = Queue()
        self.stdin_queue = Queue()
        self.remote_pid = None
        self.remote_process = self.sftp_app.start_remote_process_multiprocess(
            self.config.remote_command,
            pid_queue=self.pid_queue,
            stop_event=self.stop_event,
            status_queue=self.event_queue,
            stdin_queue=self.stdin_queue,
        )
        self.display.remote_requested = True
        self.display.trigger_active = False
        self.display.manual_capture_pending = False
        _clear_capture_modal(self.display)
        try:
            self.remote_pid = self.pid_queue.get(timeout=5)
            self.logger.info(f"[REMOTE] PID: {self.remote_pid}", allow_repeat=True)
        except Exception:
            self.remote_pid = None

    def stop_remote_process(self, reason="user"):
        if self.stop_event is None and self.remote_process is None and self.remote_pid is None:
            self.display.remote_requested = False
            self.display.trigger_active = False
            self.display.manual_capture_pending = False
            _clear_capture_modal(self.display)
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
        self.stdin_queue = None
        self.display.remote_requested = False
        self.display.trigger_active = False
        self.display.manual_capture_pending = False
        _clear_capture_modal(self.display)

    def _process_remote_events(self):
        if self.event_queue is not None:
            try:
                while True:
                    msg = self.event_queue.get_nowait()
                    _process_remote_event_impl(
                        msg=msg,
                        display=self.display,
                        logger=self.logger,
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
            rotation_state=self.live_rotation_state_remote,
            logger=self.logger,
            image_extensions=self.config.image_extensions,
            max_images=self.config.max_images,
        )

    def _load_historic_index(self, force_rescan=False):
        d = self.display

        visible_dir = self._get_visible_historic_dir()
        if not self.file_manager.exists(visible_dir):
            d._historic_index_cache = []
            d._historic_jsn_cache = []
            d._historic_index_mtime = None
            d._historic_index_last_scan = time.monotonic()
            return []

        current_mtime = None
        try:
            current_mtime = self.file_manager.getmtime(visible_dir)
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

        files = self.file_manager.listdir(visible_dir)
        images_with_jsn = [
            name
            for name in files
            if name.lower().endswith(self.config.image_extensions) and name.startswith("11861")
        ]

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

    def _refresh_historic_index_async(self):
        """Rescan historic directory in a background thread and update d.historic_images."""
        if getattr(self, '_historic_index_refresh_running', False):
            return
        self._historic_index_refresh_running = True

        def _worker():
            try:
                new_index = self._load_historic_index(force_rescan=True)
                d = self.display
                if not new_index:
                    self._historic_index_refresh_running = False
                    return
                # Preserve current JSN position
                current_jsn = None
                if d.historic_images:
                    try:
                        batch = d.historic_images[d.historic_offset]
                        if batch:
                            current_jsn = batch[0].split("_")[0] if "_" in batch[0] else batch[0]
                    except Exception:
                        pass
                d.historic_images = new_index
                if current_jsn:
                    for idx, batch in enumerate(new_index):
                        if batch and (batch[0].split("_")[0] if "_" in batch[0] else batch[0]) == current_jsn:
                            d.historic_offset = idx
                            break
                    else:
                        d.historic_offset = min(d.historic_offset, len(new_index) - 1)
                else:
                    d.historic_offset = min(d.historic_offset, len(new_index) - 1)
            except Exception as exc:
                print(f"Error refreshing historic index: {exc}")
            finally:
                self._historic_index_refresh_running = False

        Thread(target=_worker, name="historic-index-refresh", daemon=True).start()

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
            # Use cached index immediately (never blocks); trigger async rescan if stale
            cached = self._load_historic_index(force_rescan=False)

            if not cached:
                if not d.historic_mode:
                    self._show_no_images_dialog("No images available")
                # Kick off a background rescan so next call has fresh data
                self._refresh_historic_index_async()
                return

            d.historic_images = cached

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

            # Trigger async rescan so the index stays fresh without blocking
            self._refresh_historic_index_async()
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
        d.show_rebuild_confirm = False
        d.show_piece_date_dialog = False
        d.show_piece_number_dialog = False
        d.piece_number_dialog_input = ""
        d.piece_number_dialog_replace_on_input = False
        if hasattr(d, "historic_jsn_rect"):
            d.historic_jsn_rect = None
        if hasattr(d, "toast_message"):
            d.toast_message = ""

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
        total_batches = len(d.historic_images)
        d.historic_offset = (d.historic_offset - 1) % total_batches

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

    def _sanitize_search_jsn(self, value, max_length=21):
        """Normalize search input so keyboard and paste use the same JSN rules."""
        if value is None:
            return ""
        return "".join(ch for ch in str(value) if ch.isdigit())[:max_length]

    def _find_historic_batch_index(self, historic_images, target_jsn):
        """Return the historic batch index for a JSN, or None when absent."""
        for idx, batch in enumerate(historic_images or []):
            if not batch:
                continue
            batch_jsn = batch[0].split("_")[0] if "_" in batch[0] else batch[0]
            if batch_jsn == target_jsn:
                return idx
        return None

    def _get_current_historic_piece_number(self):
        """Return the current visible historic piece number using UI numbering."""
        d = self.display
        total_pieces = len(d.historic_images or [])
        if total_pieces <= 0:
            return 0

        current_piece = total_pieces - int(d.historic_offset or 0)
        if current_piece < 1:
            return 1
        if current_piece > total_pieces:
            return total_pieces
        return current_piece

    def _close_piece_number_dialog(self, clear_input=False):
        """Hide the go-to-piece dialog and optionally clear its input."""
        d = self.display
        d.show_piece_number_dialog = False
        d.piece_number_dialog_replace_on_input = False
        if clear_input:
            d.piece_number_dialog_input = ""

    def _open_piece_number_dialog(self):
        """Open the go-to-piece dialog prefilled with the current piece number."""
        d = self.display
        total_pieces = len(d.historic_images or [])
        if total_pieces <= 0:
            self._show_no_images_dialog("No images available")
            return False

        d.show_piece_date_dialog = False
        d.show_piece_number_dialog = True
        d.show_reset_confirm = False
        d.show_delete_confirm = False
        d.show_rebuild_confirm = False
        d.search_active = False
        d.filtered_suggestions = []
        d.selected_suggestion_idx = -1
        d.piece_number_dialog_input = str(self._get_current_historic_piece_number())
        d.piece_number_dialog_replace_on_input = True
        return True

    def go_to_historic_jsn(self, jsn, show_missing_dialog=False, close_stats_modal=False):
        """Navigate to a JSN in historic mode using the freshest local index available."""
        d = self.display
        target_jsn = self._sanitize_search_jsn(jsn)
        if not target_jsn:
            return False

        if close_stats_modal:
            d.show_stats_class_modal = False
            if hasattr(d, "_reset_stats_class_modal_state"):
                d._reset_stats_class_modal_state()

        historic_index = self._load_historic_index(force_rescan=False) or []
        target_idx = self._find_historic_batch_index(historic_index, target_jsn)

        if target_idx is None:
            historic_index = self._load_historic_index(force_rescan=True) or []
            target_idx = self._find_historic_batch_index(historic_index, target_jsn)

        if target_idx is None:
            if show_missing_dialog:
                self._show_no_images_dialog(f"JSN {target_jsn} not in historic folder")
            return False

        d.historic_images = historic_index
        d.historic_mode = True
        d.historic_offset = target_idx
        d.search_active = False
        d.filtered_suggestions = []
        d.selected_suggestion_idx = -1
        self._refresh_historic_index_async()
        return True

    def go_to_historic_piece_number(self, piece_number, show_missing_dialog=False):
        """Navigate to a historic piece number using the displayed numbering."""
        d = self.display
        try:
            target_piece = int(str(piece_number).strip())
        except (TypeError, ValueError):
            if show_missing_dialog:
                self._show_no_images_dialog(f"Piece {piece_number} not available")
            return False

        historic_index = self._load_historic_index(force_rescan=False) or []
        if not historic_index:
            historic_index = self._load_historic_index(force_rescan=True) or []

        total_pieces = len(historic_index)
        if target_piece < 1 or target_piece > total_pieces:
            if show_missing_dialog:
                self._show_no_images_dialog(f"Piece {target_piece} not available")
            return False

        d.historic_images = historic_index
        d.historic_mode = True
        d.historic_offset = total_pieces - target_piece
        d.search_active = False
        d.filtered_suggestions = []
        d.selected_suggestion_idx = -1
        self._refresh_historic_index_async()
        return True

    def perform_jsn_search(self):
        d = self.display
        if not d.search_jsn.strip():
            print("No JSN entered for search")
            return

        search_term = d.search_jsn.strip()
        if self.go_to_historic_jsn(search_term, show_missing_dialog=False, close_stats_modal=False):
            print(f"JSN {search_term} found at position {d.historic_offset}")
        else:
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

        local_sources = [
            ("annotated", self._get_visible_historic_dir()),
            ("historic", self._get_export_historic_dir()),
        ]
        local_deleted = 0
        local_candidates = []
        for label, local_dir in local_sources:
            folder_candidates = []
            if self.file_manager.exists(local_dir):
                try:
                    for name in self.file_manager.listdir(local_dir):
                        if name.startswith(jsn) and name.lower().endswith(self.config.image_extensions):
                            folder_candidates.append(self.file_manager.join(local_dir, name))
                    for path in folder_candidates:
                        try:
                            self.file_manager.remove(path)
                            local_deleted += 1
                        except Exception as exc:
                            print(f"Error deleting local {label} file {path}: {exc}")
                    print(f"Local {label} delete: {len(folder_candidates)} candidates")
                    local_candidates.extend(folder_candidates)
                except Exception as exc:
                    print(f"Error reading local {label} folder: {exc}")
            else:
                print(f"Local {label} folder does not exist")

        remote_deleted = 0
        remote_sources = [
            ("annotated", self.config.remote_annotated_dir),
            ("historic", self.config.remote_hist_dir),
        ]
        if d.sftp_client:
            for label, remote_dir in remote_sources:
                try:
                    self.file_manager.sftp_chdir(d.sftp_client, remote_dir)
                    remote_files = self.file_manager.sftp_listdir(d.sftp_client)
                    remote_candidates = [
                        f
                        for f in remote_files
                        if f.startswith(jsn) and f.lower().endswith(self.config.image_extensions)
                    ]
                    for remote_file in remote_candidates:
                        try:
                            file_path = f"{remote_dir}/{remote_file}"
                            self.file_manager.sftp_remove(d.sftp_client, file_path)
                            remote_deleted += 1
                        except Exception as exc:
                            print(f"Error deleting remote {label} file {remote_file}: {exc}")
                    print(f"Remote {label} delete: {len(remote_candidates)} candidates")
                except Exception as exc:
                    print(f"Error accessing remote {label} folder: {exc}")
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
        visible_dir = self._get_visible_historic_dir()
        if self.file_manager.exists(visible_dir):
            remaining_images = self._list_local_image_names(
                visible_dir,
                require_jsn_prefix=True,
            )

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

    def _invalidate_dataset_runtime_state(self, clear_historic_images=False):
        d = self.display
        if clear_historic_images:
            d.historic_images = []
            d.historic_offset = 0
        d.temp_results = {}
        d.available_jsns = []
        d.filtered_suggestions = []
        d.historic_db_registered = False
        d._db_registered_images.clear()
        d._historic_index_cache = None
        d._historic_index_mtime = None
        d._historic_index_last_scan = 0.0
        d._historic_jsn_cache = []
        d._db_result_cache.clear()
        d._image_cache.clear()
        if hasattr(d, "historic_jsn_rect"):
            d.historic_jsn_rect = None
        if hasattr(d, "toast_message"):
            d.toast_message = ""

    def _clear_final_classification_dir(self):
        base_dir = str(FINAL_CLASSIFICATION_DIR)
        self.file_manager.makedirs(base_dir, exist_ok=True)

        expected_folders = {
            folder_name
            for position_dirs in FINAL_CLASSIFICATION_DIRS.values()
            for folder_name in position_dirs.values()
        }

        removed_entries = 0
        for entry_name in list(self.file_manager.listdir(base_dir)):
            entry_path = self.file_manager.join(base_dir, entry_name)
            if self.file_manager.is_dir(entry_path):
                if entry_name not in expected_folders:
                    self.file_manager.rmtree(entry_path)
                    removed_entries += 1
                    continue
                for child_name in list(self.file_manager.listdir(entry_path)):
                    child_path = self.file_manager.join(entry_path, child_name)
                    if self.file_manager.is_dir(child_path):
                        self.file_manager.rmtree(child_path)
                    else:
                        self.file_manager.remove(child_path)
                    removed_entries += 1
            else:
                self.file_manager.remove(entry_path)
                removed_entries += 1

        for position_dirs in FINAL_CLASSIFICATION_DIRS.values():
            for folder_name in position_dirs.values():
                self.file_manager.makedirs(
                    self.file_manager.join(base_dir, folder_name),
                    exist_ok=True,
                )

        return removed_entries

    def _clear_sync_images_base_dir(self):
        base_dir = str(SYNC_IMAGES_BASE_DIR)
        self.file_manager.makedirs(base_dir, exist_ok=True)

        expected_folders = {
            folder_name
            for position_dirs in STATUS_SYNC_DIRS.values()
            for folder_name in position_dirs.values()
        }

        removed_entries = 0
        for entry_name in list(self.file_manager.listdir(base_dir)):
            entry_path = self.file_manager.join(base_dir, entry_name)
            if self.file_manager.is_dir(entry_path):
                if entry_name not in expected_folders:
                    self.file_manager.rmtree(entry_path)
                    removed_entries += 1
                    continue

                for child_name in list(self.file_manager.listdir(entry_path)):
                    child_path = self.file_manager.join(entry_path, child_name)
                    if self.file_manager.is_dir(child_path):
                        self.file_manager.rmtree(child_path)
                    else:
                        self.file_manager.remove(child_path)
                    removed_entries += 1
            else:
                self.file_manager.remove(entry_path)
                removed_entries += 1

        for folder_name in expected_folders:
            self.file_manager.makedirs(
                self.file_manager.join(base_dir, folder_name),
                exist_ok=True,
            )

        return removed_entries

    def perform_rebuild_db_from_historic(self, db_client=None, progress_callback=None):
        d = self.display
        db = db_client or d.db
        print("\n" + "=" * 70)
        print("STARTING DATABASE REBUILD FROM ANNOTATED HISTORIC SOURCE")
        print("=" * 70)

        visible_dir = self._get_visible_historic_dir()
        errors = []
        total_steps = 4
        completed_steps = 0

        def _advance(stage):
            nonlocal completed_steps
            completed_steps += 1
            if callable(progress_callback):
                progress_callback(completed_steps, total_steps, stage)

        if callable(progress_callback):
            progress_callback(0, total_steps, "Preparing rebuild")

        if not db:
            message = "No database connection available"
            print(message)
            return {"ok": False, "error": message}

        if not self.file_manager.exists(visible_dir):
            try:
                self.file_manager.makedirs(visible_dir, exist_ok=True)
                print(f"Annotated directory not found; created empty folder: {visible_dir}")
            except Exception as exc:
                message = f"Unable to create annotated directory: {exc}"
                print(message)
                return {"ok": False, "error": message}

        try:
            historic_images = sorted(
                [
                    name
                    for name in self.file_manager.listdir(visible_dir)
                    if name.lower().endswith(self.config.image_extensions)
                ]
            )
        except Exception as exc:
            message = f"Unable to scan annotated directory: {exc}"
            print(message)
            return {"ok": False, "error": message}

        _advance("Scanning annotated images")

        try:
            truncated_tables = db.truncate_app_tables()
            print(f"Truncated {truncated_tables} app tables")
        except Exception as exc:
            message = f"Error clearing database tables: {exc}"
            print(message)
            return {"ok": False, "error": message}
        _advance("Clearing database tables")

        try:
            self._register_local_images_in_db(
                visible_dir,
                image_names=historic_images,
                db_client=db,
                track_registered=False,
            )
            self._backfill_piece_result(db_client=db)
            count_rows = db.fetch("SELECT COUNT(*) AS cnt FROM img_results")
            inserted_count = int(count_rows[0]["cnt"]) if count_rows else 0
            print(f"Rebuilt {inserted_count}/{len(historic_images)} img_results rows from annotated")
            if inserted_count != len(historic_images):
                errors.append(
                    f"Expected {len(historic_images)} img_results rows after rebuild, found {inserted_count}"
                )
            if not historic_images:
                print("Annotated directory is empty; database remains empty after rebuild")
        except Exception as exc:
            message = f"Error rebuilding database from annotated: {exc}"
            print(message)
            return {"ok": False, "error": message}
        _advance("Rebuilding database from annotated")

        self._invalidate_dataset_runtime_state(clear_historic_images=False)
        if historic_images:
            self.enter_historic_mode()
        else:
            self.exit_historic_mode()
        _advance("Refreshing historic view")

        print("=" * 70)
        if errors:
            print("DATABASE REBUILD COMPLETED WITH ISSUES")
        else:
            print("DATABASE REBUILD COMPLETED SUCCESSFULLY")
        print("=" * 70 + "\n")

        if callable(progress_callback):
            progress_callback(total_steps, total_steps, "Completed")
        if errors:
            return {"ok": False, "error": errors[0], "errors": errors}
        return {"ok": True}

    def perform_reset(self, db_client=None, progress_callback=None):
        d = self.display
        db = db_client or d.db
        print("\n" + "=" * 70)
        print("STARTING COMPLETE RESET")
        print("=" * 70)
        errors = []
        result = None

        local_targets = [
            ("historic", self._get_export_historic_dir()),
            ("annotated", self._get_visible_historic_dir()),
        ]
        remote_targets = [
            ("historic", self.config.remote_hist_dir),
            ("annotated", self.config.remote_annotated_dir),
        ]

        self.stop_historic_download_worker()

        try:
            def _scan_local_entries(folder_label, folder_path):
                if not self.file_manager.exists(folder_path):
                    return []
                try:
                    return list(self.file_manager.listdir(folder_path))
                except Exception as exc:
                    errors.append(
                        f"Unable to scan local {folder_label} folder: {exc}"
                    )
                    print(f"Error scanning local {folder_label} folder: {exc}")
                    return []

            def _scan_remote_entries(folder_label, remote_dir):
                if not d.sftp_client:
                    return []
                try:
                    self.file_manager.sftp_chdir(d.sftp_client, remote_dir)
                    return list(self.file_manager.sftp_listdir(d.sftp_client))
                except FileNotFoundError:
                    return []
                except Exception as exc:
                    errors.append(
                        f"Unable to access remote {folder_label} folder: {exc}"
                    )
                    print(f"Error accessing remote {folder_label} folder: {exc}")
                    return []

            local_entries_by_label = {
                folder_label: _scan_local_entries(folder_label, folder_path)
                for folder_label, folder_path in local_targets
            }
            remote_entries_by_label = {
                folder_label: _scan_remote_entries(folder_label, remote_dir)
                for folder_label, remote_dir in remote_targets
            }

            local_steps = sum(
                max(1, len(local_entries_by_label[folder_label]))
                for folder_label, _ in local_targets
            )
            remote_steps = (
                sum(
                    max(1, len(remote_entries_by_label[folder_label]))
                    for folder_label, _ in remote_targets
                )
                if d.sftp_client
                else len(remote_targets)
            )
            db_steps = 1
            classified_steps = 1
            final_classification_steps = 1
            final_steps = 1
            total_steps = (
                local_steps
                + remote_steps
                + db_steps
                + classified_steps
                + final_classification_steps
                + final_steps
            )
            completed_steps = 0

            def _advance(stage):
                nonlocal completed_steps
                completed_steps += 1
                if callable(progress_callback):
                    progress_callback(completed_steps, total_steps, stage)

            def _clear_local_folder(folder_label, folder_path, entry_names):
                if self.file_manager.exists(folder_path):
                    if entry_names:
                        for idx, entry_name in enumerate(entry_names, start=1):
                            entry_path = self.file_manager.join(folder_path, entry_name)
                            try:
                                if self.file_manager.is_dir(entry_path):
                                    self.file_manager.rmtree(entry_path)
                                else:
                                    self.file_manager.remove(entry_path)
                            except Exception as exc:
                                errors.append(
                                    f"Error removing local {folder_label} entry "
                                    f"'{entry_name}': {exc}"
                                )
                                print(
                                    f"Error removing local {folder_label} entry "
                                    f"{entry_name}: {exc}"
                                )
                            _advance(
                                f"Clearing local {folder_label} folder "
                                f"({idx}/{len(entry_names)})"
                            )
                        print(f"Local {folder_label} folder cleared")
                    else:
                        print(f"Local {folder_label} folder is already empty")
                        _advance(f"Local {folder_label} folder is already empty")
                else:
                    print(f"Local {folder_label} folder did not exist")
                    _advance(f"Preparing local {folder_label} folder")

                try:
                    self.file_manager.makedirs(folder_path, exist_ok=True)
                except Exception as exc:
                    errors.append(
                        f"Error recreating local {folder_label} folder: {exc}"
                    )
                    print(f"Error recreating local {folder_label} folder: {exc}")

            def _clear_remote_folder(folder_label, remote_dir, entry_names):
                if entry_names:
                    print(
                        f"Deleting {len(entry_names)} files from remote "
                        f"{folder_label} folder..."
                    )
                    deleted_count = 0
                    for idx, remote_file in enumerate(entry_names, start=1):
                        try:
                            file_path = f"{remote_dir}/{remote_file}"
                            self.file_manager.sftp_remove(d.sftp_client, file_path)
                            deleted_count += 1
                        except Exception as exc:
                            errors.append(
                                f"Error deleting remote {folder_label} file "
                                f"'{remote_file}': {exc}"
                            )
                            print(
                                f"Error deleting remote {folder_label} file "
                                f"{remote_file}: {exc}"
                            )
                        _advance(
                            f"Clearing remote {folder_label} folder "
                            f"({idx}/{len(entry_names)})"
                        )
                    print(
                        f"Deleted {deleted_count}/{len(entry_names)} files from "
                        f"remote {folder_label} folder"
                    )
                else:
                    print(f"Remote {folder_label} folder is already empty")
                    _advance(f"Remote {folder_label} folder is already empty")

            if callable(progress_callback):
                progress_callback(0, total_steps, "Preparing reset")

            for folder_label, folder_path in local_targets:
                _clear_local_folder(
                    folder_label,
                    folder_path,
                    local_entries_by_label[folder_label],
                )

            if d.sftp_client:
                for folder_label, remote_dir in remote_targets:
                    _clear_remote_folder(
                        folder_label,
                        remote_dir,
                        remote_entries_by_label[folder_label],
                    )
            else:
                print("No SFTP connection available")
                for folder_label, _remote_dir in remote_targets:
                    _advance(f"Remote {folder_label} reset skipped (no SFTP connection)")

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

            try:
                removed_entries = self._clear_sync_images_base_dir()
                print(f"Cleared {removed_entries} entries from sync_images base directory")
            except Exception as exc:
                errors.append(f"Error clearing sync_images base directory: {exc}")
                print(f"Error clearing sync_images base directory: {exc}")
            _advance("Clearing classified folders")

            try:
                removed_entries = self._clear_final_classification_dir()
                print(f"Cleared {removed_entries} entries from final_classification")
            except Exception as exc:
                errors.append(f"Error clearing final_classification: {exc}")
                print(f"Error clearing final_classification: {exc}")
            _advance("Clearing final classification")

            self._invalidate_dataset_runtime_state(clear_historic_images=True)
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
                result = {"ok": False, "error": errors[0], "errors": errors}
            else:
                result = {"ok": True}
        finally:
            try:
                self.start_historic_download_on_startup(
                    self.config.temp_dir,
                    check_interval=self.config.historic_download_check_interval,
                )
            except Exception as exc:
                errors.append(f"Error restarting historic download workers: {exc}")
                print(f"Error restarting historic download workers: {exc}")
                if result is None or result.get("ok"):
                    result = {"ok": False, "error": errors[0], "errors": errors}

        return result

    def start_historic_download_on_startup(self, local_path, check_interval=30):
        d = self.display
        historic_temp_dir = self.file_manager.join(local_path, HISTORIC_SUBDIR_NAME)
        annotated_temp_dir = self.file_manager.join(local_path, ANNOTATED_SUBDIR_NAME)
        self.file_manager.makedirs(historic_temp_dir, exist_ok=True)
        self.file_manager.makedirs(annotated_temp_dir, exist_ok=True)

        creds = self.sftp_credentials or d.sftp_credentials
        if not creds:
            print("SFTP background downloader disabled: missing credentials")
            return

        hostname = creds.get("hostname")
        port = creds.get("port")
        username = creds.get("username")
        password = creds.get("password")
        if not all([hostname, port, username, password]):
            print("SFTP background downloader disabled: incomplete credentials")
            return

        def _is_alive(process):
            try:
                return process is not None and process.is_alive()
            except Exception:
                return False

        def _start_worker(remote_dir, local_dir, process_attr, stop_attr, worker_label):
            process = getattr(d, process_attr, None)
            if _is_alive(process):
                print(f"Background download already running for {worker_label}")
                return

            try:
                stop_event = Event()
                process = Process(
                    target=_download_images_background_worker,
                    args=(
                        hostname,
                        port,
                        username,
                        password,
                        remote_dir,
                        local_dir,
                        check_interval,
                        10,
                        stop_event,
                        worker_label,
                    ),
                )
                process.daemon = True
                process.start()
                setattr(d, stop_attr, stop_event)
                setattr(d, process_attr, process)
            except Exception as exc:
                print(f"Error starting background download for {worker_label}: {exc}")
                setattr(d, process_attr, None)
                setattr(d, stop_attr, None)

        _start_worker(
            remote_dir=self.config.remote_hist_dir,
            local_dir=historic_temp_dir,
            process_attr="download_process",
            stop_attr="download_stop_event",
            worker_label="HIST_SYNC_SSH",
        )
        _start_worker(
            remote_dir=self.config.remote_annotated_dir,
            local_dir=annotated_temp_dir,
            process_attr="annotated_download_process",
            stop_attr="annotated_download_stop_event",
            worker_label="ANNOTATED_SYNC_SSH",
        )

    def stop_historic_download_worker(self):
        d = self.display
        def _stop_worker(process_attr, stop_attr):
            stop_event = getattr(d, stop_attr, None)
            if stop_event is not None:
                try:
                    stop_event.set()
                except Exception:
                    pass

            process = getattr(d, process_attr, None)
            if process is not None:
                try:
                    process.join(timeout=2)
                except Exception:
                    pass

                try:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=1)
                except Exception:
                    pass

            setattr(d, process_attr, None)
            setattr(d, stop_attr, None)

        _stop_worker("download_process", "download_stop_event")
        _stop_worker("annotated_download_process", "annotated_download_stop_event")

    def download_historic_batch(self, local_path, max_images=7):
        d = self.display
        if not d.historic_images:
            return []

        try:
            annotated_temp_dir = self.file_manager.join(local_path, ANNOTATED_SUBDIR_NAME)
            batch_images = d.historic_images[d.historic_offset]

            downloaded_files = []
            for img in batch_images:
                local_file = self.file_manager.join(annotated_temp_dir, img)
                if self.file_manager.exists(local_file):
                    downloaded_files.append(local_file)

            # Only register when the batch changes, not every loop iteration
            batch_key = (d.historic_offset, tuple(batch_images))
            if getattr(self, '_last_registered_batch_key', None) != batch_key:
                self._register_local_images_in_db(annotated_temp_dir, image_names=batch_images)
                self._last_registered_batch_key = batch_key

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
                        self._upsert_classification(img_name, "OK", db_client=db)
                    except Exception as exc:
                        print(f"Error inserting {img_name}: {exc}")

            if track_registered:
                d._db_registered_images.update(pending)
            d.historic_db_registered = True

        except Exception as exc:
            print(f"General error registering images in DB: {exc}")

    def _backfill_piece_result(self, db_client=None):
        """Populate piece_result for any img_results rows whose JSN is not yet in piece_result."""
        db = db_client or self.display.db
        if not db:
            return
        try:
            rows = db.fetch(
                """
                SELECT img_name, result FROM img_results
                WHERE SPLIT_PART(img_name, '_', 1) NOT IN (SELECT jsn FROM piece_result)
                """
            )
            if not rows:
                return
            self.logger.info(f"[DB] Backfilling piece_result for {len(rows)} images", allow_repeat=True)
            for row in rows:
                img_name = row.get("img_name")
                operator_result = str(row.get("result") or "OK").strip().upper()
                if operator_result not in ("OK", "NOK"):
                    operator_result = "OK"
                if img_name:
                    self._upsert_classification(img_name, operator_result, db_client=db)
        except Exception as exc:
            print(f"Error backfilling piece_result: {exc}")

    def _upsert_classification(self, img_name, operator_result, db_client=None):
        db = db_client or self.display.db
        if not db:
            return

        m = re.search(r"_(OK|NOK)\.\w+$", img_name, re.IGNORECASE)
        model_result = m.group(1).upper() if m else "OK"

        jsn = img_name.split("_")[0] if "_" in img_name else img_name

        try:
            with db.get_cursor() as cursor:
                cursor.execute(
                    "INSERT INTO piece_result (jsn, operator_result, model_result) "
                    "VALUES (%s, %s, %s) "
                    "ON CONFLICT (jsn) DO UPDATE SET "
                    "operator_result = CASE WHEN piece_result.operator_result = 'NOK' "
                    "  OR EXCLUDED.operator_result = 'NOK' THEN 'NOK' ELSE 'OK' END, "
                    "model_result = CASE WHEN piece_result.model_result = 'NOK' "
                    "  OR EXCLUDED.model_result = 'NOK' THEN 'NOK' ELSE 'OK' END "
                    "RETURNING id",
                    (jsn, operator_result, model_result),
                )
                piece_id = cursor.fetchone()["id"]

                cursor.execute(
                    "INSERT INTO classified_images "
                    "(img_name, operator_result, model_result, piece_id) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (img_name) DO UPDATE SET "
                    "operator_result = EXCLUDED.operator_result, "
                    "model_result = EXCLUDED.model_result, "
                    "piece_id = EXCLUDED.piece_id",
                    (img_name, operator_result, model_result, piece_id),
                )
        except Exception as exc:
            print(f"Error upserting classification for {img_name}: {exc}")

    def _recalculate_piece_result(self, jsn, db_client=None):
        db = db_client or self.display.db
        if not db:
            return
        try:
            db.execute(
                "UPDATE piece_result SET "
                "operator_result = COALESCE("
                "  (SELECT 'NOK' FROM classified_images "
                "   WHERE piece_id = piece_result.id AND operator_result = 'NOK' LIMIT 1), 'OK'), "
                "model_result = COALESCE("
                "  (SELECT 'NOK' FROM classified_images "
                "   WHERE piece_id = piece_result.id AND model_result = 'NOK' LIMIT 1), 'OK') "
                "WHERE jsn = %s",
                (jsn,),
            )

            piece_rows = db.fetch(
                "SELECT id FROM piece_result WHERE jsn = %s",
                (jsn,),
            )
            if not piece_rows:
                return

            piece_id = piece_rows[0]["id"]
            db.execute(
                "DELETE FROM piece_result_defects WHERE piece_result_id = %s",
                (piece_id,),
            )
            db.execute(
                "INSERT INTO piece_result_defects (piece_result_id, class_name, confidence) "
                "SELECT %s, selected.class_name, selected.confidence "
                "FROM ("
                "  SELECT cid.class_name, cid.confidence "
                "  FROM classified_image_defects cid "
                "  JOIN classified_images ci ON ci.id = cid.classified_image_id "
                "  WHERE ci.piece_id = %s "
                "  AND ("
                "    UPPER(cid.class_name) <> 'OK' "
                "    OR NOT EXISTS ("
                "      SELECT 1 "
                "      FROM classified_image_defects cid_non_ok "
                "      JOIN classified_images ci_non_ok ON ci_non_ok.id = cid_non_ok.classified_image_id "
                "      WHERE ci_non_ok.piece_id = %s "
                "      AND UPPER(cid_non_ok.class_name) <> 'OK'"
                "    )"
                "  ) "
                "  ORDER BY cid.confidence DESC, cid.created_at DESC, cid.id DESC "
                "  LIMIT 1"
                ") AS selected",
                (piece_id, piece_id, piece_id),
            )
        except Exception as exc:
            print(f"Error recalculating piece_result for {jsn}: {exc}")

    def _update_result_in_db(self, img_name, new_value):
        d = self.display
        try:
            query_update = "UPDATE img_results SET result = %s WHERE img_name = %s"
            d.db.execute(query_update, (new_value, img_name))
            d._db_result_cache[img_name] = new_value

            d.db.execute(
                "UPDATE classified_images SET operator_result = %s WHERE img_name = %s",
                (new_value, img_name),
            )
            jsn = img_name.split("_")[0] if "_" in img_name else img_name
            self._recalculate_piece_result(jsn)
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
        visible_images_snapshot=None,
    ):
        d = self.display
        db = db_client or d.db
        historic_dir = historic_dir or self._get_export_historic_dir()
        base_dir = base_dir or str(SYNC_IMAGES_BASE_DIR)
        if visible_images_snapshot is None:
            visible_images_snapshot = self._get_visible_historic_image_snapshot()
        visible_images = set(visible_images_snapshot)

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

            if visible_images and img_name not in visible_images:
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
                error_count += 1
                print(f"Historic source missing for dataset sync: {img_name}")
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
            "rows_snapshot": rows,
            "copied": copied_count,
            "removed": removed_count,
            "errors": error_count,
            "visible_images": len(visible_images),
            "visible_images_snapshot": visible_images_snapshot,
        }

    def save_classification_results(
        self,
        db_client=None,
        historic_dir=None,
        progress_callback=None,
        visible_images_snapshot=None,
        export_stats_report=False,
    ):
        """Copy images to final_classification folders (P/N/FP/FN per position).

        DB writes happen at data arrival time (_upsert_classification).
        This method only handles folder copying based on current img_results state.
        """
        d = self.display
        db = db_client or d.db
        if not db:
            return {"ok": False, "error": "No DB connection"}
        if visible_images_snapshot is None:
            visible_images_snapshot = self._get_visible_historic_image_snapshot()
        visible_images = set(visible_images_snapshot)

        try:
            rows = db.fetch("SELECT img_name, result FROM img_results")
        except Exception as exc:
            return {"ok": False, "error": f"Error fetching img_results: {exc}"}

        if not rows:
            return {"ok": False, "error": "No rows in img_results"}

        # ---- build per-image data ----
        all_images = []
        for row in rows:
            img_name = row.get("img_name") or row.get("name")
            operator_result = row.get("result")
            if not img_name or operator_result is None:
                continue
            if visible_images and img_name not in visible_images:
                continue

            operator_result = str(operator_result).strip().upper()
            if operator_result not in ("OK", "NOK"):
                continue

            m = re.search(r"_(OK|NOK)\.\w+$", img_name, re.IGNORECASE)
            model_result = m.group(1).upper() if m else "OK"

            all_images.append({
                "img_name": img_name,
                "operator_result": operator_result,
                "model_result": model_result,
            })

        if not all_images:
            return {"ok": False, "error": "No valid images found"}

        # ---- copy images to final_classification folders ----
        historic_dir = historic_dir or self._get_export_historic_dir()
        base_dir = str(FINAL_CLASSIFICATION_DIR)

        for position_dirs in FINAL_CLASSIFICATION_DIRS.values():
            for folder_name in position_dirs.values():
                self.file_manager.makedirs(
                    self.file_manager.join(base_dir, folder_name), exist_ok=True
                )

        files_copied = 0
        copy_errors = []
        expected_counts = defaultdict(int)
        total_images = len(all_images)

        if callable(progress_callback):
            progress_callback(0, total_images, "Classifying images")

        for idx, img in enumerate(all_images, start=1):
            img_name = img["img_name"]
            op = img["operator_result"]
            mdl = img["model_result"]

            if op == "OK" and mdl == "OK":
                tag = "P"
            elif op == "NOK" and mdl == "NOK":
                tag = "N"
            elif op == "NOK" and mdl == "OK":
                tag = "FP"
            elif op == "OK" and mdl == "NOK":
                tag = "FN"
            else:
                if callable(progress_callback):
                    progress_callback(idx, total_images, "Classifying images")
                continue

            pos_match = re.search(r"(side|front|diag)", img_name, re.IGNORECASE)
            if not pos_match:
                if callable(progress_callback):
                    progress_callback(idx, total_images, "Classifying images")
                continue
            position = pos_match.group(1).lower()

            folder_name = FINAL_CLASSIFICATION_DIRS[position][tag]
            target_path = self.file_manager.join(base_dir, folder_name, img_name)
            source_path = self.file_manager.join(historic_dir, img_name)

            expected_counts[folder_name] += 1

            # Remove from wrong folders (idempotent: status may have changed)
            for other_tag, other_folder in FINAL_CLASSIFICATION_DIRS[position].items():
                if other_tag == tag:
                    continue
                old_path = self.file_manager.join(base_dir, other_folder, img_name)
                if self.file_manager.exists(old_path):
                    try:
                        self.file_manager.remove(old_path)
                    except Exception:
                        pass

            if not self.file_manager.exists(source_path):
                copy_errors.append(f"Source missing: {img_name}")
            else:
                try:
                    self.file_manager.copy2(source_path, target_path)
                    files_copied += 1
                except Exception as exc:
                    copy_errors.append(f"{img_name}: {exc}")

            if callable(progress_callback):
                progress_callback(idx, total_images, "Classifying images")

        # ---- verify copied files ----
        folder_errors = []
        for folder_name, expected in expected_counts.items():
            folder_path = self.file_manager.join(base_dir, folder_name)
            try:
                actual = len(self.file_manager.listdir(folder_path))
            except Exception:
                actual = 0
            if actual < expected:
                folder_errors.append(
                    f"{folder_name}: expected {expected}, found {actual}"
                )

        result = {
            "ok": True,
            "images": total_images,
            "files_copied": files_copied,
        }

        if copy_errors or folder_errors:
            all_errors = copy_errors + folder_errors
            result["classification_folder_errors"] = all_errors
            print(
                f"save_classification_results: {len(all_errors)} folder issues: "
                + "; ".join(all_errors[:5])
            )

        if export_stats_report:
            try:
                from report_exporter import export_stats_report

                result["stats_report_path"] = export_stats_report(
                    self,
                    db_client=db,
                )
            except Exception as exc:
                result["stats_report_error"] = str(exc)
                self.logger.warn(
                    f"[SYNC] Stats report export failed: {exc}",
                    allow_repeat=True,
                )

        return result

    def verify_sync_images_by_status(
        self,
        historic_dir=None,
        base_dir=None,
        db_client=None,
        progress_callback=None,
        rows_snapshot=None,
        visible_images_snapshot=None,
    ):
        db = db_client or self.display.db
        historic_dir = historic_dir or self._get_export_historic_dir()
        base_dir = base_dir or str(SYNC_IMAGES_BASE_DIR)
        visible_dir = self._get_visible_historic_dir()

        if not db:
            return {"verified": False, "issue_count": 1, "issues": {"db": ["No database connection"]}}

        if not self.file_manager.exists(historic_dir):
            return {
                "verified": False,
                "issue_count": 1,
                "issues": {"historic": [f"Historic folder not found: {historic_dir}"]},
            }

        rows = rows_snapshot if rows_snapshot is not None else db.fetch("SELECT img_name, result FROM img_results ORDER BY img_name")
        if not rows:
            return {
                "verified": False,
                "issue_count": 1,
                "issues": {"db_rows": ["img_results returned no rows"]},
            }

        image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        visible_images = None
        if visible_images_snapshot is not None:
            visible_images = sorted(visible_images_snapshot)
        elif self.file_manager.exists(visible_dir):
            visible_images = sorted(
                name
                for name in self.file_manager.listdir(visible_dir)
                if self.file_manager.is_file(self.file_manager.join(visible_dir, name))
                and any(name.lower().endswith(ext) for ext in image_extensions)
            )
        elif rows_snapshot is None:
            return {
                "verified": False,
                "issue_count": 1,
                "issues": {"annotated": [f"Annotated folder not found: {visible_dir}"]},
            }

        if rows_snapshot is not None:
            row_images = sorted(
                row.get("img_name") or row.get("name")
                for row in rows_snapshot
                if (row.get("img_name") or row.get("name"))
                and any((row.get("img_name") or row.get("name", "")).lower().endswith(ext) for ext in image_extensions)
            )
            if visible_images is None:
                historic_images = row_images
            else:
                visible_set = set(visible_images)
                historic_images = [img_name for img_name in row_images if img_name in visible_set]
        else:
            historic_images = visible_images or []
        if not historic_images:
            return {
                "verified": False,
                "issue_count": 1,
                "issues": {"annotated_images": ["No image files found in annotated folder"]},
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
        missing_historic_source = []

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

            source_path = self.file_manager.join(historic_dir, img_name)
            if not self.file_manager.exists(source_path):
                missing_historic_source.append(img_name)
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
            "missing_historic_source": missing_historic_source,
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
            visible_dir = self._get_visible_historic_dir()
            image_path = self.file_manager.join(visible_dir, first_image)
            if self.file_manager.exists(image_path):
                import datetime

                mtime = self.file_manager.getmtime(image_path)
                dt = datetime.datetime.fromtimestamp(mtime)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            return "N/A"
        except Exception as exc:
            print(f"Error getting piece date: {exc}")
            return "N/A"

    def get_piece_class_summary(self, db_client=None):
        db = db_client or self.display.db
        if not db:
            return []

        try:
            rows = db.fetch(
                "SELECT COALESCE(prd.class_name, 'UNCLASSIFIED') AS class_name, "
                "COUNT(DISTINCT pr.id) AS piece_count "
                "FROM piece_result pr "
                "LEFT JOIN piece_result_defects prd ON prd.piece_result_id = pr.id "
                "WHERE pr.final_result IN ('OK', 'NOK', 'FOK', 'FNOK') "
                "GROUP BY COALESCE(prd.class_name, 'UNCLASSIFIED') "
                "ORDER BY piece_count DESC, class_name ASC"
            )
            return self._append_summary_total_row(rows, "class_name")
        except Exception as exc:
            self.logger.error(f"Error fetching piece class summary: {exc}")
            return []

    def get_piece_status_summary(self, db_client=None):
        db = db_client or self.display.db
        if not db:
            return []

        try:
            rows = db.fetch(
                "SELECT statuses.final_result, COALESCE(COUNT(pr.id), 0) AS piece_count "
                "FROM (VALUES ('OK', 1), ('NOK', 2), ('FNOK', 3), ('FOK', 4)) "
                "AS statuses(final_result, sort_order) "
                "LEFT JOIN piece_result pr ON pr.final_result = statuses.final_result "
                "GROUP BY statuses.final_result, statuses.sort_order "
                "ORDER BY statuses.sort_order ASC"
            )
            return self._append_summary_total_row(rows, "final_result")
        except Exception as exc:
            self.logger.error(f"Error fetching piece status summary: {exc}")
            return []

    def _append_summary_total_row(self, rows, label_key):
        normalized_rows = []
        total_pieces = 0

        for row in rows or []:
            row_data = dict(row)
            try:
                piece_count = int(row_data.get("piece_count", 0) or 0)
            except (TypeError, ValueError):
                piece_count = 0
            row_data["piece_count"] = piece_count
            row_data["is_total"] = False
            normalized_rows.append(row_data)
            total_pieces += piece_count

        if normalized_rows:
            normalized_rows.append(
                {
                    label_key: "Total",
                    "piece_count": total_pieces,
                    "is_total": True,
                }
            )
        return normalized_rows

    def build_piece_stats_report(self, db_client=None):
        db = db_client or self.display.db
        statuses = ("OK", "NOK", "FOK", "FNOK")
        if not db:
            return {
                "columns": list(statuses) + ["Total"],
                "rows": [],
                "start_at": None,
                "end_at": None,
            }

        try:
            aggregate_rows = db.fetch(
                "SELECT COALESCE(prd.class_name, 'UNCLASSIFIED') AS class_name, "
                "pr.final_result, COUNT(*) AS piece_count "
                "FROM piece_result pr "
                "LEFT JOIN piece_result_defects prd ON prd.piece_result_id = pr.id "
                "WHERE pr.final_result IN ('OK', 'NOK', 'FOK', 'FNOK') "
                "GROUP BY COALESCE(prd.class_name, 'UNCLASSIFIED'), pr.final_result"
            )
            date_range_rows = db.fetch(
                "SELECT MIN(created_at) AS start_at, MAX(created_at) AS end_at "
                "FROM piece_result"
            )
        except Exception as exc:
            self.logger.error(f"Error building piece stats report: {exc}")
            return {
                "columns": list(statuses) + ["Total"],
                "rows": [],
                "start_at": None,
                "end_at": None,
                "error": str(exc),
            }

        row_map = {}
        for row in aggregate_rows or []:
            status = str(row.get("final_result") or "").strip().upper()
            if status not in statuses:
                continue

            class_name = str(row.get("class_name") or "").strip() or "UNCLASSIFIED"
            normalized_name = class_name.upper()
            if normalized_name == "OK":
                class_name = "OK"
            elif normalized_name == "UNCLASSIFIED":
                class_name = "UNCLASSIFIED"

            row_entry = row_map.setdefault(
                class_name,
                {
                    "class_name": class_name,
                    "OK": 0,
                    "NOK": 0,
                    "FOK": 0,
                    "FNOK": 0,
                    "Total": 0,
                    "is_total": False,
                },
            )
            try:
                piece_count = int(row.get("piece_count", 0) or 0)
            except (TypeError, ValueError):
                piece_count = 0
            row_entry[status] += piece_count
            row_entry["Total"] += piece_count

        def _sort_key(label):
            row_total = row_map[label]["Total"]
            return (-row_total, str(label).lower())

        ordered_labels = []
        if "OK" in row_map:
            ordered_labels.append("OK")

        for label in sorted(
            [
                current
                for current in row_map
                if current not in {"OK", "UNCLASSIFIED"}
            ],
            key=_sort_key,
        ):
            ordered_labels.append(label)

        if "UNCLASSIFIED" in row_map:
            ordered_labels.append("UNCLASSIFIED")

        ordered_rows = [row_map[label] for label in ordered_labels]
        if ordered_rows:
            total_row = {
                "class_name": "Total",
                "OK": sum(row["OK"] for row in ordered_rows),
                "NOK": sum(row["NOK"] for row in ordered_rows),
                "FOK": sum(row["FOK"] for row in ordered_rows),
                "FNOK": sum(row["FNOK"] for row in ordered_rows),
                "Total": sum(row["Total"] for row in ordered_rows),
                "is_total": True,
            }
            ordered_rows.append(total_row)

        range_row = (date_range_rows or [{}])[0]
        return {
            "columns": list(statuses) + ["Total"],
            "rows": ordered_rows,
            "start_at": range_row.get("start_at"),
            "end_at": range_row.get("end_at"),
        }

    def _get_historic_jsn_index_map(self):
        """Map each historic JSN to the 1-based piece number shown in historic mode."""
        try:
            historic_index = self._load_historic_index(force_rescan=False) or []
        except Exception as exc:
            self.logger.error(f"Error loading historic index map: {exc}")
            return {}

        total_batches = len(historic_index)
        jsn_index_map = {}
        for idx, batch in enumerate(historic_index):
            if not batch:
                continue
            jsn = batch[0].split("_")[0] if "_" in batch[0] else batch[0]
            if jsn and jsn not in jsn_index_map:
                jsn_index_map[jsn] = total_batches - idx
        return jsn_index_map

    def _attach_historic_indices(self, rows, jsn_key="jsn"):
        jsn_index_map = self._get_historic_jsn_index_map()
        enriched_rows = []
        for row in rows or []:
            row_data = dict(row)
            jsn_value = str(row_data.get(jsn_key) or "").strip()
            row_data["historic_index"] = jsn_index_map.get(jsn_value)
            enriched_rows.append(row_data)
        return enriched_rows

    def _normalize_piece_date_display(self, value):
        if value is None:
            return "N/A"

        if hasattr(value, "strftime"):
            try:
                return value.strftime("%Y-%m-%d-%H-%M")
            except Exception:
                pass

        text = str(value).strip()
        if not text:
            return "N/A"
        text = text.replace("T", " ")
        if len(text) >= 16:
            text = text[:16]
        return text.replace(" ", "-").replace(":", "-")

    def get_piece_jsns_for_class(self, class_name, db_client=None):
        db = db_client or self.display.db
        normalized_class_name = str(class_name or "").strip()
        if not db or not normalized_class_name:
            return []

        try:
            rows = db.fetch(
                "SELECT DISTINCT pr.jsn, pr.created_at, pr.final_result "
                "FROM piece_result pr "
                "LEFT JOIN piece_result_defects prd ON prd.piece_result_id = pr.id "
                "WHERE COALESCE(prd.class_name, 'UNCLASSIFIED') = %s "
                "AND pr.jsn IS NOT NULL "
                "AND pr.jsn <> '' "
                "ORDER BY pr.jsn DESC",
                (normalized_class_name,),
            )
            enriched_rows = self._attach_historic_indices(rows)
            for row in enriched_rows:
                row["piece_date_display"] = self._normalize_piece_date_display(
                    row.get("created_at")
                )
            return enriched_rows
        except Exception as exc:
            self.logger.error(f"Error fetching JSNs for class '{normalized_class_name}': {exc}")
            return []

    def get_piece_jsns_for_status(self, final_result, db_client=None):
        db = db_client or self.display.db
        normalized_result = str(final_result or "").strip().upper()
        if not db or normalized_result not in {"OK", "NOK", "FNOK", "FOK"}:
            return []

        try:
            rows = db.fetch(
                "SELECT pr.jsn, pr.created_at, "
                "COALESCE(prd.class_name, 'UNCLASSIFIED') AS class_name "
                "FROM piece_result pr "
                "LEFT JOIN piece_result_defects prd ON prd.piece_result_id = pr.id "
                "WHERE pr.final_result = %s "
                "AND pr.jsn IS NOT NULL "
                "AND pr.jsn <> '' "
                "ORDER BY pr.jsn DESC",
                (normalized_result,),
            )
            enriched_rows = self._attach_historic_indices(rows)
            for row in enriched_rows:
                row["piece_date_display"] = self._normalize_piece_date_display(
                    row.get("created_at")
                )
            return enriched_rows
        except Exception as exc:
            self.logger.error(
                f"Error fetching JSNs for final_result '{normalized_result}': {exc}"
            )
            return []

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
        elif action == "send_remote_input":
            if self.stdin_queue is not None:
                self.stdin_queue.put("t\n")
                d.manual_capture_pending = True
                _show_capture_modal(d, "capturing", (0, 0, 200))
        elif action == "next_historic_batch":
            self.next_historic_batch()
        elif action == "prev_historic_batch":
            self.prev_historic_batch()
        elif action == "open_piece_date_dialog":
            self._close_piece_number_dialog(clear_input=False)
            d.show_piece_date_dialog = True
        elif action == "close_piece_date_dialog":
            d.show_piece_date_dialog = False
        elif action == "open_piece_number_dialog":
            self._open_piece_number_dialog()
        elif action == "close_piece_number_dialog":
            self._close_piece_number_dialog(clear_input=True)
        elif action == "piece_number_append_digit":
            digit = payload.get("digit")
            if digit is not None and str(digit).isdigit():
                max_length = max(1, len(str(max(1, len(d.historic_images or [])))))
                digit_str = str(digit)
                current_input = str(getattr(d, "piece_number_dialog_input", "") or "")
                if getattr(d, "piece_number_dialog_replace_on_input", False):
                    next_input = digit_str
                else:
                    next_input = f"{current_input}{digit_str}"[:max_length]
                d.piece_number_dialog_input = next_input[:max_length]
                d.piece_number_dialog_replace_on_input = False
        elif action == "piece_number_backspace":
            d.piece_number_dialog_input = str(getattr(d, "piece_number_dialog_input", "") or "")[:-1]
            d.piece_number_dialog_replace_on_input = False
        elif action == "submit_piece_number_dialog":
            piece_number = str(getattr(d, "piece_number_dialog_input", "") or "").strip()
            if self.go_to_historic_piece_number(piece_number, show_missing_dialog=True):
                self._close_piece_number_dialog(clear_input=True)
        elif action == "open_stats_class_modal":
            if hasattr(d, "_reset_stats_class_modal_state"):
                d._reset_stats_class_modal_state()
            d.stats_class_modal_rows = self.get_piece_class_summary()
            d.stats_class_modal_status_rows = self.get_piece_status_summary()
            d.stats_class_modal_matrix_rows = self.build_piece_stats_report().get("rows", [])
            d.show_stats_class_modal = True
        elif action == "close_stats_class_modal":
            d.show_stats_class_modal = False
            if hasattr(d, "_reset_stats_class_modal_state"):
                d._reset_stats_class_modal_state()
        elif action == "open_stats_summary_view":
            d.stats_class_modal_view = "summary"
        elif action == "open_stats_matrix_view":
            d.stats_class_modal_view = "matrix"
            d.stats_class_modal_matrix_offset = 0
        elif action == "open_stats_class_detail":
            class_name = str(payload.get("class_name") or "").strip()
            d.stats_class_modal_view = "detail"
            d.stats_class_modal_selected_kind = "class"
            d.stats_class_modal_selected_label = class_name
            d.stats_class_modal_detail_rows = self.get_piece_jsns_for_class(class_name)
            d.stats_class_modal_detail_offset = 0
        elif action == "open_stats_status_detail":
            final_result = str(payload.get("final_result") or "").strip().upper()
            d.stats_class_modal_view = "detail"
            d.stats_class_modal_selected_kind = "status"
            d.stats_class_modal_selected_label = final_result
            d.stats_class_modal_detail_rows = self.get_piece_jsns_for_status(final_result)
            d.stats_class_modal_detail_offset = 0
        elif action == "close_stats_class_detail":
            d.stats_class_modal_view = "summary"
            d.stats_class_modal_selected_kind = ""
            d.stats_class_modal_selected_label = ""
            d.stats_class_modal_detail_rows = []
            d.stats_class_modal_detail_offset = 0
            d.stats_class_modal_detail_visible_rows = 1
        elif action == "open_historic_jsn_from_stats":
            self.go_to_historic_jsn(
                payload.get("jsn"),
                show_missing_dialog=True,
                close_stats_modal=True,
            )
        elif action == "copy_stats_jsn":
            d._copy_stats_modal_jsn(payload.get("jsn"))
        elif action == "stats_detail_scroll":
            delta = int(payload.get("delta") or 0)
            if delta:
                steps = max(1, abs(delta) // 120)
                direction = -1 if delta > 0 else 1
                d.stats_class_modal_detail_offset += direction * steps
                if hasattr(d, "_clamp_stats_class_modal_detail_offset"):
                    d._clamp_stats_class_modal_detail_offset()
        elif action == "stats_matrix_scroll":
            delta = int(payload.get("delta") or 0)
            if delta:
                steps = max(1, abs(delta) // 120)
                direction = -1 if delta > 0 else 1
                d.stats_class_modal_matrix_offset += direction * steps
                if hasattr(d, "_clamp_stats_class_modal_matrix_offset"):
                    d._clamp_stats_class_modal_matrix_offset()
        elif action == "open_reset_confirm":
            d.show_reset_confirm = True
            d.show_delete_confirm = False
            d.show_rebuild_confirm = False
        elif action == "cancel_reset_confirm":
            d.show_reset_confirm = False
        elif action == "confirm_reset":
            d.show_reset_confirm = False
            self.start_reset_async()
        elif action == "open_delete_confirm":
            d.show_delete_confirm = True
            d.show_reset_confirm = False
            d.show_rebuild_confirm = False
        elif action == "cancel_delete_confirm":
            d.show_delete_confirm = False
        elif action == "confirm_delete":
            d.show_delete_confirm = False
            self.perform_delete_current_piece()
        elif action == "open_rebuild_db_confirm":
            d.show_rebuild_confirm = True
            d.show_reset_confirm = False
            d.show_delete_confirm = False
        elif action == "cancel_rebuild_db_confirm":
            d.show_rebuild_confirm = False
        elif action == "confirm_rebuild_db_from_historic":
            d.show_rebuild_confirm = False
            self.start_rebuild_db_from_historic_async()
        elif action == "sync_images_by_status":
            self.start_sync_images_by_status_async()
        elif action == "export_display_state":
            self.start_export_display_state_async()
        elif action == "import_display_state":
            self.start_import_display_state_async(payload.get("package_path"))
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
        elif action == "search_paste":
            pasted_text = self._sanitize_search_jsn(payload.get("text"))
            if pasted_text:
                d.search_jsn = pasted_text
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

                # Periodic check for new historic images
                if time.monotonic() - self.last_historic_check > 1.0:
                    self._check_and_register_new_historic_images()
                    self.last_historic_check = time.monotonic()

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

                    if self._pending_remote_images:
                        # Don't download again until the current remote batch is confirmed on disk
                        if all(self.file_manager.exists(p) for p in self._pending_remote_images):
                            images = self._pending_remote_images
                            self._pending_remote_images = None
                        else:
                            images = self.display.image_paths or []
                    elif self.sftp_connected and self.sftp_app:
                        remote_images = self._download_live_images_remote()
                        if not self.sftp_app.sftp_client:
                            self.stop_remote_process("sftp-disconnect")
                            self.handle_disconnect("live-download-failure")
                        elif remote_images:
                            # Delete old files immediately so tmp_display never exceeds max_images
                            new_images_set = set(remote_images)
                            for prev_path in self.display.image_paths:
                                if prev_path not in new_images_set:
                                    try:
                                        self.file_manager.remove(prev_path)
                                    except Exception:
                                        pass
                            self._pending_remote_images = remote_images
                            # Live batches in tmp_display are display-only; DB state
                            # must come from annotated images exclusively.
                            images = self.display.image_paths or []
                        else:
                            images = self._download_live_images_local()
                    else:
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
        self.stop_remote_db_polling()
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

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from threading import Thread

from paths_config import EXPORTS_DIR
from state_package import estimate_display_state_export_size, export_display_state


class DailyExportMaintenance:
    """Owns the daily export/reset workflow so MainController only calls tick()."""

    def __init__(self, controller, now_fn=None):
        self.controller = controller
        self.file_manager = controller.file_manager
        self.logger = controller.logger
        self.config = controller.config
        self.now_fn = now_fn or datetime.now
        self.worker_thread = None

        self.enabled = bool(getattr(self.config, "daily_maintenance_enabled", True))
        self.hour = int(getattr(self.config, "daily_maintenance_hour", 5))
        self.minute = int(getattr(self.config, "daily_maintenance_minute", 45))
        self.min_free_bytes = int(
            getattr(self.config, "daily_maintenance_min_free_bytes", 512 * 1024 * 1024)
            or 0
        )
        self.retry_interval_sec = float(
            getattr(self.config, "daily_maintenance_retry_interval_sec", 30 * 60)
            or 0
        )
        self.exports_dir = Path(
            getattr(self.config, "daily_maintenance_exports_dir", str(EXPORTS_DIR))
            or EXPORTS_DIR
        )
        self.state_path = self.exports_dir / ".daily_export_reset_state.json"

    def tick(self):
        if not self.enabled:
            return False
        if self._worker_is_running():
            return False
        if getattr(self.controller, "historic_bootstrap_loading", False):
            return False

        now = self.now_fn()
        scheduled_at = now.replace(
            hour=self.hour,
            minute=self.minute,
            second=0,
            microsecond=0,
        )
        if now < scheduled_at:
            return False

        state = self._load_state()
        today = now.date().isoformat()
        if state.get("last_success_date") == today:
            return False

        last_attempt_at = self._parse_datetime(state.get("last_attempt_at"))
        if last_attempt_at and last_attempt_at.date().isoformat() == today:
            elapsed = (now - last_attempt_at).total_seconds()
            if elapsed < self.retry_interval_sec:
                return False

        return self.start_async(reason="scheduled")

    def start_async(self, reason="manual"):
        d = self.controller.display
        if self._worker_is_running():
            return False
        if getattr(d, "sync_in_progress", False) or getattr(d, "reset_in_progress", False):
            return False

        self.controller.dataset_transfer_active = True
        self.worker_thread = Thread(
            target=self._run,
            args=(reason,),
            name="daily-export-maintenance-worker",
            daemon=True,
        )
        self.worker_thread.start()
        return True

    def _worker_is_running(self):
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def _run(self, reason):
        d = self.controller.display
        worker_db = None
        worker_state = None
        export_result = None
        reset_result = None
        status = "failed"
        error_text = None
        self._save_state(
            {
                **self._load_state(),
                "last_attempt_at": self.now_fn().isoformat(sep=" ", timespec="seconds"),
                "last_status": "running",
                "last_error": None,
            }
        )

        self.controller.dataset_transfer_active = True
        d.reset_in_progress = True
        d.reset_progress = 0
        d.reset_progress_title = "Daily Export Maintenance"
        d.reset_progress_helper_text = "Exporting current state, then resetting dataset and DB."
        d.reset_stage = "Preparing daily export..."
        d.sync_message = ""
        d.sync_message_is_error = False
        d.sync_message_time = 0
        d.sync_message_auto_dismiss_sec = None

        try:
            worker_state = self.controller._pause_dataset_background_workers()

            from db import get_db_connection

            worker_db = get_db_connection()

            self._set_progress("Checking export storage...", 5)
            estimate = estimate_display_state_export_size(
                self.controller,
                db_client=worker_db,
            )
            if not estimate.get("ok", False):
                raise RuntimeError(estimate.get("error", "Unable to estimate export size"))

            required_bytes = int(estimate.get("required_bytes", 0) or 0)
            storage_check = self._check_export_storage(required_bytes)
            if not storage_check.get("ok", False):
                raise RuntimeError(
                    storage_check.get("error", "Not enough storage for export")
                )

            self._set_progress("Creating export...", 15)

            def _export_progress(done, total, stage):
                percent = self._phase_percent(done, total, 15, 50)
                self._set_progress(stage, percent)

            export_result = export_display_state(
                self.controller,
                output_dir=str(self.exports_dir),
                db_client=worker_db,
                progress_callback=_export_progress,
            )
            if not export_result.get("ok", False):
                raise RuntimeError(export_result.get("error", "Dataset export failed"))

            def _reset_progress(done, total, stage):
                percent = self._phase_percent(done, total, 50, 90)
                self._set_progress(stage, percent)

            self._set_progress("Resetting dataset...", 50)
            reset_result = self.controller.perform_reset(
                db_client=worker_db,
                progress_callback=_reset_progress,
            )

            self._set_progress("Resetting database...", 92)
            truncated_tables = worker_db.truncate_app_tables()
            self.logger.info(
                f"[DAILY_MAINTENANCE] Truncated {truncated_tables} app tables",
                allow_repeat=True,
            )

            self._set_progress("Completed", 100)
            package_name = export_result.get("package_name") or os.path.basename(
                str(export_result.get("package_path") or "")
            )
            if reset_result and not reset_result.get("ok", False):
                status = "success_with_reset_issues"
                error_text = reset_result.get("error", "Reset completed with issues")
                d.sync_message = (
                    f"Daily export completed: {package_name}; reset had issues: {error_text}"
                )
                d.sync_message_is_error = True
            else:
                status = "success"
                d.sync_message = f"Daily export/reset completed: {package_name}"
                d.sync_message_is_error = False
                d.sync_message_auto_dismiss_sec = 5.0
            self.logger.info(
                f"[DAILY_MAINTENANCE] Completed daily export/reset ({reason}): "
                f"{export_result.get('package_path')}",
                allow_repeat=True,
            )
        except Exception as exc:
            error_text = str(exc)
            d.sync_message = f"Daily export/reset failed: {error_text}"
            d.sync_message_is_error = True
            d.sync_message_auto_dismiss_sec = None
            self.logger.error(
                f"[DAILY_MAINTENANCE] Daily export/reset failed: {exc}",
                allow_repeat=True,
            )
        finally:
            state = self._load_state()
            state.update(
                {
                    "last_attempt_at": self.now_fn().isoformat(sep=" ", timespec="seconds"),
                    "last_status": status,
                    "last_export_path": (
                        str(export_result.get("package_path"))
                        if export_result and export_result.get("package_path")
                        else state.get("last_export_path")
                    ),
                    "last_error": error_text,
                }
            )
            if status.startswith("success"):
                state["last_success_date"] = self.now_fn().date().isoformat()
            self._save_state(state)

            if worker_state is not None:
                self.controller._resume_dataset_background_workers(worker_state)
            self.controller.dataset_transfer_active = False
            d.reset_in_progress = False
            d.sync_message_time = time.time()
            if worker_db is not None:
                try:
                    worker_db.close()
                except Exception:
                    pass

    def _set_progress(self, stage, percent):
        self.controller._set_reset_progress(
            stage,
            percent,
            title="Daily Export Maintenance",
            helper_text="Exporting current state, then resetting dataset and DB.",
        )

    def _phase_percent(self, done, total, start, end):
        if total <= 0:
            return start
        fraction = max(0.0, min(1.0, float(done) / float(total)))
        return int(start + ((end - start) * fraction))

    def _check_export_storage(self, required_bytes):
        self.file_manager.makedirs(str(self.exports_dir), exist_ok=True)
        disk_root = str(self.exports_dir.resolve())
        usage = shutil.disk_usage(disk_root)
        required_bytes = max(0, int(required_bytes))
        target_free = max(required_bytes * 2, required_bytes + self.min_free_bytes)
        initial_free = int(usage.free)
        if initial_free >= target_free:
            return {
                "ok": True,
                "required_bytes": required_bytes,
                "target_free_bytes": target_free,
                "available_free_bytes": initial_free,
                "deleted_exports": [],
            }

        deleted_exports = []
        current_free = initial_free
        for export_dir in self._list_deletable_export_dirs():
            export_name = export_dir.name
            try:
                self.file_manager.rmtree(str(export_dir))
                deleted_exports.append(export_name)
                usage = shutil.disk_usage(disk_root)
                current_free = int(usage.free)
                self.logger.info(
                    "[DAILY_MAINTENANCE] Deleted old export "
                    f"{export_name}; free bytes now {current_free}",
                    allow_repeat=True,
                )
            except Exception as exc:
                self.logger.error(
                    f"[DAILY_MAINTENANCE] Unable to delete old export {export_name}: {exc}",
                    allow_repeat=True,
                )
                continue

            if current_free >= target_free:
                return {
                    "ok": True,
                    "required_bytes": required_bytes,
                    "target_free_bytes": target_free,
                    "available_free_bytes": current_free,
                    "deleted_exports": deleted_exports,
                }

        return {
            "ok": False,
            "required_bytes": required_bytes,
            "target_free_bytes": target_free,
            "available_free_bytes": current_free,
            "deleted_exports": deleted_exports,
            "error": (
                "Not enough storage for daily export/reset; export and reset were skipped "
                f"(need {target_free} free bytes, available {current_free}, "
                f"deleted {len(deleted_exports)} old exports)"
            ),
        }

    def _list_deletable_export_dirs(self):
        if not self.exports_dir.exists():
            return []

        candidates = []
        for item in self.exports_dir.iterdir():
            try:
                if not item.is_dir():
                    continue
                if not item.name.startswith("display_state"):
                    continue
                if not (item / "manifest.json").is_file():
                    continue
                candidates.append(item)
            except Exception as exc:
                self.logger.error(
                    f"[DAILY_MAINTENANCE] Unable to inspect export {item}: {exc}",
                    allow_repeat=True,
                )

        return sorted(
            candidates,
            key=lambda path: (
                path.stat().st_mtime if path.exists() else 0,
                path.name,
            ),
        )

    def _load_state(self):
        try:
            if not self.state_path.exists():
                return {}
            with self.state_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            self.logger.warn(
                f"[DAILY_MAINTENANCE] Unable to load state file: {exc}",
                allow_repeat=True,
            )
            return {}

    def _save_state(self, state):
        try:
            self.file_manager.makedirs(str(self.exports_dir), exist_ok=True)
            tmp_path = self.state_path.with_suffix(".tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2, sort_keys=True)
            os.replace(tmp_path, self.state_path)
        except Exception as exc:
            self.logger.error(
                f"[DAILY_MAINTENANCE] Unable to save state file: {exc}",
                allow_repeat=True,
            )

    def _parse_datetime(self, value):
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return None

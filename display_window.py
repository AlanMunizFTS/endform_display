
import cv2
import copy
import json
import numpy as np
import os
import re
import subprocess
import time
from collections import OrderedDict
from multiprocessing import Event, Process
from db import get_db_connection
from file_manager import FileManager
from main_controller import check_historic_images as controller_check_historic_images
from paths_config import REMOTE_HIST_DISPLAY_DIR

class DisplayWindow:
    TRASH_ICON_PATH = "./resources/trash.png"
    BACKGROUND_IMAGE_PATH = "./resources/base_screen.png"
    DEFAULT_TILE_SIZE = 360
    DEFAULT_TILE_PADDING = 96


    def __init__(
        self,
        width=800,
        height=600,
        window_name="Display Window",
        refresh_interval=5,
        sftp_client=None,
        filename_mapping=None,
        sftp_credentials=None,
        file_manager=None,
        controller=None,
        action_handler=None,
    ):
        self.width = width
        self.height = height
        self.window_name = window_name
        self.image = None
        self.save_button_rect = None  # Save button
        self.back_button_rect = None  # Back button
        self.next_button_rect = None  # Next arrow button
        self.prev_button_rect = None  # Previous arrow button
        self.save_changes_button_rect = None  # SAVE button to save changes
        self.image_paths = []  # Image paths
        self.remote_hist_dir = REMOTE_HIST_DISPLAY_DIR  # Remote folder for history
        self.refresh_interval = refresh_interval  # Seconds between updates
        self.last_refresh_time = 0
        self.sftp_client = None  # SFTP client to upload images
        self.sftp_credentials = sftp_credentials  # Credenciales SFTP para multiprocessing
        self.file_manager = file_manager or FileManager()
        self.controller = controller
        self.action_handler = action_handler
        self.filename_mapping = filename_mapping or {}  # Mapping of short names to original names
        self.historic_mode = False  # Indicates if we are in historic mode
        self.historic_offset = 0  # Offset to navigate through historic batches
        self.historic_images = []  # Complete list of historic images
        self.historic_filter_kind = ""
        self.historic_filter_label = ""
        self.historic_filter_jsns = []
        self.historic_filter_total_count = 0
        self.result_buttons = []  # List of result buttons [(rect, img_name, result_value), ...]
        self.temp_results = {}  # Dictionary for temporary changes {img_name: new_value}
        self.db = None
        self.db_blocking = False
        self.db_block_message = ""
        self.db_block_detail = ""
        try:
            self.set_db_connection(get_db_connection())
        except Exception as exc:
            self.set_db_blocked(
                "PostgreSQL is disconnected. Start postgres and wait for automatic reconnect."
            )
            self.db_block_detail = str(exc)
        self.download_process = None  # Process for background download
        self.download_stop_event = None
        self.annotated_download_process = None  # Process for annotated background download
        self.annotated_download_stop_event = None
        self.historic_db_registered = False  # Tracks whether visible historic images were registered in DB.
        self.search_button_rect = None  # Search button rect
        self.search_input_rect = None  # Search input field rect
        self.search_jsn = ""  # Current JSN search term
        self.search_active = False  # Whether search input is active
        self.historic_jsn_rect = None  # Clickable JSN banner in historic mode
        self.available_jsns = []  # List of all available JSNs
        self.filtered_suggestions = []  # Filtered suggestions based on input
        self.selected_suggestion_idx = -1  # Index of selected suggestion (-1 = none)
        self.suggestion_rects = []  # Rectangles for each suggestion
        self.reset_button_rect = None  # Reset button rect
        self.trash_button_rect = None  # Trash button rect
        self.sync_button_rect = None  # Sync button rect
        self.image_report_button_rect = None  # Historic image report button rect
        self._image_report_dialog_root = None
        self._image_report_dialog_result = None
        self._verdict_analysis_dialog_root = None
        self._verdict_analysis_dialog_request = None
        self._verdict_analysis_rows = []
        self._verdict_analysis_tree = None
        self._verdict_analysis_metric_vars = {}
        self._verdict_analysis_completion_var = None
        self._verdict_analysis_required_angles = ()
        self._verdict_analysis_confidence_thresholds = {}
        self._verdict_analysis_threshold_vars = {}
        self._verdict_analysis_threshold_summary_var = None
        self._verdict_analysis_optimization_var = None
        self._verdict_analysis_dirty = False
        self.import_button_rect = None  # Import button rect
        self.export_button_rect = None  # Export button rect
        self.exit_button_rect = None  # Exit button rect
        self.show_reset_confirm = False  # Show reset confirmation dialog
        self.reset_confirm_button_rect = None  # Confirm button rect
        self.reset_cancel_button_rect = None  # Cancel button rect
        self.show_delete_confirm = False  # Show delete-piece confirmation dialog
        self.delete_confirm_button_rect = None  # Confirm delete button rect
        self.delete_cancel_button_rect = None  # Cancel delete button rect
        self.show_rebuild_confirm = False  # Show rebuild confirmation dialog
        self.rebuild_confirm_button_rect = None  # Confirm rebuild button rect
        self.rebuild_cancel_button_rect = None  # Cancel rebuild button rect
        self.show_no_images_dialog = False  # Show no images available dialog
        self.no_images_dialog_message = "No images available"
        self.no_images_ok_button_rect = None  # OK button rect for no images dialog
        self.sync_message = ""
        self.sync_message_is_error = False
        self.sync_message_time = 0
        self.sync_message_auto_dismiss_sec = None
        self.sync_message_close_button_rect = None
        self.sync_in_progress = False
        self.sync_progress = 0
        self.sync_stage = ""
        self.sync_progress_title = "Saving Dataset"
        self.sync_progress_helper_text = "Please wait until the process finishes."
        self.reset_in_progress = False
        self.reset_progress = 0
        self.reset_stage = ""
        self.reset_progress_title = "Resetting Dataset"
        self.reset_progress_helper_text = "Clearing historic, annotated, classified, and final folders."
        self.exit_requested = False
        self.trash_icon = None
        self.trash_icon_size = None
        self._trash_icon_warned = False
        # Shift right-aligned status text/icon left (pixels)
        self.right_info_shift = 140
        self.info_icon_rect = None  # Info icon rect (top right in historic mode)
        self.show_piece_date_dialog = False  # Show piece date dialog
        self.piece_date_dialog_close_rect = None  # Close button rect for date dialog
        self.show_piece_number_dialog = False  # Show go-to-piece dialog
        self.piece_number_dialog_input = ""
        self.piece_number_dialog_replace_on_input = False
        self.piece_number_dialog_ok_rect = None
        self.piece_number_dialog_cancel_rect = None
        self.piece_identifier_rect = None
        self.show_piece_identifier_dialog = False
        self.piece_identifier_dialog_input = ""
        self.piece_identifier_dialog_replace_on_input = False
        self.piece_identifier_dialog_cancel_rect = None
        self.piece_identifier_dialog_save_rect = None
        self.piece_identifier_dialog_continue_rect = None
        self.piece_identifier_dialog_clear_rect = None
        self.piece_counter_rect = None  # Clickable counter above reset button
        self.show_stats_class_modal = False
        self.stats_class_modal_close_rect = None
        self.stats_class_modal_rows = []
        self.stats_class_modal_status_rows = []
        self.stats_class_modal_matrix_rows = []
        self.stats_class_modal_view = "summary"
        self.stats_class_modal_selected_kind = ""
        self.stats_class_modal_selected_label = ""
        self.stats_class_modal_detail_rows = []
        self.stats_class_modal_detail_offset = 0
        self.stats_class_modal_detail_visible_rows = 1
        self.stats_class_modal_matrix_offset = 0
        self.stats_class_modal_matrix_visible_rows = 1
        self.stats_class_modal_class_row_rects = []
        self.stats_class_modal_status_row_rects = []
        self.stats_class_modal_jsn_row_rects = []
        self.stats_class_modal_copy_rects = []
        self.stats_class_modal_back_rect = None
        self.stats_class_modal_summary_tab_rect = None
        self.stats_class_modal_matrix_tab_rect = None
        self.stats_class_modal_matrix_report_rect = None
        self.stats_class_modal_dataset_tab_rect = None
        self.stats_class_modal_list_rect = None
        self.stats_class_modal_scrollbar_rect = None
        self.stats_class_modal_dataset_result_options = []
        self.stats_class_modal_dataset_angle_options = []
        self.stats_class_modal_dataset_class_options = []
        self.stats_class_modal_dataset_selected_results = set()
        self.stats_class_modal_dataset_selected_angles = set()
        self.stats_class_modal_dataset_selected_classes = set()
        self.stats_class_modal_dataset_result_rects = []
        self.stats_class_modal_dataset_angle_rects = []
        self.stats_class_modal_dataset_class_rects = []
        self.stats_class_modal_dataset_export_rect = None
        self.stats_class_modal_dataset_class_offset = 0
        self.stats_class_modal_dataset_class_visible_rows = 1
        self.mouse_x = 0  # Current mouse X position
        self.mouse_y = 0  # Current mouse Y position
        self.mouse_button_down = False  # Track if left mouse button is down
        self.stats_card_rect = None
        self.stats_long_press_duration_sec = 5.0
        self._stats_long_press_active = False
        self._stats_long_press_started_at = 0.0
        self._stats_long_press_fired = False
        self.historic_auto_refresh_interval = 2.0
        self._last_historic_auto_refresh = 0.0
        self._background_cache = None
        self._background_cache_mtime = None
        self._background_cache_size = (self.width, self.height)
        self._canvas_buffer = None
        self._image_cache = OrderedDict()
        self._image_cache_max_items = 32
        self._db_result_cache = {}
        self._piece_identifier_cache = {}
        self._model_overlay_cache_key = None
        self._model_overlay_cache_value = {}
        self._model_overlay_cache_time = 0.0
        self._model_overlay_cache_ttl = 1.0
        self._db_registered_images = set()
        self._historic_index_cache = None
        self._historic_index_mtime = None
        self._historic_index_last_scan = 0.0
        self.historic_index_rescan_interval = 1.5
        self._historic_jsn_cache = []
        self.toast_message = ""
        self.toast_message_is_error = False
        self.toast_message_time = 0.0
        self.toast_message_duration_sec = 1.8
        self.set_sftp_client(sftp_client)

    def set_db_connection(self, db_client):
        """Set active DB client and release any DB blocking overlay."""
        self.db = db_client
        self.db_blocking = False
        self.db_block_message = ""
        self.db_block_detail = ""

    def set_db_blocked(self, message):
        """Enable blocking overlay while waiting for DB connectivity."""
        self.db = None
        self.db_blocking = True
        self.db_block_message = (
            message
            or "PostgreSQL is disconnected. Start postgres and wait for automatic reconnect."
        )

    def set_sftp_client(self, sftp_client):
        """Update active SFTP client."""
        self.sftp_client = sftp_client

    def set_controller(self, controller):
        self.controller = controller
        if controller is not None:
            handler = getattr(controller, "handle_ui_action", None)
            if callable(handler):
                self.action_handler = handler

    def set_action_handler(self, action_handler):
        self.action_handler = action_handler

    def _emit_action(self, action, **payload):
        if not callable(self.action_handler):
            return
        try:
            self.action_handler(action, **payload)
        except Exception as exc:
            print(f"Error dispatching UI action '{action}': {exc}")

    def _clear_sync_message(self):
        self.sync_message = ""
        self.sync_message_is_error = False
        self.sync_message_time = 0
        self.sync_message_auto_dismiss_sec = None
        self.sync_message_close_button_rect = None

    def _require_controller(self):
        if self.controller is not None:
            return self.controller
        from main_controller import MainController

        self.controller = MainController(
            display=self,
            sftp_credentials=self.sftp_credentials,
        )
        return self.controller

    def _extract_camera_label(self, img_path):
        """Extract camera label (Cam_1..Cam_7) from filename if present."""
        filename = self.file_manager.basename(img_path).lower()
        match = re.search(r"cam(?:cam)?[_-]*([1-9])", filename)
        if match:
            return f"Cam_{match.group(1)}"
        return None

    def _draw_camera_label(self, canvas, x, y, img_size, label_text):
        """Draw camera label above the image (not inside)."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]

        padding_y = 8
        label_height = text_size[1] + padding_y * 2
        label_height = max(label_height, 30)

        # Place label above the image with a small gap, shifted slightly down
        gap = 0
        offset_down = 2
        label_y2 = max(0, y - gap) + offset_down
        label_y1 = max(0, label_y2 - label_height)

        text_x = x + (img_size - text_size[0]) // 2
        text_y = label_y1 + padding_y + text_size[1]
        cv2.putText(canvas, label_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    def _get_current_historic_jsn(self):
        """Return the JSN for the visible historic batch, if any."""
        if not self.historic_images:
            return None
        if self.historic_offset < 0 or self.historic_offset >= len(self.historic_images):
            return None

        current_batch = self.historic_images[self.historic_offset]
        if not current_batch:
            return None

        first_image = current_batch[0]
        return first_image.split("_")[0] if "_" in first_image else first_image

    def _get_current_historic_piece_number(self):
        """Return the visible historic piece number using the UI numbering."""
        total_pieces = len(self.historic_images)
        if total_pieces <= 0:
            return 0

        current_piece = total_pieces - int(self.historic_offset or 0)
        if current_piece < 1:
            return 1
        if current_piece > total_pieces:
            return total_pieces
        return current_piece

    def _set_toast_message(self, message, is_error=False, duration_sec=1.8):
        """Show a short non-blocking toast message."""
        self.toast_message = str(message or "").strip()
        self.toast_message_is_error = bool(is_error)
        self.toast_message_time = time.time()
        self.toast_message_duration_sec = max(0.5, float(duration_sec))

    def _copy_text_to_clipboard(self, text):
        """Copy text to the system clipboard with a Windows-first fallback."""
        clipboard_text = str(text or "")
        if not clipboard_text:
            return False

        if os.name == "nt":
            try:
                subprocess.run(
                    ["clip"],
                    input=clipboard_text,
                    text=True,
                    check=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                return True
            except Exception:
                pass

        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            root.clipboard_clear()
            root.clipboard_append(clipboard_text)
            root.update()
            root.destroy()
            return True
        except Exception:
            return False

    def _read_text_from_clipboard(self):
        """Read plain text from the system clipboard."""
        if os.name == "nt":
            try:
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
                    capture_output=True,
                    text=True,
                    check=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                return result.stdout.rstrip("\r\n")
            except Exception:
                pass

        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            clipboard_text = root.clipboard_get()
            root.destroy()
            return str(clipboard_text)
        except Exception:
            return None

    def _copy_current_historic_jsn(self):
        """Copy the current historic JSN and show operator feedback."""
        jsn = self._get_current_historic_jsn()
        if not jsn:
            self._set_toast_message("No JSN available to copy", is_error=True)
            return False

        copied = self._copy_text_to_clipboard(jsn)
        if copied:
            self._set_toast_message(f"Copied JSN {jsn}", is_error=False)
        else:
            self._set_toast_message("Unable to copy JSN to clipboard", is_error=True)
        return copied

    def _choose_import_package_path(self):
        """Open a folder picker and return the selected export folder path."""
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.update_idletasks()
            package_path = filedialog.askdirectory(
                title="Select Display State Folder",
            )
            root.destroy()
            return package_path or None
        except Exception as exc:
            self._set_toast_message(f"Unable to open import dialog: {exc}", is_error=True)
            return None

    def _get_historic_image_report_filter_options(self):
        """Return drawable defect/angle combinations available in model_results."""
        fallback = [{"angle": "side", "class_name": "wrinkle"}]
        if not self.db:
            return fallback

        try:
            rows = self.db.fetch(
                """
                SELECT DISTINCT angle, class_name
                FROM (
                    SELECT
                        CASE
                            WHEN LOWER(img_name) LIKE '%side%' THEN 'side'
                            WHEN LOWER(img_name) LIKE '%diag%' THEN 'diag'
                            WHEN LOWER(img_name) LIKE '%front%' THEN 'front'
                        END AS angle,
                        LOWER(TRIM(class_name)) AS class_name
                    FROM model_results
                    WHERE coordinates IS NOT NULL
                ) available_filters
                WHERE angle IS NOT NULL
                  AND class_name IS NOT NULL
                  AND class_name <> ''
                  AND class_name <> 'ok'
                ORDER BY angle, class_name
                """
            )
        except Exception as exc:
            print(f"Error loading image report filters: {exc}")
            return fallback

        options = []
        seen = set()
        for row in rows or []:
            if not hasattr(row, "get"):
                continue
            angle = str(row.get("angle") or "").strip().lower()
            class_name = str(row.get("class_name") or "").strip().lower()
            key = (angle, class_name)
            if not angle or not class_name or key in seen:
                continue
            seen.add(key)
            options.append({"angle": angle, "class_name": class_name})

        classes_by_single_angle = {}
        for option in options:
            classes_by_single_angle.setdefault(option["angle"], set()).add(
                option["class_name"]
            )
        combined_classes = sorted(
            classes_by_single_angle.get("side", set())
            & classes_by_single_angle.get("diag", set())
        )
        options.extend(
            {"angle": "side+diag", "class_name": class_name}
            for class_name in combined_classes
        )

        angle_order = {"side": 0, "diag": 1, "side+diag": 2, "front": 3}
        options.sort(
            key=lambda item: (
                angle_order.get(item["angle"], 99),
                item["angle"],
                item["class_name"],
            )
        )
        return options or fallback

    def _close_historic_image_report_dialog(self):
        root = self._image_report_dialog_root
        self._image_report_dialog_root = None
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass

    def _pump_historic_image_report_dialog(self):
        """Process Tk events without blocking the OpenCV display loop."""
        root = self._image_report_dialog_root
        if root is not None:
            try:
                root.update_idletasks()
                root.update()
            except Exception:
                self._image_report_dialog_root = None

        payload = self._image_report_dialog_result
        if payload:
            self._image_report_dialog_result = None
            action = payload.pop("_action", "export_historic_image_report")
            self._emit_action(action, **payload)

        analysis_request = self._verdict_analysis_dialog_request
        if analysis_request:
            self._verdict_analysis_dialog_request = None
            self._open_historic_verdict_analysis_dialog(**analysis_request)

        analysis_root = self._verdict_analysis_dialog_root
        if analysis_root is not None:
            try:
                analysis_root.update_idletasks()
                analysis_root.update()
            except Exception:
                self._clear_historic_verdict_analysis_state(destroy_root=False)

    def _open_historic_image_report_dialog(self):
        """Open a visible non-modal report selector and return immediately."""
        existing_root = self._image_report_dialog_root
        if existing_root is not None:
            try:
                existing_root.deiconify()
                existing_root.lift()
                existing_root.focus_force()
            except Exception:
                self._image_report_dialog_root = None
            else:
                return True

        try:
            import tkinter as tk
            from tkinter import ttk

            options = self._get_historic_image_report_filter_options()
            classes_by_angle = {}
            for option in options:
                classes_by_angle.setdefault(option["angle"], []).append(
                    option["class_name"]
                )

            angles = list(classes_by_angle)
            default_angle = "side" if "side" in classes_by_angle else angles[0]
            default_classes = classes_by_angle[default_angle]
            default_class = (
                "wrinkle" if "wrinkle" in default_classes else default_classes[0]
            )

            root = tk.Tk()
            self._image_report_dialog_root = root
            root.title("Image Report")
            root.resizable(False, False)

            angle_labels = {
                angle: (
                    "SIDE + DIAG"
                    if angle == "side+diag"
                    else angle.upper()
                )
                for angle in angles
            }
            angle_values_by_label = {
                label: angle for angle, label in angle_labels.items()
            }
            endform_var = tk.StringVar(master=root)
            angle_var = tk.StringVar(
                master=root,
                value=angle_labels[default_angle],
            )
            class_var = tk.StringVar(master=root, value=default_class)

            ttk.Label(root, text="Endform type:").grid(
                row=0,
                column=0,
                padx=12,
                pady=(12, 6),
                sticky="w",
            )
            endform_entry = ttk.Entry(root, textvariable=endform_var, width=32)
            endform_entry.grid(
                row=0,
                column=1,
                padx=12,
                pady=(12, 6),
                sticky="ew",
            )

            ttk.Label(root, text="Angle:").grid(
                row=1,
                column=0,
                padx=12,
                pady=6,
                sticky="w",
            )
            angle_combo = ttk.Combobox(
                root,
                textvariable=angle_var,
                values=[angle_labels[angle] for angle in angles],
                state="readonly",
                width=29,
            )
            angle_combo.grid(row=1, column=1, padx=12, pady=6, sticky="ew")

            ttk.Label(root, text="Defect class:").grid(
                row=2,
                column=0,
                padx=12,
                pady=6,
                sticky="w",
            )
            class_combo = ttk.Combobox(
                root,
                textvariable=class_var,
                values=default_classes,
                state="readonly",
                width=29,
            )
            class_combo.grid(row=2, column=1, padx=12, pady=6, sticky="ew")

            def refresh_class_options(_event=None):
                selected_angle = angle_values_by_label.get(angle_var.get(), "")
                available_classes = classes_by_angle.get(selected_angle, [])
                class_combo.configure(values=available_classes)
                if class_var.get() not in available_classes:
                    class_var.set(
                        "wrinkle"
                        if "wrinkle" in available_classes
                        else (available_classes[0] if available_classes else "")
                    )

            def submit_dialog(action="export_historic_image_report"):
                endform_type = endform_var.get().strip()
                angle = angle_values_by_label.get(angle_var.get(), "")
                defect_class = class_var.get().strip().lower()
                if not endform_type or not angle or not defect_class:
                    self._set_toast_message(
                        "Endform type, angle, and defect class are required",
                        is_error=True,
                    )
                    return
                self._image_report_dialog_result = {
                    "endform_type": endform_type,
                    "class_name": defect_class,
                    "defect_class": defect_class,
                    "angle": angle,
                }
                if action != "export_historic_image_report":
                    self._image_report_dialog_result["_action"] = action
                self._close_historic_image_report_dialog()

            angle_combo.bind("<<ComboboxSelected>>", refresh_class_options)
            button_frame = ttk.Frame(root)
            button_frame.grid(
                row=3,
                column=0,
                columnspan=2,
                padx=12,
                pady=(8, 12),
                sticky="e",
            )
            ttk.Button(
                button_frame,
                text="Cancel",
                command=self._close_historic_image_report_dialog,
            ).pack(side="right", padx=(6, 0))
            ttk.Button(
                button_frame,
                text="Export Excel",
                command=submit_dialog,
            ).pack(
                side="right",
            )
            ttk.Button(
                button_frame,
                text="Open Analysis",
                command=lambda: submit_dialog("open_historic_verdict_analysis"),
            ).pack(side="right", padx=(0, 6))

            root.protocol(
                "WM_DELETE_WINDOW",
                self._close_historic_image_report_dialog,
            )
            root.bind("<Return>", lambda _event: submit_dialog())
            root.bind(
                "<Escape>",
                lambda _event: self._close_historic_image_report_dialog(),
            )
            root.update_idletasks()
            try:
                root.attributes("-topmost", True)
            except Exception:
                pass
            root.deiconify()
            root.lift()
            endform_entry.focus_force()
            return True
        except Exception as exc:
            self._close_historic_image_report_dialog()
            self._set_toast_message(f"Unable to open report dialog: {exc}", is_error=True)
            return False

    def queue_historic_verdict_analysis(self, rows, filters=None):
        """Queue a completed analysis snapshot for creation on the UI thread."""
        self._verdict_analysis_dialog_request = {
            "rows": copy.deepcopy(list(rows or [])),
            "filters": dict(filters or {}),
        }

    def _clear_historic_verdict_analysis_state(self, destroy_root=True):
        root = self._verdict_analysis_dialog_root
        self._verdict_analysis_dialog_root = None
        if destroy_root and root is not None:
            try:
                root.destroy()
            except Exception:
                pass
        self._verdict_analysis_rows = []
        self._verdict_analysis_tree = None
        self._verdict_analysis_metric_vars = {}
        self._verdict_analysis_completion_var = None
        self._verdict_analysis_required_angles = ()
        self._verdict_analysis_confidence_thresholds = {}
        self._verdict_analysis_threshold_vars = {}
        self._verdict_analysis_threshold_summary_var = None
        self._verdict_analysis_optimization_var = None
        self._verdict_analysis_dirty = False

    def _close_historic_verdict_analysis_dialog(self, confirm=True):
        root = self._verdict_analysis_dialog_root
        if root is None:
            self._clear_historic_verdict_analysis_state(destroy_root=False)
            return True

        if confirm and self._verdict_analysis_dirty:
            try:
                from tkinter import messagebox

                should_close = messagebox.askyesno(
                    "Discard analysis",
                    "The captured actual verdicts are temporary. Discard them and close?",
                    parent=root,
                )
            except Exception:
                should_close = False
            if not should_close:
                return False

        self._clear_historic_verdict_analysis_state(destroy_root=True)
        return True

    def _format_verdict_analysis_position(self, entry):
        if not isinstance(entry, dict):
            return "-"
        jsn = str(entry.get("jsn") or "").strip()
        if not jsn:
            return "-"
        inferred = str(entry.get("inferred_result") or "").strip().upper()
        verdict = inferred if inferred in ("OK", "NOK") else "N/D"
        return f"{verdict}  |  {jsn}"

    def _set_historic_verdict_confidence_thresholds(
        self,
        thresholds,
        status_message="Manual thresholds.",
    ):
        from verdict_analysis import normalize_confidence_thresholds

        angles = self._verdict_analysis_required_angles or ("side", "diag")
        normalized = normalize_confidence_thresholds(thresholds, angles=angles)
        self._verdict_analysis_confidence_thresholds = normalized
        for angle, value in normalized.items():
            variable = self._verdict_analysis_threshold_vars.get(angle)
            if variable is not None:
                try:
                    variable.set(value)
                except Exception:
                    pass
        status_var = self._verdict_analysis_optimization_var
        if status_var is not None and status_message is not None:
            try:
                status_var.set(status_message)
            except Exception:
                pass
        self._refresh_historic_verdict_analysis()
        return normalized

    def _optimize_historic_verdict_confidence_thresholds(self):
        from verdict_analysis import optimize_confidence_thresholds

        angles = self._verdict_analysis_required_angles or ("side", "diag")
        result = optimize_confidence_thresholds(
            self._verdict_analysis_rows,
            required_angles=angles,
            false_negative_target=0.10,
            positions=4,
        )
        status = result.get("message") or "Confidence optimization completed."
        self._set_historic_verdict_confidence_thresholds(
            result["thresholds"],
            status_message=status,
        )
        return result

    def _refresh_historic_verdict_analysis(self):
        from verdict_analysis import (
            apply_confidence_thresholds,
            calculate_average_error_rates,
            calculate_position_metrics,
        )

        rows = self._verdict_analysis_rows
        required_angles = self._verdict_analysis_required_angles
        if required_angles:
            apply_confidence_thresholds(
                rows,
                self._verdict_analysis_confidence_thresholds,
                required_angles=required_angles,
            )
        tree = self._verdict_analysis_tree
        if tree is not None:
            for row_idx, row in enumerate(rows):
                item_id = f"row_{row_idx}"
                positions = list(row.get("positions") or [])
                position_values = [
                    self._format_verdict_analysis_position(
                        positions[position_idx]
                        if position_idx < len(positions)
                        else None
                    )
                    for position_idx in range(4)
                ]
                values = (
                    row.get("group_label") or f"Pieza agrupada #{row_idx + 1}",
                    str(row.get("actual_result") or "").strip().upper(),
                    *position_values,
                )
                try:
                    if tree.exists(item_id):
                        tree.item(item_id, values=values)
                    else:
                        tree.insert("", "end", iid=item_id, values=values)
                except Exception:
                    pass

        metrics = calculate_position_metrics(rows, positions=4)
        for position, position_metrics in metrics.items():
            for metric_key, value in position_metrics.items():
                variable = self._verdict_analysis_metric_vars.get(
                    (position, metric_key)
                )
                if variable is not None:
                    try:
                        variable.set(str(value))
                    except Exception:
                        pass

        threshold_summary_var = self._verdict_analysis_threshold_summary_var
        if threshold_summary_var is not None:
            rate_summary = calculate_average_error_rates(rows, positions=4)
            false_positive_rate = rate_summary["average_false_positive_rate"]
            false_negative_rate = rate_summary["average_false_negative_rate"]
            total_evaluated = rate_summary["total_evaluated"]
            fp_text = (
                f"{false_positive_rate:.2%} "
                f"({rate_summary['total_false_positive']}/{total_evaluated})"
                if false_positive_rate is not None
                else "N/A"
            )
            fn_text = (
                f"{false_negative_rate:.2%} "
                f"({rate_summary['total_false_negative']}/{total_evaluated})"
                if false_negative_rate is not None
                else "N/A"
            )
            target_text = (
                "Target met"
                if false_negative_rate is not None
                and false_negative_rate < 0.10
                else "Target not met"
            )
            threshold_text = "    ".join(
                f"{angle.upper()}: "
                f"{self._verdict_analysis_confidence_thresholds.get(angle, 0.0):.4f}"
                for angle in required_angles
            )
            try:
                threshold_summary_var.set(
                    f"{threshold_text}    Avg FP: {fp_text}    "
                    f"Avg FN: {fn_text}    {target_text} (FN < 10%)"
                )
            except Exception:
                pass

        completion_var = self._verdict_analysis_completion_var
        if completion_var is not None:
            completed = sum(
                1
                for row in rows
                if str(row.get("actual_result") or "").strip().upper()
                in ("OK", "NOK")
            )
            try:
                completion_var.set(f"Actual values: {completed}/{len(rows)}")
            except Exception:
                pass

    def _set_historic_verdict_actual(self, row_index, value):
        from verdict_analysis import normalize_verdict

        normalized = normalize_verdict(value, allow_blank=True)
        resolved_index = int(row_index)
        if resolved_index < 0 or resolved_index >= len(self._verdict_analysis_rows):
            raise IndexError("Verdict analysis row is outside the available range")

        row = self._verdict_analysis_rows[resolved_index]
        current = str(row.get("actual_result") or "").strip().upper()
        if current != normalized:
            row["actual_result"] = normalized
            self._verdict_analysis_dirty = True
            self._refresh_historic_verdict_analysis()
        return normalized

    def _apply_historic_verdict_paste(self, text, start_index=0):
        """Atomically apply a pasted column starting at the selected table row."""
        from verdict_analysis import parse_actual_verdict_values

        values = parse_actual_verdict_values(text)
        resolved_start = int(start_index or 0)
        if resolved_start < 0 or resolved_start > len(self._verdict_analysis_rows):
            raise ValueError("Paste start row is outside the available range")
        if resolved_start + len(values) > len(self._verdict_analysis_rows):
            available = max(0, len(self._verdict_analysis_rows) - resolved_start)
            raise ValueError(
                f"The pasted list has {len(values)} values but only {available} rows are available."
            )

        if not values:
            return 0

        for offset, value in enumerate(values):
            self._verdict_analysis_rows[resolved_start + offset][
                "actual_result"
            ] = value
        self._verdict_analysis_dirty = True
        self._refresh_historic_verdict_analysis()
        return len(values)

    def _open_historic_verdict_analysis_dialog(self, rows, filters=None):
        """Open the non-modal, spreadsheet-like verdict analysis window."""
        existing_root = self._verdict_analysis_dialog_root
        if existing_root is not None:
            try:
                existing_root.deiconify()
                existing_root.lift()
                existing_root.focus_force()
            except Exception:
                self._clear_historic_verdict_analysis_state(destroy_root=False)
            else:
                self._set_toast_message(
                    "Close the current verdict analysis before opening another one",
                    is_error=True,
                )
                return False

        try:
            import tkinter as tk
            from tkinter import messagebox, ttk

            analysis_rows = copy.deepcopy(list(rows or []))
            if not analysis_rows:
                raise ValueError("No grouped verdict rows are available")

            root = tk.Tk()
            self._verdict_analysis_dialog_root = root
            self._verdict_analysis_rows = analysis_rows
            self._verdict_analysis_dirty = False
            root.title("Verdict Analysis")
            root.geometry("1320x930")
            root.minsize(1040, 620)

            outer = ttk.Frame(root, padding=12)
            outer.pack(fill="both", expand=True)

            filter_data = dict(filters or {})
            endform_type = str(filter_data.get("endform_type") or "").strip()
            angle_value = str(filter_data.get("angle") or "").strip().lower()
            angle = " + ".join(
                part.strip().upper()
                for part in angle_value.split("+")
                if part.strip()
            )
            defect_class = str(
                filter_data.get("defect_class") or ""
            ).strip().upper()
            ttk.Label(
                outer,
                text="Verdict Analysis",
                font=("Segoe UI", 16, "bold"),
            ).pack(anchor="w")
            ttk.Label(
                outer,
                text=f"Endform: {endform_type}    Angle: {angle}    Defect: {defect_class}",
            ).pack(anchor="w", pady=(2, 10))

            required_angles = tuple(
                str(item or "").strip().lower()
                for item in (filter_data.get("required_angles") or [])
                if str(item or "").strip()
            )
            if angle_value == "side+diag" and required_angles == ("side", "diag"):
                from verdict_analysis import normalize_confidence_thresholds

                self._verdict_analysis_required_angles = required_angles
                self._verdict_analysis_confidence_thresholds = (
                    normalize_confidence_thresholds(
                        filter_data.get("confidence_thresholds") or {},
                        angles=required_angles,
                    )
                )

                confidence_frame = ttk.LabelFrame(
                    outer,
                    text="Global confidence thresholds (temporary)",
                    padding=8,
                )
                confidence_frame.pack(fill="x", pady=(0, 10))
                self._verdict_analysis_threshold_vars = {}

                def commit_threshold(selected_angle, raw_value=None):
                    try:
                        value = (
                            raw_value
                            if raw_value is not None
                            else self._verdict_analysis_threshold_vars[
                                selected_angle
                            ].get()
                        )
                        thresholds = dict(
                            self._verdict_analysis_confidence_thresholds
                        )
                        thresholds[selected_angle] = float(value)
                        self._set_historic_verdict_confidence_thresholds(thresholds)
                    except Exception as exc:
                        variable = self._verdict_analysis_threshold_vars.get(
                            selected_angle
                        )
                        if variable is not None:
                            variable.set(
                                self._verdict_analysis_confidence_thresholds.get(
                                    selected_angle,
                                    0.0,
                                )
                            )
                        show_error(exc)

                for column_idx, selected_angle in enumerate(required_angles):
                    value_var = tk.DoubleVar(
                        master=root,
                        value=self._verdict_analysis_confidence_thresholds.get(
                            selected_angle,
                            0.0,
                        ),
                    )
                    self._verdict_analysis_threshold_vars[selected_angle] = value_var
                    ttk.Label(
                        confidence_frame,
                        text=f"{selected_angle.upper()}:",
                        font=("Segoe UI", 9, "bold"),
                    ).grid(
                        row=column_idx,
                        column=0,
                        padx=(4, 8),
                        pady=3,
                        sticky="w",
                    )
                    ttk.Scale(
                        confidence_frame,
                        from_=0.0,
                        to=1.0,
                        variable=value_var,
                        command=lambda raw, selected_angle=selected_angle: commit_threshold(
                            selected_angle,
                            raw,
                        ),
                    ).grid(
                        row=column_idx,
                        column=1,
                        padx=4,
                        pady=3,
                        sticky="ew",
                    )
                    spinbox = ttk.Spinbox(
                        confidence_frame,
                        from_=0.0,
                        to=1.0,
                        increment=0.0001,
                        textvariable=value_var,
                        width=9,
                        format="%.4f",
                        command=lambda selected_angle=selected_angle: commit_threshold(
                            selected_angle
                        ),
                    )
                    spinbox.grid(
                        row=column_idx,
                        column=2,
                        padx=(8, 12),
                        pady=3,
                    )
                    spinbox.bind(
                        "<Return>",
                        lambda _event, selected_angle=selected_angle: commit_threshold(
                            selected_angle
                        ),
                    )
                    spinbox.bind(
                        "<FocusOut>",
                        lambda _event, selected_angle=selected_angle: commit_threshold(
                            selected_angle
                        ),
                    )

                confidence_frame.columnconfigure(1, weight=1)
                threshold_summary_var = tk.StringVar(master=root)
                self._verdict_analysis_threshold_summary_var = threshold_summary_var
                ttk.Label(
                    confidence_frame,
                    textvariable=threshold_summary_var,
                ).grid(
                    row=2,
                    column=0,
                    columnspan=3,
                    padx=4,
                    pady=(6, 2),
                    sticky="w",
                )
                optimization_var = tk.StringVar(
                    master=root,
                    value="Enter actual OK/NOK values, then find the best point.",
                )
                self._verdict_analysis_optimization_var = optimization_var
                ttk.Label(
                    confidence_frame,
                    textvariable=optimization_var,
                ).grid(
                    row=3,
                    column=0,
                    columnspan=2,
                    padx=4,
                    pady=(2, 4),
                    sticky="w",
                )

                def find_best_point():
                    try:
                        self._optimize_historic_verdict_confidence_thresholds()
                    except Exception as exc:
                        show_error(exc)

                def reset_thresholds():
                    self._set_historic_verdict_confidence_thresholds(
                        {selected_angle: 0.0 for selected_angle in required_angles},
                        status_message="Thresholds reset to 0.0000.",
                    )

                button_box = ttk.Frame(confidence_frame)
                button_box.grid(
                    row=3,
                    column=2,
                    padx=(8, 4),
                    pady=(2, 4),
                    sticky="e",
                )
                ttk.Button(
                    button_box,
                    text="Reset thresholds",
                    command=reset_thresholds,
                ).pack(side="right")
                ttk.Button(
                    button_box,
                    text="Find best point",
                    command=find_best_point,
                ).pack(side="right", padx=(0, 6))

            metrics_frame = ttk.LabelFrame(
                outer,
                text="Immediate comparison by report position",
                padding=8,
            )
            metrics_frame.pack(fill="x", pady=(0, 10))
            metric_rows = (
                ("true_ok", "True OK"),
                ("true_nok", "True NOK"),
                ("false_negative", "False Negative"),
                ("false_positive", "False Positive"),
                ("evaluated", "Evaluated"),
            )
            ttk.Label(metrics_frame, text="Metric", font=("Segoe UI", 9, "bold")).grid(
                row=0, column=0, padx=8, pady=3, sticky="w"
            )
            for position in range(1, 5):
                ttk.Label(
                    metrics_frame,
                    text=f"Position {position}",
                    font=("Segoe UI", 9, "bold"),
                ).grid(row=0, column=position, padx=22, pady=3)
                metrics_frame.columnconfigure(position, weight=1)
            self._verdict_analysis_metric_vars = {}
            for metric_row_idx, (metric_key, metric_label) in enumerate(
                metric_rows,
                start=1,
            ):
                ttk.Label(metrics_frame, text=metric_label).grid(
                    row=metric_row_idx,
                    column=0,
                    padx=8,
                    pady=2,
                    sticky="w",
                )
                for position in range(1, 5):
                    metric_var = tk.StringVar(master=root, value="0")
                    self._verdict_analysis_metric_vars[(position, metric_key)] = (
                        metric_var
                    )
                    ttk.Label(
                        metrics_frame,
                        textvariable=metric_var,
                        anchor="center",
                    ).grid(
                        row=metric_row_idx,
                        column=position,
                        padx=22,
                        pady=2,
                        sticky="ew",
                    )

            toolbar = ttk.Frame(outer)
            toolbar.pack(fill="x", pady=(0, 8))
            completion_var = tk.StringVar(master=root)
            self._verdict_analysis_completion_var = completion_var
            ttk.Label(toolbar, textvariable=completion_var).pack(side="left")

            table_frame = ttk.Frame(outer)
            table_frame.pack(fill="both", expand=True)
            columns = ("group", "actual", "position_1", "position_2", "position_3", "position_4")
            tree = ttk.Treeview(
                table_frame,
                columns=columns,
                show="headings",
                selectmode="browse",
            )
            self._verdict_analysis_tree = tree
            tree.heading("group", text="Group")
            tree.heading("actual", text="Actual value")
            for position in range(1, 5):
                tree.heading(f"position_{position}", text=f"Position {position} inferred")
            tree.column("group", width=160, minwidth=130, anchor="w", stretch=False)
            tree.column("actual", width=105, minwidth=90, anchor="center", stretch=False)
            for position in range(1, 5):
                tree.column(
                    f"position_{position}",
                    width=245,
                    minwidth=150,
                    anchor="center",
                )
            scrollbar = ttk.Scrollbar(
                table_frame,
                orient="vertical",
                command=tree.yview,
            )
            tree.configure(yscrollcommand=scrollbar.set)
            tree.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            def selected_row_index(default=0):
                selection = tree.selection()
                if not selection:
                    return default
                item_id = str(selection[0])
                if item_id.startswith("row_"):
                    try:
                        return int(item_id.split("_", 1)[1])
                    except (TypeError, ValueError):
                        pass
                return default

            def show_error(message):
                messagebox.showerror("Verdict Analysis", str(message), parent=root)

            def paste_values(_event=None):
                try:
                    clipboard_text = root.clipboard_get()
                    self._apply_historic_verdict_paste(
                        clipboard_text,
                        start_index=selected_row_index(),
                    )
                except Exception as exc:
                    show_error(exc)
                return "break"

            def clear_values():
                had_values = any(
                    str(row.get("actual_result") or "").strip()
                    for row in self._verdict_analysis_rows
                )
                for row in self._verdict_analysis_rows:
                    row["actual_result"] = ""
                if had_values:
                    self._verdict_analysis_dirty = True
                self._refresh_historic_verdict_analysis()

            def set_selected_value(value):
                try:
                    self._set_historic_verdict_actual(selected_row_index(), value)
                except Exception as exc:
                    show_error(exc)

            def handle_tree_key(event):
                key = str(getattr(event, "keysym", "") or "").lower()
                if key == "o":
                    set_selected_value("OK")
                    return "break"
                if key == "n":
                    set_selected_value("NOK")
                    return "break"
                if key in ("delete", "backspace"):
                    set_selected_value("")
                    return "break"
                return None

            def edit_actual_cell(event):
                if tree.identify_region(event.x, event.y) != "cell":
                    return
                if tree.identify_column(event.x) != "#2":
                    return
                item_id = tree.identify_row(event.y)
                if not item_id:
                    return
                try:
                    row_index = int(item_id.split("_", 1)[1])
                except (IndexError, ValueError):
                    return
                bbox = tree.bbox(item_id, "actual")
                if not bbox:
                    return
                x, y, width, height = bbox
                editor_var = tk.StringVar(
                    master=root,
                    value=str(
                        self._verdict_analysis_rows[row_index].get("actual_result")
                        or ""
                    ),
                )
                editor = ttk.Combobox(
                    tree,
                    textvariable=editor_var,
                    values=("", "OK", "NOK"),
                    state="readonly",
                )
                editor.place(x=x, y=y, width=width, height=height)

                committed = {"done": False}

                def finish(commit=True):
                    if committed["done"]:
                        return
                    committed["done"] = True
                    if commit:
                        try:
                            self._set_historic_verdict_actual(
                                row_index,
                                editor_var.get(),
                            )
                        except Exception as exc:
                            show_error(exc)
                    editor.destroy()
                    tree.focus_set()

                editor.bind("<<ComboboxSelected>>", lambda _event: finish(True))
                editor.bind("<Return>", lambda _event: finish(True))
                editor.bind("<Escape>", lambda _event: finish(False))
                editor.bind("<FocusOut>", lambda _event: finish(True))
                editor.focus_set()

            ttk.Button(toolbar, text="Paste values", command=paste_values).pack(
                side="right"
            )
            ttk.Button(toolbar, text="Clear", command=clear_values).pack(
                side="right", padx=(0, 6)
            )
            ttk.Button(
                toolbar,
                text="Close",
                command=self._close_historic_verdict_analysis_dialog,
            ).pack(side="right", padx=(0, 6))

            tree.bind("<Control-v>", paste_values)
            tree.bind("<Control-V>", paste_values)
            tree.bind("<Key>", handle_tree_key)
            tree.bind("<Double-1>", edit_actual_cell)
            root.protocol("WM_DELETE_WINDOW", self._close_historic_verdict_analysis_dialog)
            root.bind(
                "<Escape>",
                lambda _event: self._close_historic_verdict_analysis_dialog(),
            )

            self._refresh_historic_verdict_analysis()
            if self._verdict_analysis_rows:
                tree.selection_set("row_0")
                tree.focus("row_0")
            root.update_idletasks()
            root.deiconify()
            root.lift()
            tree.focus_set()
            return True
        except Exception as exc:
            self._clear_historic_verdict_analysis_state(destroy_root=True)
            self._set_toast_message(
                f"Unable to open verdict analysis: {exc}",
                is_error=True,
            )
            return False

    def _copy_stats_modal_jsn(self, jsn):
        """Copy a JSN from the stats modal detail view and show operator feedback."""
        jsn_text = str(jsn or "").strip()
        if not jsn_text:
            self._set_toast_message("No JSN available to copy", is_error=True)
            return False

        copied = self._copy_text_to_clipboard(jsn_text)
        if copied:
            self._set_toast_message(f"Copied JSN {jsn_text}", is_error=False)
        else:
            self._set_toast_message("Unable to copy JSN to clipboard", is_error=True)
        return copied

    def _reset_historic_filter_state(self):
        """Clear the in-memory historic subset selected from stats."""
        self.historic_filter_kind = ""
        self.historic_filter_label = ""
        self.historic_filter_jsns = []
        self.historic_filter_total_count = 0

    def _reset_stats_class_modal_state(self):
        """Reset the stats modal drill-down state."""
        self.stats_class_modal_view = "summary"
        self.stats_class_modal_selected_kind = ""
        self.stats_class_modal_selected_label = ""
        self.stats_class_modal_detail_rows = []
        self.stats_class_modal_detail_offset = 0
        self.stats_class_modal_detail_visible_rows = 1
        self.stats_class_modal_matrix_offset = 0
        self.stats_class_modal_matrix_visible_rows = 1
        self.stats_class_modal_class_row_rects = []
        self.stats_class_modal_status_row_rects = []
        self.stats_class_modal_jsn_row_rects = []
        self.stats_class_modal_copy_rects = []
        self.stats_class_modal_back_rect = None
        self.stats_class_modal_summary_tab_rect = None
        self.stats_class_modal_matrix_tab_rect = None
        self.stats_class_modal_matrix_report_rect = None
        self.stats_class_modal_dataset_tab_rect = None
        self.stats_class_modal_list_rect = None
        self.stats_class_modal_scrollbar_rect = None
        self.stats_class_modal_dataset_result_options = []
        self.stats_class_modal_dataset_angle_options = []
        self.stats_class_modal_dataset_class_options = []
        self.stats_class_modal_dataset_selected_results = set()
        self.stats_class_modal_dataset_selected_angles = set()
        self.stats_class_modal_dataset_selected_classes = set()
        self.stats_class_modal_dataset_result_rects = []
        self.stats_class_modal_dataset_angle_rects = []
        self.stats_class_modal_dataset_class_rects = []
        self.stats_class_modal_dataset_export_rect = None
        self.stats_class_modal_dataset_class_offset = 0
        self.stats_class_modal_dataset_class_visible_rows = 1

    def _clamp_stats_class_modal_detail_offset(self):
        """Keep the detail-list offset within its valid range."""
        total_rows = len(self.stats_class_modal_detail_rows or [])
        visible_rows = max(1, int(getattr(self, "stats_class_modal_detail_visible_rows", 1) or 1))
        max_offset = max(0, total_rows - visible_rows)
        self.stats_class_modal_detail_offset = max(
            0,
            min(int(self.stats_class_modal_detail_offset or 0), max_offset),
        )
        return self.stats_class_modal_detail_offset

    def _clamp_stats_class_modal_matrix_offset(self):
        """Keep the matrix-list offset within its valid range."""
        rows = list(self.stats_class_modal_matrix_rows or [])
        if rows and rows[-1].get("is_total"):
            rows = rows[:-1]
        total_rows = len(rows)
        visible_rows = max(1, int(getattr(self, "stats_class_modal_matrix_visible_rows", 1) or 1))
        max_offset = max(0, total_rows - visible_rows)
        self.stats_class_modal_matrix_offset = max(
            0,
            min(int(self.stats_class_modal_matrix_offset or 0), max_offset),
        )
        return self.stats_class_modal_matrix_offset

    def _clamp_stats_class_modal_dataset_class_offset(self):
        """Keep the dataset class-grid offset within its valid range."""
        class_options = list(self.stats_class_modal_dataset_class_options or [])
        total_rows = (len(class_options) + 1) // 2
        visible_rows = max(
            1,
            int(getattr(self, "stats_class_modal_dataset_class_visible_rows", 1) or 1),
        )
        max_offset = max(0, total_rows - visible_rows)
        self.stats_class_modal_dataset_class_offset = max(
            0,
            min(int(self.stats_class_modal_dataset_class_offset or 0), max_offset),
        )
        return self.stats_class_modal_dataset_class_offset

    def _paste_clipboard_into_search(self):
        """Paste numeric clipboard content into the historic JSN search field."""
        clipboard_text = self._read_text_from_clipboard()
        if clipboard_text is None:
            self._set_toast_message("Unable to read clipboard", is_error=True)
            return False

        pasted_jsn = "".join(ch for ch in str(clipboard_text) if ch.isdigit())[:21]
        if not pasted_jsn:
            self._set_toast_message("Clipboard has no JSN digits", is_error=True)
            return False

        self._emit_action("search_paste", text=pasted_jsn)
        self._set_toast_message(f"Pasted JSN {pasted_jsn}", is_error=False, duration_sec=1.4)
        return True
    
    def _get_piece_date(self):
        """Delegate piece date resolution to controller business logic."""
        try:
            return self._require_controller().get_piece_date()
        except Exception as e:
            print(f"Error getting piece date: {e}")
            return "N/A"

    def _get_piece_result_counts(self, jsns=None):
        """Return OK/NOK/FOK/FNOK counts from DB piece_result."""
        counts = {"OK": 0, "NOK": 0, "FOK": 0, "FNOK": 0}
        if self.db is None:
            return counts

        try:
            rows = self.db.fetch(
                "SELECT final_result, COUNT(*) AS cnt FROM piece_result GROUP BY final_result",
            )
            for row in rows:
                final_result = row.get("final_result")
                cnt = int(row.get("cnt", 0))
                if final_result in counts:
                    counts[final_result] = cnt
        except Exception:
            pass
        return counts

    def _get_piece_status_from_batch(self, batch_images):
        """Fallback: derive OK/NOK from file naming when DB has no record yet."""
        if not batch_images:
            return "OK"
        for img in batch_images:
            base = os.path.splitext(img)[0]
            if base.endswith("_NOK"):
                return "NOK"
        return "OK"

    def _get_stats_result_color(self, result_label):
        """Return a stable BGR color for each final result row."""
        palette = {
            "OK": (34, 100, 34),
            "NOK": (25, 25, 160),
            "FNOK": (100, 30, 130),
            "FOK": (10, 110, 200),
        }
        return palette.get(str(result_label or "").upper(), (90, 90, 90))

    def _draw_stats_card(self, canvas, x, y, size, ok_count, nok_count, fok_count, fnok_count):
        """Draw a stats card that fills the entire tile slot."""
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Background (dark charcoal) + border
        cv2.rectangle(canvas, (x, y), (x + size, y + size), (45, 45, 45), -1)
        cv2.rectangle(canvas, (x, y), (x + size, y + size), (20, 20, 20), 2)

        # Title area: top 16% of tile
        title = "PIECE STATS"
        title_area_h = int(size * 0.16)
        title_scale = size / 360 * 0.9
        title_thick = 2
        title_sz = cv2.getTextSize(title, font, title_scale, title_thick)[0]
        title_x = x + (size - title_sz[0]) // 2
        title_y = y + (title_area_h + title_sz[1]) // 2
        cv2.putText(canvas, title, (title_x, title_y), font, title_scale, (220, 220, 220), title_thick)

        # Divider below title
        div_y = y + title_area_h + 2
        cv2.line(canvas, (x + 6, div_y), (x + size - 6, div_y), (100, 100, 100), 1)

        # Stat rows fill the remaining height
        stats = [
            ("OK", ok_count, self._get_stats_result_color("OK")),
            ("NOK", nok_count, self._get_stats_result_color("NOK")),
            ("FNOK", fnok_count, self._get_stats_result_color("FNOK")),
            ("FOK", fok_count, self._get_stats_result_color("FOK")),
        ]
        remaining_h = size - title_area_h - 4
        row_h = remaining_h // len(stats)
        gap = 4
        row_font_scale = size / 360 * 1.15
        row_thick = 2

        for idx, (label, value, color) in enumerate(stats):
            ry = y + title_area_h + 4 + idx * row_h
            # Colored row background
            cv2.rectangle(canvas, (x + gap, ry), (x + size - gap, ry + row_h - gap), color, -1)
            text = f"{label}:  {value}"
            t_sz = cv2.getTextSize(text, font, row_font_scale, row_thick)[0]
            tx = x + (size - t_sz[0]) // 2
            ty = ry + (row_h - gap + t_sz[1]) // 2
            cv2.putText(canvas, text, (tx, ty), font, row_font_scale, (255, 255, 255), row_thick)

    def create_white_display(self):
        """Create a white display"""
        self.image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
    def _is_point_in_rect(self, x, y, rect):
        """Check if point (x, y) is inside rectangle rect (x, y, width, height)"""
        if rect is None:
            return False
        bx, by, bw, bh = rect
        return bx <= x <= bx + bw and by <= y <= by + bh
    
    def _scale_rect(self, rect, scale_factor):
        """Scale a rectangle (x, y, w, h) by a factor from its center"""
        if rect is None:
            return None
        x, y, w, h = rect
        center_x = x + w / 2
        center_y = y + h / 2
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        new_x = int(center_x - new_w / 2)
        new_y = int(center_y - new_h / 2)
        return (new_x, new_y, new_w, new_h)

    def _truncate_text_to_width(self, text, font, font_scale, thickness, max_width):
        """Trim long text so it fits in the available pixel width."""
        display_text = str(text or "")
        if max_width <= 0:
            return ""
        if cv2.getTextSize(display_text, font, font_scale, thickness)[0][0] <= max_width:
            return display_text

        ellipsis = "..."
        truncated = display_text
        while truncated:
            candidate = truncated + ellipsis
            if cv2.getTextSize(candidate, font, font_scale, thickness)[0][0] <= max_width:
                return candidate
            truncated = truncated[:-1]
        return ellipsis

    def _wrap_text_to_width(self, text, font, font_scale, thickness, max_width, max_lines=None):
        """Wrap text into lines that fit the available pixel width."""
        display_text = str(text or "").strip()
        if not display_text:
            return []

        words = display_text.split()
        if not words:
            return [display_text]

        lines = []
        current_line = ""

        for word in words:
            candidate = f"{current_line} {word}".strip()
            candidate_width = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
            if candidate_width <= max_width:
                current_line = candidate
                continue

            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                lines.append(self._truncate_text_to_width(word, font, font_scale, thickness, max_width))
                current_line = ""

        if current_line:
            if cv2.getTextSize(current_line, font, font_scale, thickness)[0][0] > max_width:
                current_line = self._truncate_text_to_width(current_line, font, font_scale, thickness, max_width)
            lines.append(current_line)

        if max_lines is not None and len(lines) > max_lines:
            overflow_text = " ".join(lines[max_lines - 1:])
            lines = lines[: max_lines - 1] + [
                self._truncate_text_to_width(overflow_text, font, font_scale, thickness, max_width)
            ]

        return lines

    def _prepare_wrapped_text_block(
        self,
        paragraphs,
        font,
        font_scale,
        thickness,
        max_width,
        line_spacing=40,
        max_lines_per_paragraph=None,
    ):
        """Wrap one or more paragraphs and return metrics for dynamic dialog layout."""
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]

        lines = []
        for paragraph in paragraphs or []:
            paragraph_text = str(paragraph or "").strip()
            if not paragraph_text:
                continue
            wrapped = self._wrap_text_to_width(
                paragraph_text,
                font,
                font_scale,
                thickness,
                max_width,
                max_lines=max_lines_per_paragraph,
            )
            if wrapped:
                lines.extend(wrapped)

        line_height = cv2.getTextSize("Ag", font, font_scale, thickness)[0][1]
        max_line_width = 0
        for line in lines:
            max_line_width = max(max_line_width, cv2.getTextSize(line, font, font_scale, thickness)[0][0])

        if not lines:
            text_height = 0
        else:
            text_height = line_height + ((len(lines) - 1) * line_spacing)

        return {
            "lines": lines,
            "line_height": line_height,
            "line_spacing": line_spacing,
            "text_width": max_line_width,
            "text_height": text_height,
        }

    def _draw_text_lines(self, canvas, lines, x, first_baseline_y, font, font_scale, color, thickness, line_spacing=40):
        """Draw a list of lines using a stable baseline spacing."""
        for idx, line in enumerate(lines or []):
            cv2.putText(
                canvas,
                line,
                (x, first_baseline_y + (idx * line_spacing)),
                font,
                font_scale,
                color,
                thickness,
            )
        return canvas

    def _draw_modal_frame(
        self,
        canvas,
        dialog_width,
        dialog_height,
        overlay_alpha=0.5,
        fill_color=(240, 240, 240),
        border_color=(0, 0, 0),
        border_width=3,
    ):
        """Draw a centered modal frame and return its position."""
        dialog_x = (self.width - dialog_width) // 2
        dialog_y = (self.height - dialog_height) // 2

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, overlay_alpha, canvas, 1.0 - overlay_alpha, 0, canvas)

        cv2.rectangle(
            canvas,
            (dialog_x, dialog_y),
            (dialog_x + dialog_width, dialog_y + dialog_height),
            fill_color,
            -1,
        )
        cv2.rectangle(
            canvas,
            (dialog_x, dialog_y),
            (dialog_x + dialog_width, dialog_y + dialog_height),
            border_color,
            border_width,
        )
        return canvas, dialog_x, dialog_y

    def _draw_modal_button(self, canvas, rect, label, fill_color, font_scale=0.7, thickness=2):
        """Draw a modal button with the shared hover/press behavior."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, rect)
        is_pressed = is_hovered and self.mouse_button_down
        scale_factor = 0.95 if is_pressed else (1.08 if is_hovered else 1.0)
        x_btn, y_btn, w_btn, h_btn = self._scale_rect(rect, scale_factor)

        cv2.rectangle(canvas, (x_btn, y_btn), (x_btn + w_btn, y_btn + h_btn), fill_color, -1)
        cv2.rectangle(canvas, (x_btn, y_btn), (x_btn + w_btn, y_btn + h_btn), (0, 0, 0), 2)

        label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        label_x = x_btn + (w_btn - label_size[0]) // 2
        label_y = y_btn + (h_btn + label_size[1]) // 2
        cv2.putText(canvas, label, (label_x, label_y), font, font_scale, (255, 255, 255), thickness)
        return canvas

    def _can_track_stats_long_press(self):
        return (
            self.stats_card_rect is not None
            and not self.db_blocking
            and not self.sync_in_progress
            and not self.reset_in_progress
            and not self.show_no_images_dialog
            and not self.show_piece_date_dialog
            and not self.show_stats_class_modal
            and not self.show_reset_confirm
            and not self.show_delete_confirm
            and not self.show_rebuild_confirm
        )

    def _cancel_stats_long_press(self):
        self._stats_long_press_active = False
        self._stats_long_press_started_at = 0.0
        self._stats_long_press_fired = False

    def _start_stats_long_press(self):
        self._stats_long_press_active = True
        self._stats_long_press_started_at = time.monotonic()
        self._stats_long_press_fired = False

    def _check_stats_long_press(self):
        if not self._stats_long_press_active:
            return False
        if (
            not self._can_track_stats_long_press()
            or not self.mouse_button_down
            or not self._is_point_in_rect(self.mouse_x, self.mouse_y, self.stats_card_rect)
        ):
            self._cancel_stats_long_press()
            return False
        if self._stats_long_press_fired:
            return False
        if (time.monotonic() - self._stats_long_press_started_at) < self.stats_long_press_duration_sec:
            return False

        self._stats_long_press_fired = True
        self._stats_long_press_active = False
        self._emit_action("open_rebuild_db_confirm")
        return True

    def draw_stats_class_modal(self, canvas):
        """Draw modal dialog showing piece breakdown by class and final result."""
        if self.stats_class_modal_view == "detail":
            dialog_width = 1440
            dialog_height = 650
        elif self.stats_class_modal_view == "dataset":
            dialog_width = 1260
            dialog_height = 690
        else:
            dialog_width = 1120
            dialog_height = 620
        dialog_x = (self.width - dialog_width) // 2
        dialog_y = (self.height - dialog_height) // 2

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        canvas = cv2.addWeighted(canvas, 0.3, overlay, 0.7, 0)

        cv2.rectangle(
            canvas,
            (dialog_x, dialog_y),
            (dialog_x + dialog_width, dialog_y + dialog_height),
            (245, 245, 245),
            -1,
        )
        cv2.rectangle(
            canvas,
            (dialog_x, dialog_y),
            (dialog_x + dialog_width, dialog_y + dialog_height),
            (0, 0, 0),
            3,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        title_font = cv2.FONT_HERSHEY_DUPLEX
        title = "Piece Breakdown"
        title_scale = 0.92
        title_thickness = 2
        title_size = cv2.getTextSize(title, title_font, title_scale, title_thickness)[0]
        title_x = dialog_x + (dialog_width - title_size[0]) // 2
        title_y = dialog_y + 52
        cv2.putText(
            canvas,
            title,
            (title_x, title_y),
            title_font,
            title_scale,
            (0, 0, 0),
            title_thickness,
        )

        self.stats_class_modal_class_row_rects = []
        self.stats_class_modal_status_row_rects = []
        self.stats_class_modal_jsn_row_rects = []
        self.stats_class_modal_copy_rects = []
        self.stats_class_modal_back_rect = None
        self.stats_class_modal_summary_tab_rect = None
        self.stats_class_modal_matrix_tab_rect = None
        self.stats_class_modal_matrix_report_rect = None
        self.stats_class_modal_dataset_tab_rect = None
        self.stats_class_modal_list_rect = None
        self.stats_class_modal_scrollbar_rect = None
        self.stats_class_modal_dataset_result_rects = []
        self.stats_class_modal_dataset_angle_rects = []
        self.stats_class_modal_dataset_class_rects = []
        self.stats_class_modal_dataset_export_rect = None

        tabs_y = dialog_y + 72
        tabs_height = 38
        tab_width = 130

        def draw_tab_button(rect, label, is_active):
            fill = (55, 55, 55) if is_active else (125, 125, 125)
            hover = self._is_point_in_rect(self.mouse_x, self.mouse_y, rect)
            if hover and not is_active:
                fill = (105, 105, 105)
            bx, by, bw, bh = rect
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), fill, -1)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (0, 0, 0), 2)
            text_size = cv2.getTextSize(label, font, 0.63, 2)[0]
            text_x = bx + (bw - text_size[0]) // 2
            text_y = by + (bh + text_size[1]) // 2
            cv2.putText(canvas, label, (text_x, text_y), font, 0.63, (255, 255, 255), 2)

        if self.stats_class_modal_view == "detail":
            table_x = dialog_x + 24
            table_y = dialog_y + 95
            table_w = dialog_width - 48
            table_h = dialog_height - 190
            list_content_y = table_y + 54
            list_content_h = table_h - 70
            header_h = 46

            cv2.rectangle(
                canvas,
                (table_x, table_y),
                (table_x + table_w, table_y + table_h),
                (230, 230, 230),
                -1,
            )
            cv2.rectangle(
                canvas,
                (table_x, table_y),
                (table_x + table_w, table_y + table_h),
                (120, 120, 120),
                2,
            )
            cv2.rectangle(
                canvas,
                (table_x + 2, table_y + 2),
                (table_x + table_w - 2, table_y + header_h),
                (65, 65, 65),
                -1,
            )

            back_width = 88
            back_height = 34
            back_x = dialog_x + 20
            back_y = dialog_y + 20
            self.stats_class_modal_back_rect = (back_x, back_y, back_width, back_height)
            back_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.stats_class_modal_back_rect)
            back_color = (90, 90, 90) if back_hovered else (110, 110, 110)
            cv2.rectangle(canvas, (back_x, back_y), (back_x + back_width, back_y + back_height), back_color, -1)
            cv2.rectangle(canvas, (back_x, back_y), (back_x + back_width, back_y + back_height), (0, 0, 0), 2)
            cv2.putText(canvas, "Back", (back_x + 20, back_y + 23), font, 0.62, (255, 255, 255), 2)

            header_left = "JSN"
            header_index = "Hist #"
            header_date = "Date"
            is_status_detail = self.stats_class_modal_selected_kind == "status"
            header_context = "Defect" if is_status_detail else "Final Result"
            header_right = "Action"
            detail_rows = []
            for row in self.stats_class_modal_detail_rows or []:
                jsn_value = str(row.get("jsn") or "").strip()
                if not jsn_value:
                    continue
                detail_rows.append(
                    {
                        "jsn": jsn_value,
                        "historic_index": row.get("historic_index"),
                        "piece_date_display": str(row.get("piece_date_display") or "N/A").strip() or "N/A",
                        "context_value": (
                            str(row.get("class_name") or "UNCLASSIFIED").strip() or "UNCLASSIFIED"
                        ) if is_status_detail else (
                            str(row.get("final_result") or "N/A").strip() or "N/A"
                        ),
                    }
                )
            row_h = 48
            visible_capacity = max(1, list_content_h // row_h)
            self.stats_class_modal_detail_visible_rows = visible_capacity
            self._clamp_stats_class_modal_detail_offset()
            start_idx = self.stats_class_modal_detail_offset
            visible_rows = detail_rows[start_idx:start_idx + visible_capacity]
            self.stats_class_modal_list_rect = (table_x + 8, list_content_y, table_w - 16, list_content_h)

            if not visible_rows:
                empty_text = (
                    "No JSNs available for this final result"
                    if is_status_detail
                    else "No JSNs available for this class"
                )
                empty_size = cv2.getTextSize(empty_text, font, 0.8, 2)[0]
                empty_x = table_x + (table_w - empty_size[0]) // 2
                empty_y = table_y + header_h + (table_h - header_h) // 2
                cv2.putText(canvas, empty_text, (empty_x, empty_y), font, 0.8, (70, 70, 70), 2)
            else:
                copy_button_width = 92
                index_col_width = 80
                date_col_width = 240
                context_col_width = 140
                scroll_track_width = 12 if len(detail_rows) > visible_capacity else 0
                row_left = table_x + 6
                row_right = table_x + table_w - 6 - scroll_track_width
                copy_left = row_right - copy_button_width - 8
                context_left = copy_left - context_col_width - 10
                date_left = context_left - date_col_width - 10
                index_left = date_left - index_col_width - 10
                jsn_col_width = max(170, index_left - row_left - 12)
                jsn_max_width = max(140, jsn_col_width - 16)
                cv2.putText(canvas, header_left, (row_left + 12, table_y + 32), font, 0.72, (255, 255, 255), 2)
                header_index_size = cv2.getTextSize(header_index, font, 0.68, 2)[0]
                header_index_x = index_left + (index_col_width - header_index_size[0]) // 2
                cv2.putText(
                    canvas,
                    header_index,
                    (header_index_x, table_y + 32),
                    font,
                    0.68,
                    (255, 255, 255),
                    2,
                )
                header_date_size = cv2.getTextSize(header_date, font, 0.68, 2)[0]
                header_date_x = date_left + (date_col_width - header_date_size[0]) // 2
                cv2.putText(
                    canvas,
                    header_date,
                    (header_date_x, table_y + 32),
                    font,
                    0.68,
                    (255, 255, 255),
                    2,
                )
                header_context_size = cv2.getTextSize(header_context, font, 0.68, 2)[0]
                header_context_x = context_left + (context_col_width - header_context_size[0]) // 2
                cv2.putText(
                    canvas,
                    header_context,
                    (header_context_x, table_y + 32),
                    font,
                    0.68,
                    (255, 255, 255),
                    2,
                )
                action_size = cv2.getTextSize(header_right, font, 0.68, 2)[0]
                action_x = copy_left + (copy_button_width - action_size[0]) // 2
                cv2.putText(
                    canvas,
                    header_right,
                    (action_x, table_y + 32),
                    font,
                    0.68,
                    (255, 255, 255),
                    2,
                )

                for visible_idx, detail_row in enumerate(visible_rows):
                    absolute_idx = start_idx + visible_idx
                    row_top = list_content_y + visible_idx * row_h
                    row_bottom = min(list_content_y + list_content_h - 4, row_top + row_h - 4)
                    row_rect = (row_left, row_top, row_right - row_left, row_bottom - row_top)
                    copy_rect = (copy_left, row_top + 5, copy_button_width, row_bottom - row_top - 10)
                    jsn_rect = (row_left, row_top, jsn_col_width, row_bottom - row_top)
                    row_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, jsn_rect)
                    copy_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, copy_rect)
                    fill_color = (242, 247, 255) if row_hovered else ((248, 248, 248) if absolute_idx % 2 == 0 else (238, 238, 238))
                    cv2.rectangle(
                        canvas,
                        (row_rect[0], row_rect[1]),
                        (row_rect[0] + row_rect[2], row_rect[1] + row_rect[3]),
                        fill_color,
                        -1,
                    )

                    jsn_value = detail_row["jsn"]
                    historic_index = detail_row.get("historic_index")
                    jsn_text = self._truncate_text_to_width(jsn_value, font, 0.68, 2, jsn_max_width)
                    cv2.putText(canvas, jsn_text, (row_left + 14, row_top + 30), font, 0.68, (30, 30, 30), 2)
                    historic_text = str(historic_index) if historic_index is not None else "-"
                    historic_size = cv2.getTextSize(historic_text, font, 0.68, 2)[0]
                    historic_x = index_left + (index_col_width - historic_size[0]) // 2
                    cv2.putText(
                        canvas,
                        historic_text,
                        (historic_x, row_top + 30),
                        font,
                        0.68,
                        (30, 30, 30),
                        2,
                    )
                    piece_date_text = self._truncate_text_to_width(
                        detail_row.get("piece_date_display") or "N/A",
                        font,
                        0.62,
                        2,
                        date_col_width - 16,
                    )
                    cv2.putText(
                        canvas,
                        piece_date_text,
                        (date_left + 8, row_top + 29),
                        font,
                        0.62,
                        (30, 30, 30),
                        2,
                    )
                    context_text = self._truncate_text_to_width(
                        detail_row.get("context_value") or "N/A",
                        font,
                        0.62,
                        2,
                        context_col_width - 16,
                    )
                    cv2.putText(
                        canvas,
                        context_text,
                        (context_left + 8, row_top + 29),
                        font,
                        0.62,
                        (30, 30, 30),
                        2,
                    )

                    copy_fill = (70, 130, 70) if copy_hovered else (90, 150, 90)
                    cv2.rectangle(
                        canvas,
                        (copy_rect[0], copy_rect[1]),
                        (copy_rect[0] + copy_rect[2], copy_rect[1] + copy_rect[3]),
                        copy_fill,
                        -1,
                    )
                    cv2.rectangle(
                        canvas,
                        (copy_rect[0], copy_rect[1]),
                        (copy_rect[0] + copy_rect[2], copy_rect[1] + copy_rect[3]),
                        (0, 0, 0),
                        2,
                    )
                    copy_label = "Copy"
                    copy_size = cv2.getTextSize(copy_label, font, 0.64, 2)[0]
                    cv2.putText(
                        canvas,
                        copy_label,
                        (copy_rect[0] + (copy_rect[2] - copy_size[0]) // 2, copy_rect[1] + (copy_rect[3] + copy_size[1]) // 2),
                        font,
                        0.64,
                        (255, 255, 255),
                        2,
                    )

                    self.stats_class_modal_jsn_row_rects.append((jsn_rect, jsn_value))
                    self.stats_class_modal_copy_rects.append((copy_rect, jsn_value))

                footer_text = f"Showing {start_idx + 1}-{start_idx + len(visible_rows)} of {len(detail_rows)}"
                cv2.putText(
                    canvas,
                    footer_text,
                    (table_x + 14, table_y + table_h - 12),
                    font,
                    0.55,
                    (90, 90, 90),
                    2,
                )

                if len(detail_rows) > visible_capacity:
                    track_x = table_x + table_w - 20
                    track_y = list_content_y
                    track_h = list_content_h - 4
                    thumb_h = max(30, int(track_h * (visible_capacity / len(detail_rows))))
                    max_offset = max(1, len(detail_rows) - visible_capacity)
                    thumb_y = track_y + int((track_h - thumb_h) * (start_idx / max_offset))
                    self.stats_class_modal_scrollbar_rect = (track_x, thumb_y, 10, thumb_h)
                    cv2.rectangle(canvas, (track_x, track_y), (track_x + 10, track_y + track_h), (210, 210, 210), -1)
                    cv2.rectangle(canvas, (track_x, thumb_y), (track_x + 10, thumb_y + thumb_h), (120, 120, 120), -1)
                    cv2.rectangle(canvas, (track_x, track_y), (track_x + 10, track_y + track_h), (90, 90, 90), 1)
        else:
            summary_tab_x = dialog_x + (dialog_width // 2) - tab_width - 146
            matrix_tab_x = dialog_x + (dialog_width // 2) - (tab_width // 2)
            dataset_tab_x = dialog_x + (dialog_width // 2) + 146
            self.stats_class_modal_summary_tab_rect = (
                summary_tab_x,
                tabs_y,
                tab_width,
                tabs_height,
            )
            self.stats_class_modal_matrix_tab_rect = (
                matrix_tab_x,
                tabs_y,
                tab_width,
                tabs_height,
            )
            self.stats_class_modal_dataset_tab_rect = (
                dataset_tab_x,
                tabs_y,
                tab_width,
                tabs_height,
            )
            draw_tab_button(
                self.stats_class_modal_summary_tab_rect,
                "Summary",
                self.stats_class_modal_view == "summary",
            )
            draw_tab_button(
                self.stats_class_modal_matrix_tab_rect,
                "Matrix",
                self.stats_class_modal_view == "matrix",
            )
            draw_tab_button(
                self.stats_class_modal_dataset_tab_rect,
                "Dataset",
                self.stats_class_modal_view == "dataset",
            )

            panel_gap = 28
            panel_y = dialog_y + 128
            panel_h = dialog_height - 223
            panel_x = dialog_x + 35
            panel_w = (dialog_width - 70 - panel_gap) // 2
            header_h = 46
            summary_panel_top_offset = 24

            def draw_summary_panel(
                panel_left,
                panel_title,
                left_header,
                rows,
                key_name,
                rect_target,
                empty_text,
                use_status_colors=False,
                top_offset=0,
            ):
                panel_top = panel_y + top_offset
                panel_height = max(140, panel_h - top_offset)
                cv2.putText(
                    canvas,
                    panel_title,
                    (panel_left + 6, panel_top - 14),
                    font,
                    0.76,
                    (35, 35, 35),
                    2,
                )

                cv2.rectangle(
                    canvas,
                    (panel_left, panel_top),
                    (panel_left + panel_w, panel_top + panel_height),
                    (230, 230, 230),
                    -1,
                )
                cv2.rectangle(
                    canvas,
                    (panel_left, panel_top),
                    (panel_left + panel_w, panel_top + panel_height),
                    (120, 120, 120),
                    2,
                )
                cv2.rectangle(
                    canvas,
                    (panel_left + 2, panel_top + 2),
                    (panel_left + panel_w - 2, panel_top + header_h),
                    (65, 65, 65),
                    -1,
                )

                cv2.putText(
                    canvas,
                    left_header,
                    (panel_left + 20, panel_top + 31),
                    font,
                    0.72,
                    (255, 255, 255),
                    2,
                )
                count_label = "Pieces"
                count_label_size = cv2.getTextSize(count_label, font, 0.72, 2)[0]
                cv2.putText(
                    canvas,
                    count_label,
                    (panel_left + panel_w - 20 - count_label_size[0], panel_top + 31),
                    font,
                    0.72,
                    (255, 255, 255),
                    2,
                )

                row_h = 38
                max_rows = max(1, (panel_height - header_h - 18) // row_h)
                all_rows = list(rows or [])
                total_row = all_rows[-1] if all_rows and all_rows[-1].get("is_total") else None
                data_rows = all_rows[:-1] if total_row else all_rows
                if total_row and max_rows >= 2:
                    visible_rows = data_rows[: max_rows - 1] + [total_row]
                    hidden_count = max(0, len(data_rows) - (max_rows - 1))
                else:
                    visible_rows = all_rows[:max_rows]
                    hidden_count = max(0, len(all_rows) - len(visible_rows))

                if not visible_rows:
                    empty_size = cv2.getTextSize(empty_text, font, 0.75, 2)[0]
                    empty_x = panel_left + (panel_w - empty_size[0]) // 2
                    empty_y = panel_top + header_h + (panel_height - header_h) // 2
                    cv2.putText(canvas, empty_text, (empty_x, empty_y), font, 0.75, (70, 70, 70), 2)
                    return

                label_max_width = panel_w - 170
                for idx, row in enumerate(visible_rows):
                    row_top = panel_top + header_h + 8 + idx * row_h
                    row_bottom = row_top + row_h - 4
                    row_rect = (panel_left + 8, row_top, panel_w - 16, row_bottom - row_top)
                    label_value = str(row.get(key_name) or "N/A")
                    piece_count = str(row.get("piece_count", 0))
                    is_total = bool(row.get("is_total"))
                    row_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, row_rect)
                    fill_color = (242, 247, 255) if row_hovered else ((248, 248, 248) if idx % 2 == 0 else (238, 238, 238))
                    text_color = (30, 30, 30)

                    if is_total:
                        fill_color = (85, 85, 85)
                        text_color = (255, 255, 255)
                    elif use_status_colors:
                        status_fill = self._get_stats_result_color(label_value)
                        fill_color = tuple(min(255, channel + 50) for channel in status_fill) if row_hovered else status_fill
                        text_color = (255, 255, 255)

                    cv2.rectangle(
                        canvas,
                        (row_rect[0], row_rect[1]),
                        (row_rect[0] + row_rect[2], row_rect[1] + row_rect[3]),
                        fill_color,
                        -1,
                    )

                    label_text = self._truncate_text_to_width(label_value, font, 0.68, 2, label_max_width)
                    cv2.putText(
                        canvas,
                        label_text,
                        (panel_left + 22, row_top + 24),
                        font,
                        0.68,
                        text_color,
                        2,
                    )
                    piece_size = cv2.getTextSize(piece_count, font, 0.68, 2)[0]
                    cv2.putText(
                        canvas,
                        piece_count,
                        (panel_left + panel_w - 22 - piece_size[0], row_top + 24),
                        font,
                        0.68,
                        text_color,
                        2,
                    )
                    if not is_total:
                        rect_target.append((row_rect, label_value))

                if hidden_count > 0:
                    more_text = f"+{hidden_count} more"
                    cv2.putText(
                        canvas,
                        more_text,
                        (panel_left + 16, panel_top + panel_height - 14),
                        font,
                        0.6,
                        (90, 90, 90),
                        2,
                    )

            if self.stats_class_modal_view == "summary":
                draw_summary_panel(
                    panel_x,
                    "By Class Name",
                    "Class Name",
                    list(self.stats_class_modal_rows or []),
                    "class_name",
                    self.stats_class_modal_class_row_rects,
                    "No class data available",
                    top_offset=summary_panel_top_offset,
                )
                draw_summary_panel(
                    panel_x + panel_w + panel_gap,
                    "By Final Result",
                    "Result",
                    list(self.stats_class_modal_status_rows or []),
                    "final_result",
                    self.stats_class_modal_status_row_rects,
                    "No result data available",
                    use_status_colors=True,
                    top_offset=summary_panel_top_offset,
                )
            elif self.stats_class_modal_view == "dataset":
                panel_y = dialog_y + 126
                panel_h = dialog_height - 220
                left_panel_w = 320
                gap_w = 24
                class_panel_x = dialog_x + 28 + left_panel_w + gap_w
                class_panel_w = dialog_width - 56 - left_panel_w - gap_w
                section_gap = 22
                info_text = "Filters are applied by intersection. Images with multiple defects are copied once per matching class."
                info_lines = self._wrap_text_to_width(
                    info_text,
                    font,
                    0.58,
                    2,
                    dialog_width - 120,
                    max_lines=2,
                )
                info_y = dialog_y + 108
                for idx, line in enumerate(info_lines):
                    cv2.putText(
                        canvas,
                        line,
                        (dialog_x + 36, info_y + idx * 24),
                        font,
                        0.58,
                        (65, 65, 65),
                        2,
                    )

                def draw_dataset_section(panel_rect, title, options, selected_values, rect_target, value_formatter=None):
                    px, py, pw, ph = panel_rect
                    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (230, 230, 230), -1)
                    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (120, 120, 120), 2)
                    cv2.rectangle(canvas, (px + 2, py + 2), (px + pw - 2, py + 44), (65, 65, 65), -1)
                    cv2.putText(canvas, title, (px + 16, py + 30), font, 0.72, (255, 255, 255), 2)

                    chip_x = px + 12
                    chip_y = py + 58
                    chip_w = pw - 24
                    chip_h = 42
                    row_gap = 10
                    for idx, option in enumerate(options or []):
                        rect = (chip_x, chip_y + idx * (chip_h + row_gap), chip_w, chip_h)
                        is_selected = option in (selected_values or set())
                        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, rect)
                        fill = (75, 135, 75) if is_selected else (245, 245, 245)
                        text_color = (255, 255, 255) if is_selected else (35, 35, 35)
                        if title == "Result" and is_selected:
                            fill = self._get_stats_result_color(option)
                        if is_hovered and not is_selected:
                            fill = (232, 239, 250)
                        cv2.rectangle(
                            canvas,
                            (rect[0], rect[1]),
                            (rect[0] + rect[2], rect[1] + rect[3]),
                            fill,
                            -1,
                        )
                        cv2.rectangle(
                            canvas,
                            (rect[0], rect[1]),
                            (rect[0] + rect[2], rect[1] + rect[3]),
                            (70, 70, 70),
                            2,
                        )
                        label = value_formatter(option) if callable(value_formatter) else str(option)
                        label = self._truncate_text_to_width(label, font, 0.65, 2, chip_w - 20)
                        label_size = cv2.getTextSize(label, font, 0.65, 2)[0]
                        label_x = rect[0] + 12
                        label_y = rect[1] + (rect[3] + label_size[1]) // 2
                        cv2.putText(canvas, label, (label_x, label_y), font, 0.65, text_color, 2)
                        rect_target.append((rect, option))

                result_panel_h = 242
                angle_panel_h = 190
                result_panel_rect = (dialog_x + 28, panel_y, left_panel_w, result_panel_h)
                angle_panel_rect = (
                    dialog_x + 28,
                    panel_y + result_panel_h + section_gap,
                    left_panel_w,
                    angle_panel_h,
                )

                draw_dataset_section(
                    result_panel_rect,
                    "Result",
                    list(self.stats_class_modal_dataset_result_options or []),
                    set(self.stats_class_modal_dataset_selected_results or set()),
                    self.stats_class_modal_dataset_result_rects,
                )
                draw_dataset_section(
                    angle_panel_rect,
                    "Angle",
                    list(self.stats_class_modal_dataset_angle_options or []),
                    set(self.stats_class_modal_dataset_selected_angles or set()),
                    self.stats_class_modal_dataset_angle_rects,
                    value_formatter=lambda value: str(value).upper(),
                )

                class_panel_rect = (class_panel_x, panel_y, class_panel_w, panel_h)
                cv2.rectangle(
                    canvas,
                    (class_panel_rect[0], class_panel_rect[1]),
                    (class_panel_rect[0] + class_panel_rect[2], class_panel_rect[1] + class_panel_rect[3]),
                    (230, 230, 230),
                    -1,
                )
                cv2.rectangle(
                    canvas,
                    (class_panel_rect[0], class_panel_rect[1]),
                    (class_panel_rect[0] + class_panel_rect[2], class_panel_rect[1] + class_panel_rect[3]),
                    (120, 120, 120),
                    2,
                )
                cv2.rectangle(
                    canvas,
                    (class_panel_rect[0] + 2, class_panel_rect[1] + 2),
                    (class_panel_rect[0] + class_panel_rect[2] - 2, class_panel_rect[1] + 44),
                    (65, 65, 65),
                    -1,
                )
                cv2.putText(
                    canvas,
                    "Class Name",
                    (class_panel_rect[0] + 16, class_panel_rect[1] + 30),
                    font,
                    0.72,
                    (255, 255, 255),
                    2,
                )

                export_btn_w = 220
                export_btn_h = 44
                export_btn_x = class_panel_rect[0] + class_panel_rect[2] - export_btn_w - 16
                export_btn_y = class_panel_rect[1] + class_panel_rect[3] - export_btn_h - 14
                self.stats_class_modal_dataset_export_rect = (
                    export_btn_x,
                    export_btn_y,
                    export_btn_w,
                    export_btn_h,
                )

                summary_text = (
                    f"Selected: {len(self.stats_class_modal_dataset_selected_results or [])} results, "
                    f"{len(self.stats_class_modal_dataset_selected_angles or [])} angles, "
                    f"{len(self.stats_class_modal_dataset_selected_classes or [])} class filters"
                )
                cv2.putText(
                    canvas,
                    self._truncate_text_to_width(summary_text, font, 0.56, 2, class_panel_rect[2] - 270),
                    (class_panel_rect[0] + 16, class_panel_rect[1] + class_panel_rect[3] - 22),
                    font,
                    0.56,
                    (80, 80, 80),
                    2,
                )

                class_list_x = class_panel_rect[0] + 12
                class_list_y = class_panel_rect[1] + 58
                class_list_w = class_panel_rect[2] - 24
                class_list_h = class_panel_rect[3] - 128
                self.stats_class_modal_list_rect = (
                    class_list_x,
                    class_list_y,
                    class_list_w,
                    class_list_h,
                )

                class_options = list(self.stats_class_modal_dataset_class_options or [])
                cols = 2
                chip_gap_x = 12
                chip_gap_y = 10
                scroll_track_width = 12 if len(class_options) > 0 else 0
                chip_w = (class_list_w - scroll_track_width - chip_gap_x) // cols
                chip_h = 42
                total_rows = (len(class_options) + cols - 1) // cols
                visible_rows = max(1, class_list_h // (chip_h + chip_gap_y))
                self.stats_class_modal_dataset_class_visible_rows = visible_rows
                self._clamp_stats_class_modal_dataset_class_offset()
                start_row = self.stats_class_modal_dataset_class_offset
                start_idx = start_row * cols
                end_idx = min(len(class_options), start_idx + visible_rows * cols)
                visible_class_options = class_options[start_idx:end_idx]

                if not visible_class_options:
                    empty_text = "No class filters available"
                    empty_size = cv2.getTextSize(empty_text, font, 0.78, 2)[0]
                    empty_x = class_panel_rect[0] + (class_panel_rect[2] - empty_size[0]) // 2
                    empty_y = class_panel_rect[1] + 120
                    cv2.putText(canvas, empty_text, (empty_x, empty_y), font, 0.78, (70, 70, 70), 2)
                else:
                    selected_classes = set(self.stats_class_modal_dataset_selected_classes or set())
                    for option_idx, option in enumerate(visible_class_options):
                        row_idx = option_idx // cols
                        col_idx = option_idx % cols
                        rect = (
                            class_list_x + col_idx * (chip_w + chip_gap_x),
                            class_list_y + row_idx * (chip_h + chip_gap_y),
                            chip_w,
                            chip_h,
                        )
                        is_selected = option in selected_classes
                        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, rect)
                        fill = (75, 135, 75) if is_selected else (245, 245, 245)
                        text_color = (255, 255, 255) if is_selected else (35, 35, 35)
                        if option == "All" and is_selected:
                            fill = (80, 110, 160)
                        if is_hovered and not is_selected:
                            fill = (232, 239, 250)
                        cv2.rectangle(
                            canvas,
                            (rect[0], rect[1]),
                            (rect[0] + rect[2], rect[1] + rect[3]),
                            fill,
                            -1,
                        )
                        cv2.rectangle(
                            canvas,
                            (rect[0], rect[1]),
                            (rect[0] + rect[2], rect[1] + rect[3]),
                            (70, 70, 70),
                            2,
                        )
                        label = self._truncate_text_to_width(str(option), font, 0.62, 2, chip_w - 20)
                        label_size = cv2.getTextSize(label, font, 0.62, 2)[0]
                        label_x = rect[0] + 10
                        label_y = rect[1] + (rect[3] + label_size[1]) // 2
                        cv2.putText(canvas, label, (label_x, label_y), font, 0.62, text_color, 2)
                        self.stats_class_modal_dataset_class_rects.append((rect, option))

                    if total_rows > visible_rows:
                        track_x = class_panel_rect[0] + class_panel_rect[2] - 20
                        track_y = class_list_y
                        track_h = class_list_h - 4
                        thumb_h = max(30, int(track_h * (visible_rows / total_rows)))
                        max_offset = max(1, total_rows - visible_rows)
                        thumb_y = track_y + int((track_h - thumb_h) * (start_row / max_offset))
                        self.stats_class_modal_scrollbar_rect = (track_x, thumb_y, 10, thumb_h)
                        cv2.rectangle(canvas, (track_x, track_y), (track_x + 10, track_y + track_h), (210, 210, 210), -1)
                        cv2.rectangle(canvas, (track_x, thumb_y), (track_x + 10, thumb_y + thumb_h), (120, 120, 120), -1)
                        cv2.rectangle(canvas, (track_x, track_y), (track_x + 10, track_y + track_h), (90, 90, 90), 1)

                export_hovered = self._is_point_in_rect(
                    self.mouse_x,
                    self.mouse_y,
                    self.stats_class_modal_dataset_export_rect,
                )
                export_fill = (68, 120, 196) if export_hovered else (88, 140, 216)
                cv2.rectangle(
                    canvas,
                    (export_btn_x, export_btn_y),
                    (export_btn_x + export_btn_w, export_btn_y + export_btn_h),
                    export_fill,
                    -1,
                )
                cv2.rectangle(
                    canvas,
                    (export_btn_x, export_btn_y),
                    (export_btn_x + export_btn_w, export_btn_y + export_btn_h),
                    (0, 0, 0),
                    2,
                )
                export_label = "Export Dataset"
                export_label_size = cv2.getTextSize(export_label, font, 0.68, 2)[0]
                cv2.putText(
                    canvas,
                    export_label,
                    (
                        export_btn_x + (export_btn_w - export_label_size[0]) // 2,
                        export_btn_y + (export_btn_h + export_label_size[1]) // 2,
                    ),
                    font,
                    0.68,
                    (255, 255, 255),
                    2,
                )
            else:
                table_x = dialog_x + 45
                table_y = dialog_y + 128
                table_w = dialog_width - 90
                table_h = dialog_height - 223
                header_h = 46
                list_content_y = table_y + header_h + 8
                list_content_h = table_h - header_h - 26
                self.stats_class_modal_list_rect = (table_x + 8, list_content_y, table_w - 16, list_content_h)

                cv2.rectangle(
                    canvas,
                    (table_x, table_y),
                    (table_x + table_w, table_y + table_h),
                    (230, 230, 230),
                    -1,
                )
                cv2.rectangle(
                    canvas,
                    (table_x, table_y),
                    (table_x + table_w, table_y + table_h),
                    (120, 120, 120),
                    2,
                )

                report_button_width = 180
                report_button_height = 36
                report_button_x = dialog_x + dialog_width - report_button_width - 34
                report_button_y = dialog_y + 82
                self.stats_class_modal_matrix_report_rect = (
                    report_button_x,
                    report_button_y,
                    report_button_width,
                    report_button_height,
                )
                report_hovered = self._is_point_in_rect(
                    self.mouse_x,
                    self.mouse_y,
                    self.stats_class_modal_matrix_report_rect,
                )
                report_fill = (68, 120, 196) if report_hovered else (88, 140, 216)
                cv2.rectangle(
                    canvas,
                    (report_button_x, report_button_y),
                    (
                        report_button_x + report_button_width,
                        report_button_y + report_button_height,
                    ),
                    report_fill,
                    -1,
                )
                cv2.rectangle(
                    canvas,
                    (report_button_x, report_button_y),
                    (
                        report_button_x + report_button_width,
                        report_button_y + report_button_height,
                    ),
                    (0, 0, 0),
                    2,
                )
                report_text = "Export Excel"
                report_text_size = cv2.getTextSize(report_text, font, 0.66, 2)[0]
                cv2.putText(
                    canvas,
                    report_text,
                    (
                        report_button_x
                        + (report_button_width - report_text_size[0]) // 2,
                        report_button_y
                        + (report_button_height + report_text_size[1]) // 2,
                    ),
                    font,
                    0.66,
                    (255, 255, 255),
                    2,
                )

                matrix_rows = list(self.stats_class_modal_matrix_rows or [])
                total_row = matrix_rows[-1] if matrix_rows and matrix_rows[-1].get("is_total") else None
                data_rows = matrix_rows[:-1] if total_row else matrix_rows
                row_h = 40
                total_reserved = 1 if total_row else 0
                visible_capacity = max(1, list_content_h // row_h)
                visible_data_capacity = max(1, visible_capacity - total_reserved) if total_reserved else visible_capacity
                self.stats_class_modal_matrix_visible_rows = visible_data_capacity
                self._clamp_stats_class_modal_matrix_offset()
                start_idx = self.stats_class_modal_matrix_offset
                visible_rows = data_rows[start_idx:start_idx + visible_data_capacity]
                if total_row:
                    visible_rows = visible_rows + [total_row]

                label_col_w = 340
                value_col_w = (table_w - 16 - label_col_w) // 5
                header_labels = ["Class Name", "OK", "NOK", "FOK", "FNOK", "Total"]
                header_x = table_x + 8
                for idx, label in enumerate(header_labels):
                    col_x = header_x if idx == 0 else header_x + label_col_w + (idx - 1) * value_col_w
                    col_w = label_col_w if idx == 0 else value_col_w
                    cv2.rectangle(
                        canvas,
                        (col_x, table_y + 2),
                        (col_x + col_w, table_y + header_h),
                        (65, 65, 65),
                        -1,
                    )
                    text_size = cv2.getTextSize(label, font, 0.7, 2)[0]
                    text_x = col_x + (col_w - text_size[0]) // 2
                    cv2.putText(
                        canvas,
                        label,
                        (text_x, table_y + 31),
                        font,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                if not visible_rows:
                    empty_text = "No matrix data available"
                    empty_size = cv2.getTextSize(empty_text, font, 0.8, 2)[0]
                    empty_x = table_x + (table_w - empty_size[0]) // 2
                    empty_y = table_y + header_h + (table_h - header_h) // 2
                    cv2.putText(canvas, empty_text, (empty_x, empty_y), font, 0.8, (70, 70, 70), 2)
                else:
                    scroll_track_width = 12 if len(data_rows) > visible_data_capacity else 0
                    usable_row_w = table_w - 16 - scroll_track_width
                    numeric_w = (usable_row_w - label_col_w) // 5
                    label_w = usable_row_w - numeric_w * 5
                    note_text = "Right and bottom totals should match."
                    note_size = cv2.getTextSize(note_text, font, 0.58, 2)[0]
                    cv2.putText(
                        canvas,
                        note_text,
                        (table_x + table_w - 18 - note_size[0], table_y + table_h - 10),
                        font,
                        0.58,
                        (80, 80, 80),
                        2,
                    )

                    for row_idx, row in enumerate(visible_rows):
                        row_top = list_content_y + row_idx * row_h
                        row_bottom = min(table_y + table_h - 28, row_top + row_h - 4)
                        is_total = bool(row.get("is_total"))
                        fill = (88, 88, 88) if is_total else ((248, 248, 248) if row_idx % 2 == 0 else (238, 238, 238))
                        text_color = (255, 255, 255) if is_total else (30, 30, 30)

                        cv2.rectangle(
                            canvas,
                            (table_x + 8, row_top),
                            (table_x + 8 + usable_row_w, row_bottom),
                            fill,
                            -1,
                        )

                        label_value = self._truncate_text_to_width(
                            str(row.get("class_name") or ""),
                            font,
                            0.65,
                            2,
                            label_w - 18,
                        )
                        cv2.putText(
                            canvas,
                            label_value,
                            (table_x + 20, row_top + 25),
                            font,
                            0.65,
                            text_color,
                            2,
                        )

                        for col_idx, key in enumerate(("OK", "NOK", "FOK", "FNOK", "Total")):
                            cell_x = table_x + 8 + label_w + col_idx * numeric_w
                            cell_text = str(row.get(key, 0))
                            cell_size = cv2.getTextSize(cell_text, font, 0.65, 2)[0]
                            text_x = cell_x + (numeric_w - cell_size[0]) // 2
                            cv2.putText(
                                canvas,
                                cell_text,
                                (text_x, row_top + 25),
                                font,
                                0.65,
                                text_color,
                                2,
                            )

                    shown_without_total = len(visible_rows) - (1 if total_row and visible_rows else 0)
                    footer_text = f"Showing {start_idx + 1}-{start_idx + shown_without_total} of {len(data_rows)}"
                    if not data_rows:
                        footer_text = "Showing 0 of 0"
                    cv2.putText(
                        canvas,
                        footer_text,
                        (table_x + 14, table_y + table_h - 10),
                        font,
                        0.55,
                        (90, 90, 90),
                        2,
                    )

                    if len(data_rows) > visible_data_capacity:
                        track_x = table_x + table_w - 20
                        track_y = list_content_y
                        track_h = list_content_h - 24
                        thumb_h = max(30, int(track_h * (visible_data_capacity / len(data_rows))))
                        max_offset = max(1, len(data_rows) - visible_data_capacity)
                        thumb_y = track_y + int((track_h - thumb_h) * (start_idx / max_offset))
                        self.stats_class_modal_scrollbar_rect = (track_x, thumb_y, 10, thumb_h)
                        cv2.rectangle(canvas, (track_x, track_y), (track_x + 10, track_y + track_h), (210, 210, 210), -1)
                        cv2.rectangle(canvas, (track_x, thumb_y), (track_x + 10, thumb_y + thumb_h), (120, 120, 120), -1)
                        cv2.rectangle(canvas, (track_x, track_y), (track_x + 10, track_y + track_h), (90, 90, 90), 1)

        button_width = 110
        button_height = 38
        button_x = dialog_x + (dialog_width - button_width) // 2
        button_y = dialog_y + dialog_height - 58
        self.stats_class_modal_close_rect = (button_x, button_y, button_width, button_height)

        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.stats_class_modal_close_rect)
        is_pressed = is_hovered and self.mouse_button_down
        scale_factor = 0.95 if is_pressed else (1.08 if is_hovered else 1.0)
        scaled_rect = self._scale_rect(self.stats_class_modal_close_rect, scale_factor)
        bx, by, bw, bh = scaled_rect

        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (100, 100, 100), -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (0, 0, 0), 2)

        close_text = "Close"
        close_size = cv2.getTextSize(close_text, font, 0.7, 2)[0]
        close_x = bx + (bw - close_size[0]) // 2
        close_y = by + (bh + close_size[1]) // 2
        cv2.putText(canvas, close_text, (close_x, close_y), font, 0.7, (255, 255, 255), 2)

        if self.stats_class_modal_view == "detail":
            detail_prefix = "Final Result: " if self.stats_class_modal_selected_kind == "status" else "Class: "
            subtitle_text = self._truncate_text_to_width(
                f"{detail_prefix}{self.stats_class_modal_selected_label}",
                font,
                0.72,
                2,
                dialog_width - 210,
            )
            subtitle_size = cv2.getTextSize(subtitle_text, font, 0.72, 2)[0]
            subtitle_x = dialog_x + (dialog_width - subtitle_size[0]) // 2
            subtitle_y = dialog_y + 84
            cv2.putText(canvas, subtitle_text, (subtitle_x, subtitle_y), font, 0.72, (40, 40, 40), 2)

        return canvas

    def mouse_callback(self, event, x, y, flags, _param):
        """Callback to handle mouse events"""
        # Track mouse position and button state
        self.mouse_x = x
        self.mouse_y = y
        self.mouse_button_down = (flags & cv2.EVENT_FLAG_LBUTTON) != 0

        if (
            event == getattr(cv2, "EVENT_MOUSEWHEEL", -1)
            and self.show_stats_class_modal
            and self._is_point_in_rect(x, y, self.stats_class_modal_list_rect)
        ):
            wheel_delta = 0
            try:
                wheel_delta = cv2.getMouseWheelDelta(flags)
            except Exception:
                wheel_delta = ((flags >> 16) & 0xFFFF)
                if wheel_delta > 32767:
                    wheel_delta -= 65536
            if wheel_delta:
                if self.stats_class_modal_view == "detail":
                    self._emit_action("stats_detail_scroll", delta=wheel_delta)
                elif self.stats_class_modal_view == "matrix":
                    self._emit_action("stats_matrix_scroll", delta=wheel_delta)
                elif self.stats_class_modal_view == "dataset":
                    self._emit_action("stats_dataset_class_scroll", delta=wheel_delta)
            return

        if event == cv2.EVENT_LBUTTONUP:
            if (
                self._stats_long_press_active
                and not self._stats_long_press_fired
                and self._can_track_stats_long_press()
                and self._is_point_in_rect(x, y, self.stats_card_rect)
            ):
                self._cancel_stats_long_press()
                self._emit_action("open_stats_class_modal")
                return
            self._cancel_stats_long_press()
            return

        if self._stats_long_press_active and (
            not self.mouse_button_down
            or not self._is_point_in_rect(x, y, self.stats_card_rect)
            or not self._can_track_stats_long_press()
        ):
            self._cancel_stats_long_press()
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.db_blocking:
                return

            # Dataset completion/error message is modal until the operator closes it.
            if self.sync_message:
                if self.sync_message_close_button_rect:
                    bx, by, bw, bh = self.sync_message_close_button_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._clear_sync_message()
                return

            # Piece date dialog close button (highest priority)
            if self.show_piece_date_dialog and self.piece_date_dialog_close_rect:
                bx, by, bw, bh = self.piece_date_dialog_close_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._emit_action("close_piece_date_dialog")
                return  # Exit early to prevent other clicks

            # Stats class modal close button (highest priority)
            if self.show_stats_class_modal and self.stats_class_modal_close_rect:
                bx, by, bw, bh = self.stats_class_modal_close_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._emit_action("close_stats_class_modal")
                    return

            if self.show_stats_class_modal:
                if self.stats_class_modal_back_rect:
                    bx, by, bw, bh = self.stats_class_modal_back_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("close_stats_class_detail")
                        return

                if self.stats_class_modal_summary_tab_rect:
                    bx, by, bw, bh = self.stats_class_modal_summary_tab_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("open_stats_summary_view")
                        return

                if self.stats_class_modal_matrix_tab_rect:
                    bx, by, bw, bh = self.stats_class_modal_matrix_tab_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("open_stats_matrix_view")
                        return

                if self.stats_class_modal_matrix_report_rect:
                    bx, by, bw, bh = self.stats_class_modal_matrix_report_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("export_stats_matrix_report")
                        return

                if self.stats_class_modal_dataset_tab_rect:
                    bx, by, bw, bh = self.stats_class_modal_dataset_tab_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("open_stats_dataset_view")
                        return

                for rect, jsn_value in self.stats_class_modal_copy_rects:
                    bx, by, bw, bh = rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("copy_stats_jsn", jsn=jsn_value)
                        return

                for rect, jsn_value in self.stats_class_modal_jsn_row_rects:
                    bx, by, bw, bh = rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action(
                            "open_historic_jsn_from_stats",
                            jsn=jsn_value,
                            filter_kind=self.stats_class_modal_selected_kind,
                            filter_label=self.stats_class_modal_selected_label,
                            filter_rows=list(self.stats_class_modal_detail_rows or []),
                        )
                        return

                for rect, class_name in self.stats_class_modal_class_row_rects:
                    bx, by, bw, bh = rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("open_stats_class_detail", class_name=class_name)
                        return

                for rect, final_result in self.stats_class_modal_status_row_rects:
                    bx, by, bw, bh = rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("open_stats_status_detail", final_result=final_result)
                        return

                for rect, value in self.stats_class_modal_dataset_result_rects:
                    bx, by, bw, bh = rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("toggle_stats_dataset_result", value=value)
                        return

                for rect, value in self.stats_class_modal_dataset_angle_rects:
                    bx, by, bw, bh = rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("toggle_stats_dataset_angle", value=value)
                        return

                for rect, value in self.stats_class_modal_dataset_class_rects:
                    bx, by, bw, bh = rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("toggle_stats_dataset_class", value=value)
                        return

                if self.stats_class_modal_dataset_export_rect:
                    bx, by, bw, bh = self.stats_class_modal_dataset_export_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("export_stats_dataset")
                        return

                return

            # No images dialog OK button (highest priority)
            if self.show_no_images_dialog and self.no_images_ok_button_rect:
                bx, by, bw, bh = self.no_images_ok_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._emit_action("dismiss_no_images_dialog")
                return  # Exit early to prevent other clicks

            # Piece number dialog buttons (high priority)
            if self.show_piece_identifier_dialog:
                for rect, action in (
                    (self.piece_identifier_dialog_cancel_rect, "close_piece_identifier_dialog"),
                    (self.piece_identifier_dialog_save_rect, "save_piece_identifier_only"),
                    (self.piece_identifier_dialog_continue_rect, "save_piece_identifier_and_continue"),
                    (self.piece_identifier_dialog_clear_rect, "clear_piece_identifier"),
                ):
                    if rect:
                        bx, by, bw, bh = rect
                        if bx <= x <= bx + bw and by <= y <= by + bh:
                            self._emit_action(action)
                            return
                return

            if self.show_piece_number_dialog:
                if self.piece_number_dialog_ok_rect:
                    bx, by, bw, bh = self.piece_number_dialog_ok_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("submit_piece_number_dialog")
                        return

                if self.piece_number_dialog_cancel_rect:
                    bx, by, bw, bh = self.piece_number_dialog_cancel_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("close_piece_number_dialog")
                        return

                return

            # Delete-piece confirmation buttons (high priority)
            if self.show_delete_confirm:
                # Confirm button
                if self.delete_confirm_button_rect:
                    bx, by, bw, bh = self.delete_confirm_button_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("confirm_delete")
                        return

                # Cancel button
                if self.delete_cancel_button_rect:
                    bx, by, bw, bh = self.delete_cancel_button_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("cancel_delete_confirm")
                        return
                # If dialog is shown, don't process other clicks
                return
            
            # Reset confirmation buttons (high priority)
            if self.show_reset_confirm:
                # Confirm button
                if self.reset_confirm_button_rect:
                    bx, by, bw, bh = self.reset_confirm_button_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("confirm_reset")
                        return
                
                # Cancel button
                if self.reset_cancel_button_rect:
                    bx, by, bw, bh = self.reset_cancel_button_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("cancel_reset_confirm")
                        return

                # If dialog is shown, don't process other clicks
                return

            # Rebuild confirmation buttons (high priority)
            if self.show_rebuild_confirm:
                if self.rebuild_confirm_button_rect:
                    bx, by, bw, bh = self.rebuild_confirm_button_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("confirm_rebuild_db_from_historic")
                        return

                if self.rebuild_cancel_button_rect:
                    bx, by, bw, bh = self.rebuild_cancel_button_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("cancel_rebuild_db_confirm")
                        return

                return

            # Block UI interactions while long-running operations are active.
            if self.sync_in_progress or self.reset_in_progress:
                return

            if self._can_track_stats_long_press() and self._is_point_in_rect(x, y, self.stats_card_rect):
                self._start_stats_long_press()
                return
            
            # HISTORIC button - only to activate historic mode
            if self.save_button_rect and not self.historic_mode and not self.show_no_images_dialog:
                bx, by, bw, bh = self.save_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._emit_action("enter_historic_mode")
                    return

            # EXIT button - only in normal mode
            if self.exit_button_rect and not self.historic_mode and not self.show_no_images_dialog:
                bx, by, bw, bh = self.exit_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._emit_action("request_exit")
                    return

            if self.export_button_rect and not self.show_no_images_dialog:
                bx, by, bw, bh = self.export_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._emit_action("export_display_state")
                    return

            if self.image_report_button_rect and self.historic_mode and not self.show_no_images_dialog:
                bx, by, bw, bh = self.image_report_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._open_historic_image_report_dialog()
                    return

            if self.import_button_rect and not self.show_no_images_dialog:
                bx, by, bw, bh = self.import_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    package_path = self._choose_import_package_path()
                    if package_path:
                        self._emit_action("import_display_state", package_path=package_path)
                    return

            # BACK button - exit historic mode
            if self.back_button_rect and self.historic_mode:
                bx, by, bw, bh = self.back_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._emit_action("exit_historic_mode")
                    return

            # Historic JSN banner - click to copy current JSN
            if self.historic_jsn_rect and self.historic_mode:
                bx, by, bw, bh = self.historic_jsn_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._copy_current_historic_jsn()
                    return

            if self.piece_identifier_rect and self.historic_mode:
                bx, by, bw, bh = self.piece_identifier_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._emit_action("open_piece_identifier_dialog")
                    return
            
            # INFO icon - show piece date (historic mode)
            if self.info_icon_rect and self.historic_mode:
                bx, by, bw, bh = self.info_icon_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._emit_action("open_piece_date_dialog")
                    return

            if self.piece_counter_rect and self.historic_mode:
                bx, by, bw, bh = self.piece_counter_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._emit_action("open_piece_number_dialog")
                    return
            
            # NEXT ARROW button (right) - advance in historic
            if self.next_button_rect and self.historic_mode:
                bx, by, bw, bh = self.next_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._emit_action("next_historic_batch")
            
            # PREVIOUS ARROW button (left) - go back in historic
            if self.prev_button_rect and self.historic_mode:
                bx, by, bw, bh = self.prev_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._emit_action("prev_historic_batch")
            
            # Search button - in historic mode
            if self.search_button_rect and self.historic_mode:
                bx, by, bw, bh = self.search_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._emit_action("search_submit")
            
            # RESET button - in historic mode
            if self.reset_button_rect and self.historic_mode and not self.show_reset_confirm and not self.show_delete_confirm:
                bx, by, bw, bh = self.reset_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._emit_action("open_reset_confirm")

            # TRASH button - in historic mode
            if self.trash_button_rect and self.historic_mode and not self.show_reset_confirm and not self.show_delete_confirm:
                bx, by, bw, bh = self.trash_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._emit_action("open_delete_confirm")

            # SYNC button - in historic mode
            if self.sync_button_rect and self.historic_mode and not self.show_reset_confirm and not self.show_delete_confirm:
                bx, by, bw, bh = self.sync_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._emit_action("sync_images_by_status")
            
            # Search input field - in historic mode
            if self.search_input_rect and self.historic_mode:
                bx, by, bw, bh = self.search_input_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._emit_action("search_focus")
                    return
            
            # Suggestion items - in historic mode
            clicked_on_suggestion = False
            if self.historic_mode and self.suggestion_rects:
                for idx, (rect, jsn_value) in enumerate(self.suggestion_rects):
                    bx, by, bw, bh = rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self._emit_action("search_select_suggestion", jsn=jsn_value[:21])
                        clicked_on_suggestion = True
                        break
            
            # Close suggestions if clicked outside of search area
            if self.historic_mode and (self.search_active or self.filtered_suggestions) and not clicked_on_suggestion:
                # Check if click is outside search input and search button
                clicked_on_search = False
                if self.search_input_rect:
                    bx, by, bw, bh = self.search_input_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        clicked_on_search = True
                
                if self.search_button_rect and not clicked_on_search:
                    bx, by, bw, bh = self.search_button_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        clicked_on_search = True
                
                # If clicked outside, close suggestions
                if not clicked_on_search:
                    self._emit_action("search_blur")
            
            # Result buttons - in historic mode
            if self.historic_mode and self.result_buttons:
                for rect, img_name, result_value in self.result_buttons:
                    bx, by, bw, bh = rect
                    # Check against the scaled area (accounting for hover/press effects)
                    is_hovered = self._is_point_in_rect(x, y, rect)
                    
                    if is_hovered:
                        self._emit_action(
                            "toggle_result",
                            img_name=img_name,
                            result_value=result_value,
                        )
                        break
    
    def _load_historic_index(self, force_rescan=False):
        return self._require_controller()._load_historic_index(force_rescan=force_rescan)

    def enter_historic_mode(self):
        self._require_controller().enter_historic_mode()

    def exit_historic_mode(self):
        self._require_controller().exit_historic_mode()
    
    def next_historic_batch(self):
        self._require_controller().next_historic_batch()
    
    def prev_historic_batch(self):
        self._require_controller().prev_historic_batch()
    
    def collect_available_jsns(self):
        self._require_controller().collect_available_jsns()
    
    def update_suggestions(self):
        self._require_controller().update_suggestions()
    
    def perform_jsn_search(self):
        self._require_controller().perform_jsn_search()

    def _get_current_historic_jsn(self):
        return self._require_controller()._get_current_historic_jsn()

    def perform_delete_current_piece(self):
        self._require_controller().perform_delete_current_piece()

    def perform_reset(self):
        self._require_controller().start_reset_async()

    def start_historic_download_on_startup(self, local_path, check_interval=30):
        import main_controller as _main_controller

        # Keep compatibility with tests that patch display_window.Event/Process.
        _main_controller.Event = Event
        _main_controller.Process = Process
        self._require_controller().start_historic_download_on_startup(
            local_path=local_path,
            check_interval=check_interval,
        )
    
    def download_historic_batch(self, local_path, max_images=7):
        return self._require_controller().download_historic_batch(
            local_path=local_path,
            max_images=max_images,
        )
    
    def _register_local_images_in_db(self, historic_dir, image_names=None):
        self._require_controller()._register_local_images_in_db(
            historic_dir=historic_dir,
            image_names=image_names,
        )
    
    def _update_result_in_db(self, img_name, new_value):
        self._require_controller()._update_result_in_db(img_name=img_name, new_value=new_value)

    def get_result_for_image(self, img_name):
        return self._require_controller().get_result_for_image(img_name)

    def get_model_overlays_for_images(self, image_names):
        return self._require_controller().get_model_overlays_for_images(image_names)
    
    def save_temp_results_to_db(self):
        self._require_controller().save_temp_results_to_db()

    def sync_images_by_status(self, historic_dir=None, base_dir=None):
        self._require_controller().sync_images_by_status(
            historic_dir=historic_dir,
            base_dir=base_dir,
        )
    
    def draw_historic_button(self, canvas):
        """Draw historic button on canvas (visual only)"""
        button_width = 180
        button_height = 60
        margin = 30
        margin_top = 10
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # HISTORIC button (lower left corner)
        x_save = margin
        y_save = self.height - button_height - margin_top
        
        self.save_button_rect = (x_save, y_save, button_width, button_height)
        
        # Check if button is hovered or pressed
        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.save_button_rect)
        is_pressed = is_hovered and self.mouse_button_down
        
        # Scale button on hover
        scale_factor = 0.95 if is_pressed else (1.08 if is_hovered else 1.0)
        scaled_rect = self._scale_rect(self.save_button_rect, scale_factor)
        x_draw, y_draw, w_draw, h_draw = scaled_rect
        
        button_color = (133, 39, 5)
        border_color = (0, 0, 0)
        border_width = 2
        
        cv2.rectangle(canvas, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw), 
                     button_color, -1)
        cv2.rectangle(canvas, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw), 
                     border_color, border_width)
        
        text_save = "HISTORIC"
        text_size_save = cv2.getTextSize(text_save, font, font_scale, thickness)[0]
        text_x_save = x_draw + (w_draw - text_size_save[0]) // 2
        text_y_save = y_draw + (h_draw + text_size_save[1]) // 2
        
        cv2.putText(canvas, text_save, (text_x_save, text_y_save), font, font_scale, 
                   (255, 255, 255), thickness)
        
        return canvas

    def draw_import_button(self, canvas):
        """Draw IMPORT button on canvas."""
        button_width = 180
        button_height = 60
        margin_right = 30
        margin_bottom = 10
        spacing = 20

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        if self.historic_mode:
            x_reset = self.width - button_width - margin_right
            x_sync = x_reset - spacing - button_width
            x_export = x_sync - spacing - button_width
            x_import = x_export - spacing - button_width
        else:
            exit_width = 160
            x_exit = self.width - exit_width - margin_right
            x_export = x_exit - spacing - button_width
            x_import = x_export - spacing - button_width
        y_button = self.height - button_height - margin_bottom

        self.import_button_rect = (x_import, y_button, button_width, button_height)

        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.import_button_rect)
        is_pressed = is_hovered and self.mouse_button_down
        scale_factor = 0.95 if is_pressed else (1.08 if is_hovered else 1.0)
        scaled_rect = self._scale_rect(self.import_button_rect, scale_factor)
        x_draw, y_draw, w_draw, h_draw = scaled_rect

        cv2.rectangle(
            canvas,
            (x_draw, y_draw),
            (x_draw + w_draw, y_draw + h_draw),
            (0, 135, 65),
            -1,
        )
        cv2.rectangle(
            canvas,
            (x_draw, y_draw),
            (x_draw + w_draw, y_draw + h_draw),
            (0, 0, 0),
            2,
        )

        text_label = "IMPORT"
        text_size = cv2.getTextSize(text_label, font, font_scale, thickness)[0]
        text_x = x_draw + (w_draw - text_size[0]) // 2
        text_y = y_draw + (h_draw + text_size[1]) // 2
        cv2.putText(
            canvas,
            text_label,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

        return canvas

    def draw_export_button(self, canvas):
        """Draw EXPORT button on canvas."""
        button_width = 180
        button_height = 60
        margin_right = 30
        margin_bottom = 10
        spacing = 20

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        if self.historic_mode:
            x_reset = self.width - button_width - margin_right
            x_sync = x_reset - spacing - button_width
            x_export = x_sync - spacing - button_width
        else:
            exit_width = 160
            x_exit = self.width - exit_width - margin_right
            x_export = x_exit - spacing - button_width
        y_button = self.height - button_height - margin_bottom

        self.export_button_rect = (x_export, y_button, button_width, button_height)

        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.export_button_rect)
        is_pressed = is_hovered and self.mouse_button_down
        scale_factor = 0.95 if is_pressed else (1.08 if is_hovered else 1.0)
        scaled_rect = self._scale_rect(self.export_button_rect, scale_factor)
        x_draw, y_draw, w_draw, h_draw = scaled_rect

        cv2.rectangle(
            canvas,
            (x_draw, y_draw),
            (x_draw + w_draw, y_draw + h_draw),
            (30, 125, 200),
            -1,
        )
        cv2.rectangle(
            canvas,
            (x_draw, y_draw),
            (x_draw + w_draw, y_draw + h_draw),
            (0, 0, 0),
            2,
        )

        text_label = "EXPORT"
        text_size = cv2.getTextSize(text_label, font, font_scale, thickness)[0]
        text_x = x_draw + (w_draw - text_size[0]) // 2
        text_y = y_draw + (h_draw + text_size[1]) // 2
        cv2.putText(
            canvas,
            text_label,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

        return canvas

    def draw_image_report_button(self, canvas):
        """Draw historic image REPORT button on canvas."""
        button_width = 180
        button_height = 60
        margin_right = 30
        margin_bottom = 10
        spacing = 20

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        x_reset = self.width - button_width - margin_right
        x_sync = x_reset - spacing - button_width
        x_export = x_sync - spacing - button_width
        x_import = x_export - spacing - button_width
        x_report = x_import - spacing - button_width
        y_button = self.height - button_height - margin_bottom

        self.image_report_button_rect = (x_report, y_button, button_width, button_height)

        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.image_report_button_rect)
        is_pressed = is_hovered and self.mouse_button_down
        scale_factor = 0.95 if is_pressed else (1.08 if is_hovered else 1.0)
        scaled_rect = self._scale_rect(self.image_report_button_rect, scale_factor)
        x_draw, y_draw, w_draw, h_draw = scaled_rect

        cv2.rectangle(
            canvas,
            (x_draw, y_draw),
            (x_draw + w_draw, y_draw + h_draw),
            (125, 90, 180),
            -1,
        )
        cv2.rectangle(
            canvas,
            (x_draw, y_draw),
            (x_draw + w_draw, y_draw + h_draw),
            (0, 0, 0),
            2,
        )

        text_label = "REPORT"
        text_size = cv2.getTextSize(text_label, font, font_scale, thickness)[0]
        text_x = x_draw + (w_draw - text_size[0]) // 2
        text_y = y_draw + (h_draw + text_size[1]) // 2
        cv2.putText(
            canvas,
            text_label,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

        return canvas
    
    def draw_back_button(self, canvas):
        """Draw back button on canvas"""
        button_width = 180
        button_height = 60
        margin = 30
        margin_top = 10
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # BACK button (lower left corner)
        x_back = margin
        y_back = self.height - button_height - margin_top
        
        self.back_button_rect = (x_back, y_back, button_width, button_height)
        
        # Check if button is hovered or pressed
        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.back_button_rect)
        is_pressed = is_hovered and self.mouse_button_down
        
        # Scale button on hover
        scale_factor = 0.95 if is_pressed else (1.08 if is_hovered else 1.0)
        scaled_rect = self._scale_rect(self.back_button_rect, scale_factor)
        x_draw, y_draw, w_draw, h_draw = scaled_rect
        
        button_color = (132, 36, 2)
        border_color = (0, 0, 0)
        border_width = 2
        
        cv2.rectangle(canvas, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw), 
                     button_color, -1)
        cv2.rectangle(canvas, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw), 
                     border_color, border_width)
        
        text_back = "BACK"
        text_size_back = cv2.getTextSize(text_back, font, font_scale, thickness)[0]
        text_x_back = x_draw + (w_draw - text_size_back[0]) // 2
        text_y_back = y_draw + (h_draw + text_size_back[1]) // 2
        
        cv2.putText(canvas, text_back, (text_x_back, text_y_back), font, font_scale, 
                   (255, 255, 255), thickness)
        
        return canvas
    
    def draw_reset_button(self, canvas):
        """Draw RESET button on canvas with counter above it"""
        button_width = 180
        button_height = 60
        margin_right = 30
        margin_top = 10
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # RESET button (lower right corner)
        x_reset = self.width - button_width - margin_right
        y_reset = self.height - button_height - margin_top
        
        self.reset_button_rect = (x_reset, y_reset, button_width, button_height)
        
        # Check if button is hovered or pressed
        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.reset_button_rect)
        is_pressed = is_hovered and self.mouse_button_down
        
        # Scale button on hover
        scale_factor = 0.95 if is_pressed else (1.08 if is_hovered else 1.0)
        scaled_rect = self._scale_rect(self.reset_button_rect, scale_factor)
        x_draw, y_draw, w_draw, h_draw = scaled_rect
        
        button_color = (0, 0, 200)
        border_color = (0, 0, 0)
        border_width = 2
        
        cv2.rectangle(canvas, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw), 
                     button_color, -1)  # Red button
        cv2.rectangle(canvas, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw), 
                     border_color, border_width)
        
        text_reset = "RESET"
        text_size_reset = cv2.getTextSize(text_reset, font, font_scale, thickness)[0]
        text_x_reset = x_draw + (w_draw - text_size_reset[0]) // 2
        text_y_reset = y_draw + (h_draw + text_size_reset[1]) // 2
        
        cv2.putText(canvas, text_reset, (text_x_reset, text_y_reset), font, font_scale, 
                   (255, 255, 255), thickness)
        
        # Draw counter above RESET button
        total_pieces = len(self.historic_images)
        current_piece = self._get_current_historic_piece_number()
        counter_text = f"Pieces: {current_piece} of {total_pieces}"
        filter_label = str(getattr(self, "historic_filter_label", "") or "").strip()
        has_filter_label = bool(filter_label)
        counter_font_scale = 0.9
        counter_thickness = 2
        counter_color = (0, 0, 0)  # Black text
        
        counter_size = cv2.getTextSize(counter_text, font, counter_font_scale, counter_thickness)[0]
        filter_font_scale = 0.58
        filter_thickness = 2
        filter_text = ""
        filter_size = (0, 0)
        if has_filter_label:
            filter_total = int(getattr(self, "historic_filter_total_count", 0) or 0)
            if filter_total > total_pieces:
                filter_label_text = f"Filter: {filter_label} ({total_pieces}/{filter_total} local)"
            else:
                filter_label_text = f"Filter: {filter_label}"
            filter_text = self._truncate_text_to_width(
                filter_label_text,
                font,
                filter_font_scale,
                filter_thickness,
                260,
            )
            filter_size = cv2.getTextSize(filter_text, font, filter_font_scale, filter_thickness)[0]
        counter_padding_x = 18
        counter_padding_y = 12
        counter_width = max(counter_size[0], filter_size[0]) + (counter_padding_x * 2)
        counter_height = max(
            40,
            counter_size[1] + (counter_padding_y * 2) + (filter_size[1] + 8 if has_filter_label else 0),
        )
        counter_x = x_draw + (w_draw - counter_width) // 2 - self.right_info_shift
        counter_y = y_draw - counter_height - 16
        self.piece_counter_rect = (counter_x, counter_y, counter_width, counter_height)

        is_counter_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.piece_counter_rect)
        is_counter_pressed = is_counter_hovered and self.mouse_button_down
        counter_fill = (220, 220, 220) if is_counter_hovered else (245, 245, 245)
        counter_border = (0, 0, 0)
        if is_counter_pressed:
            counter_fill = (210, 210, 210)

        cv2.rectangle(
            canvas,
            (counter_x, counter_y),
            (counter_x + counter_width, counter_y + counter_height),
            counter_fill,
            -1,
        )
        cv2.rectangle(
            canvas,
            (counter_x, counter_y),
            (counter_x + counter_width, counter_y + counter_height),
            counter_border,
            2,
        )



        text_x = counter_x + (counter_width - counter_size[0]) // 2
        text_y = counter_y + 28 if has_filter_label else counter_y + counter_height - 12
        cv2.putText(canvas, counter_text, (text_x, text_y), font, counter_font_scale,
                   counter_color, counter_thickness)
        if has_filter_label:
            filter_x = counter_x + (counter_width - filter_size[0]) // 2
            filter_y = counter_y + counter_height - 12
            cv2.putText(
                canvas,
                filter_text,
                (filter_x, filter_y),
                font,
                filter_font_scale,
                (70, 70, 70),
                filter_thickness,
            )
        
        return canvas

    def draw_trash_button(self, canvas):
        """Draw TRASH button on canvas (historic mode)"""
        button_width = 90
        button_height = 60
        margin_right = 30
        margin_top = 10
        spacing = 20

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        # Place TRASH button to the left of IMPORT
        reset_width = 180
        sync_width = 180
        export_width = 180
        import_width = 180
        report_width = 180
        x_reset = self.width - reset_width - margin_right
        x_sync = x_reset - spacing - sync_width
        x_export = x_sync - spacing - export_width
        x_import = x_export - spacing - import_width
        x_report = x_import - spacing - report_width
        x_trash = x_report - spacing - button_width
        y_trash = self.height - button_height - margin_top

        self.trash_button_rect = (x_trash, y_trash, button_width, button_height)
        
        # Check if button is hovered or pressed
        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.trash_button_rect)
        is_pressed = is_hovered and self.mouse_button_down
        
        # Scale button on hover
        scale_factor = 0.95 if is_pressed else (1.08 if is_hovered else 1.0)
        scaled_rect = self._scale_rect(self.trash_button_rect, scale_factor)
        x_draw, y_draw, w_draw, h_draw = scaled_rect
        
        button_color = (200, 200, 200)
        border_color = (0, 0, 0)
        border_width = 2

        cv2.rectangle(canvas, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw),
                     button_color, -1)  # Light gray button
        cv2.rectangle(canvas, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw),
                     border_color, border_width)

        icon_size = 40
        self._load_trash_icon(icon_size)
        if self.trash_icon is not None:
            icon_x = x_draw + (w_draw - icon_size) // 2
            icon_y = y_draw + (h_draw - icon_size) // 2
            self._overlay_icon(canvas, self.trash_icon, icon_x, icon_y)
        else:
            text_trash = "TRASH"
            text_size_trash = cv2.getTextSize(text_trash, font, font_scale, thickness)[0]
            text_x_trash = x_draw + (w_draw - text_size_trash[0]) // 2
            text_y_trash = y_draw + (h_draw + text_size_trash[1]) // 2
            cv2.putText(canvas, text_trash, (text_x_trash, text_y_trash), font, font_scale,
                       (255, 255, 255), thickness)

        return canvas

    def draw_sync_button(self, canvas):
        """Draw SAVE button on canvas (historic mode)"""
        button_width = 180
        button_height = 60
        margin_right = 30
        margin_top = 10
        spacing = 20

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        # Place SYNC button to the left of RESET
        x_sync = self.width - (button_width * 2) - margin_right - spacing
        y_sync = self.height - button_height - margin_top

        self.sync_button_rect = (x_sync, y_sync, button_width, button_height)
        
        # Check if button is hovered or pressed
        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.sync_button_rect)
        is_pressed = is_hovered and self.mouse_button_down
        
        # Scale button on hover
        scale_factor = 0.95 if is_pressed else (1.08 if is_hovered else 1.0)
        scaled_rect = self._scale_rect(self.sync_button_rect, scale_factor)
        x_draw, y_draw, w_draw, h_draw = scaled_rect
        
        button_color = (0, 120, 200)
        border_color = (0, 0, 0)
        border_width = 2

        cv2.rectangle(canvas, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw),
                     button_color, -1)  # Blue-ish button
        cv2.rectangle(canvas, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw),
                     border_color, border_width)

        text_sync = "SAVE"
        text_size_sync = cv2.getTextSize(text_sync, font, font_scale, thickness)[0]
        text_x_sync = x_draw + (w_draw - text_size_sync[0]) // 2
        text_y_sync = y_draw + (h_draw + text_size_sync[1]) // 2

        cv2.putText(canvas, text_sync, (text_x_sync, text_y_sync), font, font_scale,
                   (255, 255, 255), thickness)

        return canvas

    def draw_exit_button(self, canvas):
        """Draw EXIT button on canvas (normal mode)"""
        button_width = 160
        button_height = 60
        margin_right = 30
        margin_bottom = 10

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        x_exit = self.width - button_width - margin_right
        y_exit = self.height - button_height - margin_bottom

        self.exit_button_rect = (x_exit, y_exit, button_width, button_height)
        
        # Check if button is hovered or pressed
        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.exit_button_rect)
        is_pressed = is_hovered and self.mouse_button_down
        
        # Scale button on hover
        scale_factor = 0.95 if is_pressed else (1.08 if is_hovered else 1.0)
        scaled_rect = self._scale_rect(self.exit_button_rect, scale_factor)
        x_draw, y_draw, w_draw, h_draw = scaled_rect
        
        button_color = (0, 0, 200)
        border_color = (0, 0, 0)
        border_width = 2

        cv2.rectangle(canvas, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw),
                     button_color, -1)
        cv2.rectangle(canvas, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw),
                     border_color, border_width)

        text_exit = "EXIT"
        text_size_exit = cv2.getTextSize(text_exit, font, font_scale, thickness)[0]
        text_x_exit = x_draw + (w_draw - text_size_exit[0]) // 2
        text_y_exit = y_draw + (h_draw + text_size_exit[1]) // 2

        cv2.putText(canvas, text_exit, (text_x_exit, text_y_exit), font, font_scale,
                   (255, 255, 255), thickness)

        return canvas

    def draw_info_icon(self, canvas):
        """Draw info icon at top right in historic mode"""
        icon_size = 40
        margin_right = 40
        margin_top = 100
        
        # Position at top right
        x = self.width - icon_size - margin_right
        y = margin_top
        
        self.info_icon_rect = (x, y, icon_size, icon_size)
        
        # Check if icon is hovered or pressed
        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.info_icon_rect)
        is_pressed = is_hovered and self.mouse_button_down
        
        # Scale icon on hover
        scale_factor = 0.95 if is_pressed else (1.08 if is_hovered else 1.0)
        scaled_rect = self._scale_rect(self.info_icon_rect, scale_factor)
        x_draw, y_draw, w_draw, h_draw = scaled_rect
        
        # Draw a circle background
        center_x = x_draw + w_draw // 2
        center_y = y_draw + h_draw // 2
        radius = int((icon_size // 2 - 2) * scale_factor)
        
        # Light blue background
        circle_color = (200, 150, 0)
        border_color = (0, 0, 0)
        border_width = 2
        
        cv2.circle(canvas, (center_x, center_y), radius, circle_color, -1)
        # Dark border
        cv2.circle(canvas, (center_x, center_y), radius, border_color, border_width)
        
        # Draw "i" character in the center
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        text = "i"
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2
        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        return canvas
    
    def draw_piece_date_dialog(self, canvas):
        """Draw modal dialog showing the piece date"""
        # Dialog dimensions
        dialog_width = 400
        dialog_height = 220
        
        # Center the dialog
        dialog_x = (self.width - dialog_width) // 2
        dialog_y = (self.height - dialog_height) // 2
        
        # Draw semi-transparent background overlay
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        canvas = cv2.addWeighted(canvas, 0.3, overlay, 0.7, 0)
        
        # Draw dialog box
        cv2.rectangle(canvas, (dialog_x, dialog_y), (dialog_x + dialog_width, dialog_y + dialog_height),
                     (255, 255, 255), -1)
        # Dialog border
        cv2.rectangle(canvas, (dialog_x, dialog_y), (dialog_x + dialog_width, dialog_y + dialog_height),
                     (0, 0, 0), 3)
        
        # Get piece date and split into date and time
        piece_date_full = self._get_piece_date()
        # Format: "YYYY-MM-DD HH:MM:SS"
        date_parts = piece_date_full.split(' ') if ' ' in piece_date_full else [piece_date_full, ""]
        piece_date = date_parts[0]  # "YYYY-MM-DD"
        piece_time = date_parts[1] if len(date_parts) > 1 else ""  # "HH:MM:SS"
        
        # Draw title
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        
        title_text = "Date"
        title_size = cv2.getTextSize(title_text, font, font_scale, thickness)[0]
        title_x = dialog_x + (dialog_width - title_size[0]) // 2
        title_y = dialog_y + 40
        cv2.putText(canvas, title_text, (title_x, title_y), font, font_scale, (0, 0, 0), thickness)
        
        # Draw date value
        font_scale_date = 1.0
        thickness_date = 2
        date_size = cv2.getTextSize(piece_date, font, font_scale_date, thickness_date)[0]
        date_x = dialog_x + (dialog_width - date_size[0]) // 2
        date_y = dialog_y + 90
        cv2.putText(canvas, piece_date, (date_x, date_y), font, font_scale_date, (50, 50, 200), thickness_date)
        
        # Draw time value (below the date)
        if piece_time:
            time_size = cv2.getTextSize(piece_time, font, font_scale_date, thickness_date)[0]
            time_x = dialog_x + (dialog_width - time_size[0]) // 2
            time_y = dialog_y + 140
            cv2.putText(canvas, piece_time, (time_x, time_y), font, font_scale_date, (50, 50, 200), thickness_date)
        
        # Draw close button
        button_width = 80
        button_height = 30
        button_x = dialog_x + (dialog_width - button_width) // 2
        button_y = dialog_y + dialog_height - 40
        
        self.piece_date_dialog_close_rect = (button_x, button_y, button_width, button_height)
        
        # Check if button is hovered or pressed
        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.piece_date_dialog_close_rect)
        is_pressed = is_hovered and self.mouse_button_down
        
        # Scale button on hover
        scale_factor = 0.95 if is_pressed else (1.08 if is_hovered else 1.0)
        scaled_rect = self._scale_rect(self.piece_date_dialog_close_rect, scale_factor)
        button_x_draw, button_y_draw, button_width_draw, button_height_draw = scaled_rect
        
        button_color = (100, 100, 100)
        border_color = (0, 0, 0)
        border_width = 2
        
        cv2.rectangle(canvas, (button_x_draw, button_y_draw), (button_x_draw + button_width_draw, button_y_draw + button_height_draw),
                     button_color, -1)
        cv2.rectangle(canvas, (button_x_draw, button_y_draw), (button_x_draw + button_width_draw, button_y_draw + button_height_draw),
                     border_color, border_width)
        
        button_text = "Close"
        button_text_size = cv2.getTextSize(button_text, font, font_scale - 0.2, thickness)[0]
        button_text_x = button_x_draw + (button_width_draw - button_text_size[0]) // 2
        button_text_y = button_y_draw + (button_height_draw + button_text_size[1]) // 2
        cv2.putText(canvas, button_text, (button_text_x, button_text_y), font, font_scale - 0.2,
                   (255, 255, 255), thickness)
        
        return canvas

    def draw_piece_number_dialog(self, canvas):
        """Draw modal dialog to jump to a historic piece number."""
        dialog_width = 460
        dialog_height = 260
        dialog_x = (self.width - dialog_width) // 2
        dialog_y = (self.height - dialog_height) // 2

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        canvas = cv2.addWeighted(canvas, 0.3, overlay, 0.7, 0)

        cv2.rectangle(
            canvas,
            (dialog_x, dialog_y),
            (dialog_x + dialog_width, dialog_y + dialog_height),
            (255, 255, 255),
            -1,
        )
        cv2.rectangle(
            canvas,
            (dialog_x, dialog_y),
            (dialog_x + dialog_width, dialog_y + dialog_height),
            (0, 0, 0),
            3,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX

        title_text = "Go To Piece"
        title_scale = 0.95
        title_thickness = 2
        title_size = cv2.getTextSize(title_text, font, title_scale, title_thickness)[0]
        title_x = dialog_x + (dialog_width - title_size[0]) // 2
        title_y = dialog_y + 42
        cv2.putText(canvas, title_text, (title_x, title_y), font, title_scale, (0, 0, 0), title_thickness)

        total_pieces = len(self.historic_images)
        helper_text = f"Available range: 1 to {total_pieces}"
        helper_scale = 0.7
        helper_thickness = 2
        helper_size = cv2.getTextSize(helper_text, font, helper_scale, helper_thickness)[0]
        helper_x = dialog_x + (dialog_width - helper_size[0]) // 2
        helper_y = dialog_y + 82
        cv2.putText(
            canvas,
            helper_text,
            (helper_x, helper_y),
            font,
            helper_scale,
            (80, 80, 80),
            helper_thickness,
        )

        input_width = 200
        input_height = 60
        input_x = dialog_x + (dialog_width - input_width) // 2
        input_y = dialog_y + 108
        cv2.rectangle(
            canvas,
            (input_x, input_y),
            (input_x + input_width, input_y + input_height),
            (250, 250, 220),
            -1,
        )
        cv2.rectangle(
            canvas,
            (input_x, input_y),
            (input_x + input_width, input_y + input_height),
            (0, 0, 0),
            2,
        )

        input_text = self.piece_number_dialog_input or "Piece #"
        input_color = (0, 0, 0) if self.piece_number_dialog_input else (145, 145, 145)
        input_scale = 1.0
        input_thickness = 2
        input_size = cv2.getTextSize(input_text, font, input_scale, input_thickness)[0]
        input_text_x = input_x + (input_width - input_size[0]) // 2
        input_text_y = input_y + (input_height + input_size[1]) // 2
        cv2.putText(
            canvas,
            input_text,
            (input_text_x, input_text_y),
            font,
            input_scale,
            input_color,
            input_thickness,
        )

        if self.piece_number_dialog_input:
            cursor_x = input_text_x + input_size[0] + 6
            cursor_y1 = input_y + 12
            cursor_y2 = input_y + input_height - 12
            cv2.line(canvas, (cursor_x, cursor_y1), (cursor_x, cursor_y2), (0, 0, 0), 2)

        button_width = 120
        button_height = 42
        button_gap = 24
        total_buttons_width = (button_width * 2) + button_gap
        buttons_x = dialog_x + (dialog_width - total_buttons_width) // 2
        button_y = dialog_y + dialog_height - button_height - 26

        self.piece_number_dialog_cancel_rect = (buttons_x, button_y, button_width, button_height)
        self.piece_number_dialog_ok_rect = (
            buttons_x + button_width + button_gap,
            button_y,
            button_width,
            button_height,
        )

        def _draw_dialog_button(rect, label, fill_color):
            is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, rect)
            is_pressed = is_hovered and self.mouse_button_down
            scale_factor = 0.95 if is_pressed else (1.06 if is_hovered else 1.0)
            x_btn, y_btn, w_btn, h_btn = self._scale_rect(rect, scale_factor)
            cv2.rectangle(canvas, (x_btn, y_btn), (x_btn + w_btn, y_btn + h_btn), fill_color, -1)
            cv2.rectangle(canvas, (x_btn, y_btn), (x_btn + w_btn, y_btn + h_btn), (0, 0, 0), 2)
            label_size = cv2.getTextSize(label, font, 0.72, 2)[0]
            label_x = x_btn + (w_btn - label_size[0]) // 2
            label_y = y_btn + (h_btn + label_size[1]) // 2
            cv2.putText(canvas, label, (label_x, label_y), font, 0.72, (255, 255, 255), 2)

        _draw_dialog_button(self.piece_number_dialog_cancel_rect, "Cancel", (100, 100, 100))
        _draw_dialog_button(self.piece_number_dialog_ok_rect, "Go", (132, 36, 2))

        return canvas

    def draw_piece_identifier_badge(self, canvas):
        controller = self._require_controller()
        identifier = controller.get_current_historic_piece_identifier()
        label = f"ID: {identifier if identifier is not None else '---'}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, 0.72, 2)[0]
        box_x, box_y = 28, 18
        box_w, box_h = text_size[0] + 34, 42
        self.piece_identifier_rect = (box_x, box_y, box_w, box_h)
        hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.piece_identifier_rect)
        fill = (175, 105, 30) if hovered else (130, 78, 25)
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), fill, -1)
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (240, 240, 240), 2)
        cv2.putText(
            canvas,
            label,
            (box_x + 17, box_y + 28),
            font,
            0.72,
            (255, 255, 255),
            2,
        )
        return canvas

    def draw_piece_identifier_dialog(self, canvas):
        dialog_width, dialog_height = 660, 320
        dialog_x = (self.width - dialog_width) // 2
        dialog_y = (self.height - dialog_height) // 2
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        canvas = cv2.addWeighted(canvas, 0.3, overlay, 0.7, 0)
        cv2.rectangle(canvas, (dialog_x, dialog_y), (dialog_x + dialog_width, dialog_y + dialog_height), (255, 255, 255), -1)
        cv2.rectangle(canvas, (dialog_x, dialog_y), (dialog_x + dialog_width, dialog_y + dialog_height), (0, 0, 0), 3)

        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "Piece Numeric ID"
        title_size = cv2.getTextSize(title, font, 0.9, 2)[0]
        cv2.putText(canvas, title, (dialog_x + (dialog_width - title_size[0]) // 2, dialog_y + 44), font, 0.9, (0, 0, 0), 2)
        helper = "Save only this piece, or continue automatic IDs from the next number."
        helper_size = cv2.getTextSize(helper, font, 0.52, 1)[0]
        cv2.putText(canvas, helper, (dialog_x + (dialog_width - helper_size[0]) // 2, dialog_y + 76), font, 0.52, (80, 80, 80), 1)

        input_w, input_h = 260, 56
        input_x, input_y = dialog_x + (dialog_width - input_w) // 2, dialog_y + 98
        cv2.rectangle(canvas, (input_x, input_y), (input_x + input_w, input_y + input_h), (250, 250, 220), -1)
        cv2.rectangle(canvas, (input_x, input_y), (input_x + input_w, input_y + input_h), (0, 0, 0), 2)
        input_text = self.piece_identifier_dialog_input or "Numeric ID"
        input_color = (0, 0, 0) if self.piece_identifier_dialog_input else (145, 145, 145)
        input_size = cv2.getTextSize(input_text, font, 0.9, 2)[0]
        input_text_x = input_x + (input_w - input_size[0]) // 2
        cv2.putText(canvas, input_text, (input_text_x, input_y + 36), font, 0.9, input_color, 2)

        button_y, button_h, button_gap = dialog_y + 190, 42, 14
        self.piece_identifier_dialog_cancel_rect = (dialog_x + 24, button_y, 120, button_h)
        self.piece_identifier_dialog_clear_rect = (dialog_x + 158, button_y, 110, button_h)
        self.piece_identifier_dialog_save_rect = (dialog_x + 282, button_y, 150, button_h)
        self.piece_identifier_dialog_continue_rect = (dialog_x + 446, button_y, 190, button_h)

        def draw_button(rect, text, color):
            x, y, w, h = rect
            hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, rect)
            fill = tuple(min(255, channel + 20) for channel in color) if hovered else color
            cv2.rectangle(canvas, (x, y), (x + w, y + h), fill, -1)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), 2)
            size = cv2.getTextSize(text, font, 0.56, 2)[0]
            cv2.putText(canvas, text, (x + (w - size[0]) // 2, y + 27), font, 0.56, (255, 255, 255), 2)

        draw_button(self.piece_identifier_dialog_cancel_rect, "Cancel", (100, 100, 100))
        draw_button(self.piece_identifier_dialog_clear_rect, "Clear ID", (75, 75, 165))
        draw_button(self.piece_identifier_dialog_save_rect, "This piece only", (70, 120, 70))
        draw_button(self.piece_identifier_dialog_continue_rect, "Save + continue auto", (132, 36, 2))
        return canvas

    def _load_trash_icon(self, size):
        if self.trash_icon is not None and self.trash_icon_size == size:
            return
        icon_path = self.TRASH_ICON_PATH
        if not self.file_manager.exists(icon_path):
            self.trash_icon = None
            self.trash_icon_size = size
            return
        icon = self.file_manager.read_image(icon_path, cv2.IMREAD_UNCHANGED)
        if icon is None:
            self.trash_icon = None
            self.trash_icon_size = size
            return
        if icon.shape[2] < 4:
            if not self._trash_icon_warned:
                print("[ICON] trash.png has no alpha channel (not transparent).")
                self._trash_icon_warned = True
            icon = cv2.cvtColor(icon, cv2.COLOR_BGR2BGRA)
            icon = self._apply_bg_key(icon)
        else:
            alpha = icon[:, :, 3]
            if np.all(alpha == 255):
                if not self._trash_icon_warned:
                    print("[ICON] trash.png alpha channel is fully opaque.")
                    self._trash_icon_warned = True
                icon = self._apply_bg_key(icon)
        icon = cv2.resize(icon, (size, size), interpolation=cv2.INTER_AREA)
        self.trash_icon = icon
        self.trash_icon_size = size

    def _estimate_bg_color(self, bgr):
        h, w = bgr.shape[:2]
        patch = 6
        corners = np.vstack([
            bgr[0:patch, 0:patch].reshape(-1, 3),
            bgr[0:patch, w - patch:w].reshape(-1, 3),
            bgr[h - patch:h, 0:patch].reshape(-1, 3),
            bgr[h - patch:h, w - patch:w].reshape(-1, 3),
        ])
        return np.median(corners, axis=0).astype(np.uint8)

    def _apply_bg_key(self, icon, threshold=30):
        bgr = icon[:, :, :3]
        bg = self._estimate_bg_color(bgr)
        diff = bgr.astype(np.int16) - bg.astype(np.int16)
        dist = np.linalg.norm(diff, axis=2)
        alpha = np.where(dist < threshold, 0, 255).astype(np.uint8)
        icon[:, :, 3] = alpha
        return icon

    def _overlay_icon(self, canvas, icon, x, y):
        if icon is None:
            return
        h, w = icon.shape[:2]
        if y < 0 or x < 0 or y + h > canvas.shape[0] or x + w > canvas.shape[1]:
            return
        if icon.shape[2] == 4:
            alpha = icon[:, :, 3] / 255.0
            for c in range(3):
                canvas[y:y + h, x:x + w, c] = (
                    (1 - alpha) * canvas[y:y + h, x:x + w, c]
                    + alpha * icon[:, :, c]
                )
        else:
            canvas[y:y + h, x:x + w] = icon[:, :, :3]

    def draw_sync_progress(self, canvas):
        """Draw modal loading screen with progress while syncing dataset."""
        if not self.sync_in_progress:
            return canvas

        dialog_width = 760
        dialog_height = 250
        dialog_x = (self.width - dialog_width) // 2
        dialog_y = (self.height - dialog_height) // 2

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.58, canvas, 0.42, 0, canvas)

        cv2.rectangle(
            canvas,
            (dialog_x, dialog_y),
            (dialog_x + dialog_width, dialog_y + dialog_height),
            (245, 245, 245),
            -1,
        )
        cv2.rectangle(
            canvas,
            (dialog_x, dialog_y),
            (dialog_x + dialog_width, dialog_y + dialog_height),
            (0, 0, 0),
            3,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        title = self.sync_progress_title or "Saving Dataset"
        title_scale = 1.1
        title_thickness = 3
        cv2.putText(
            canvas,
            title,
            (dialog_x + 35, dialog_y + 58),
            font,
            title_scale,
            (0, 0, 0),
            title_thickness,
        )

        stage_text = self.sync_stage or "Working..."
        stage_scale = 0.85
        stage_thickness = 2
        cv2.putText(
            canvas,
            stage_text,
            (dialog_x + 35, dialog_y + 100),
            font,
            stage_scale,
            (40, 40, 40),
            stage_thickness,
        )

        bar_x = dialog_x + 35
        bar_y = dialog_y + 135
        bar_w = dialog_width - 70
        bar_h = 34
        progress = max(0, min(100, int(self.sync_progress)))
        fill_w = int((bar_w * progress) / 100)

        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (220, 220, 220), -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 30, 30), 2)
        if fill_w > 0:
            cv2.rectangle(
                canvas,
                (bar_x + 2, bar_y + 2),
                (bar_x + fill_w - 2, bar_y + bar_h - 2),
                (67, 125, 22),
                -1,
            )

        pct_text = f"{progress}%"
        pct_size = cv2.getTextSize(pct_text, font, 0.9, 2)[0]
        pct_x = bar_x + (bar_w - pct_size[0]) // 2
        pct_y = bar_y + bar_h - 8
        cv2.putText(canvas, pct_text, (pct_x, pct_y), font, 0.9, (255, 255, 255), 2)

        helper_text = self.sync_progress_helper_text or "Please wait until the process finishes."
        cv2.putText(
            canvas,
            helper_text,
            (dialog_x + 35, dialog_y + 205),
            font,
            0.7,
            (30, 30, 30),
            2,
        )

        return canvas

    def draw_sync_message(self, canvas):
        """Draw completion/error message after syncing dataset."""
        if self.sync_message and self.sync_message_auto_dismiss_sec is not None:
            elapsed = time.time() - float(self.sync_message_time or 0)
            if elapsed >= float(self.sync_message_auto_dismiss_sec):
                self._clear_sync_message()
                return canvas

        if (
            not self.sync_message
            or self.show_reset_confirm
            or self.show_delete_confirm
            or self.show_rebuild_confirm
        ):
            self.sync_message_close_button_rect = None
            return canvas

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 3
        line_spacing = 44
        icon_diameter = 60
        left_pad = 45
        right_pad = 45
        text_gap = 30
        top_pad = 34
        button_gap = 28
        button_width = 150
        button_height = 52
        bottom_pad = 28

        message_block = self._prepare_wrapped_text_block(
            self.sync_message.strip(),
            font,
            font_scale,
            thickness,
            max_width=max(320, int(self.width * 0.62)),
            line_spacing=line_spacing,
            max_lines_per_paragraph=4,
        )
        dialog_width = min(
            max(520, left_pad + icon_diameter + text_gap + message_block["text_width"] + right_pad),
            int(self.width * 0.82),
        )
        body_height = max(icon_diameter, message_block["text_height"])
        dialog_height = max(
            250,
            top_pad + body_height + button_gap + button_height + bottom_pad,
        )

        canvas, dialog_x, dialog_y = self._draw_modal_frame(
            canvas,
            dialog_width,
            dialog_height,
            overlay_alpha=0.5,
        )

        icon_x = dialog_x + left_pad + (icon_diameter // 2)
        icon_y = dialog_y + top_pad + (body_height // 2)
        icon_color = (0, 0, 200) if self.sync_message_is_error else (0, 150, 0)
        cv2.circle(canvas, (icon_x, icon_y), 30, icon_color, -1)
        cv2.circle(canvas, (icon_x, icon_y), 30, (0, 0, 0), 2)

        icon_text = "!" if self.sync_message_is_error else "OK"
        icon_scale = 1.0 if self.sync_message_is_error else 1.1
        icon_thickness = 3
        icon_size = cv2.getTextSize(icon_text, font, icon_scale, icon_thickness)[0]
        icon_x_text = icon_x - (icon_size[0] // 2)
        icon_y_text = icon_y + (icon_size[1] // 2)
        cv2.putText(
            canvas,
            icon_text,
            (icon_x_text, icon_y_text),
            font,
            icon_scale,
            (255, 255, 255),
            icon_thickness,
        )

        text_x = dialog_x + left_pad + icon_diameter + text_gap
        text_top = dialog_y + top_pad + max(0, (body_height - message_block["text_height"]) // 2)
        first_line_y = text_top + message_block["line_height"]
        self._draw_text_lines(
            canvas,
            message_block["lines"],
            text_x,
            first_line_y,
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            line_spacing=line_spacing,
        )

        button_x = dialog_x + (dialog_width - button_width) // 2
        button_y = dialog_y + dialog_height - button_height - bottom_pad
        self.sync_message_close_button_rect = (
            button_x,
            button_y,
            button_width,
            button_height,
        )
        button_color = (0, 0, 200) if self.sync_message_is_error else (0, 130, 0)
        self._draw_modal_button(
            canvas,
            self.sync_message_close_button_rect,
            "Cerrar",
            button_color,
            font_scale=0.75,
            thickness=2,
        )

        return canvas

    def draw_toast_message(self, canvas):
        """Draw a short-lived toast for lightweight UI feedback."""
        if not self.toast_message:
            return canvas

        if (time.time() - self.toast_message_time) > self.toast_message_duration_sec:
            self.toast_message = ""
            return canvas

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 2
        text = self.toast_message
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        pad_x = 18
        pad_y = 12
        box_w = text_size[0] + pad_x * 2
        box_h = text_size[1] + pad_y * 2
        box_x = (self.width - box_w) // 2
        box_y = 82

        color = (40, 60, 190) if self.toast_message_is_error else (40, 120, 40)
        overlay = canvas.copy()
        cv2.rectangle(
            overlay,
            (box_x, box_y),
            (box_x + box_w, box_y + box_h),
            color,
            -1,
        )
        cv2.addWeighted(overlay, 0.9, canvas, 0.1, 0, canvas)
        cv2.rectangle(
            canvas,
            (box_x, box_y),
            (box_x + box_w, box_y + box_h),
            (255, 255, 255),
            2,
        )

        text_x = box_x + pad_x
        text_y = box_y + pad_y + text_size[1]
        cv2.putText(
            canvas,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )
        return canvas

    def draw_historic_jsn_banner(self, canvas):
        """Draw the historic JSN as a highlighted, clickable copy target."""
        jsn = self._get_current_historic_jsn()
        self.historic_jsn_rect = None
        if not jsn:
            return canvas

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.15
        thickness = 2
        label_text = "JSN"
        value_text = str(jsn)
        gap = 16
        pad_x = 28
        pad_y = 16

        label_size = cv2.getTextSize(label_text, font, 0.85, 2)[0]
        value_size = cv2.getTextSize(value_text, font, font_scale, thickness)[0]
        box_w = label_size[0] + gap + value_size[0] + pad_x * 2
        box_h = max(label_size[1], value_size[1]) + pad_y * 2
        box_x = (self.width - box_w) // 2
        box_y = 18
        self.historic_jsn_rect = (box_x, box_y, box_w, box_h)

        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.historic_jsn_rect)
        fill_color = (200, 120, 30) if is_hovered else (168, 104, 30)
        border_color = (255, 255, 255) if is_hovered else (230, 230, 230)

        overlay = canvas.copy()
        cv2.rectangle(
            overlay,
            (box_x, box_y),
            (box_x + box_w, box_y + box_h),
            fill_color,
            -1,
        )
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.rectangle(
            canvas,
            (box_x, box_y),
            (box_x + box_w, box_y + box_h),
            border_color,
            2,
        )

        baseline_y = box_y + (box_h + value_size[1]) // 2 - 2
        label_x = box_x + pad_x
        value_x = label_x + label_size[0] + gap

        cv2.putText(canvas, label_text, (label_x, baseline_y), font, 0.85, (245, 245, 245), 2)
        cv2.putText(canvas, value_text, (value_x, baseline_y), font, font_scale, (255, 255, 255), thickness)
        return canvas

    def draw_reset_progress(self, canvas):
        """Draw modal loading screen with progress while resetting dataset."""
        if not self.reset_in_progress:
            return canvas

        font = cv2.FONT_HERSHEY_SIMPLEX
        title = self.reset_progress_title or "Resetting Dataset"
        stage_text = self.reset_stage or "Working..."
        helper_text = self.reset_progress_helper_text or "Please wait until the process finishes."
        title_scale = 1.1
        title_thickness = 3
        stage_scale = 0.85
        stage_thickness = 2
        helper_scale = 0.7
        helper_thickness = 2
        left_pad = 35
        right_pad = 35
        top_pad = 28
        bottom_pad = 24
        title_gap = 18
        stage_gap = 16
        progress_gap = 18
        helper_gap = 18
        title_spacing = 42
        stage_spacing = 34
        helper_spacing = 30

        title_block = self._prepare_wrapped_text_block(
            title,
            font,
            title_scale,
            title_thickness,
            max_width=max(320, int(self.width * 0.5)),
            line_spacing=title_spacing,
            max_lines_per_paragraph=2,
        )
        stage_block = self._prepare_wrapped_text_block(
            stage_text,
            font,
            stage_scale,
            stage_thickness,
            max_width=max(360, int(self.width * 0.62)),
            line_spacing=stage_spacing,
            max_lines_per_paragraph=3,
        )
        helper_block = self._prepare_wrapped_text_block(
            helper_text,
            font,
            helper_scale,
            helper_thickness,
            max_width=max(360, int(self.width * 0.62)),
            line_spacing=helper_spacing,
            max_lines_per_paragraph=3,
        )

        content_width = max(
            title_block["text_width"],
            stage_block["text_width"],
            helper_block["text_width"],
            420,
        )
        dialog_width = min(
            max(760, left_pad + content_width + right_pad),
            int(self.width * 0.86),
        )
        bar_w = dialog_width - (left_pad + right_pad)
        bar_h = 34
        content_height = (
            title_block["text_height"]
            + title_gap
            + stage_block["text_height"]
            + progress_gap
            + bar_h
            + helper_gap
            + helper_block["text_height"]
        )
        dialog_height = max(250, top_pad + content_height + bottom_pad)

        canvas, dialog_x, dialog_y = self._draw_modal_frame(
            canvas,
            dialog_width,
            dialog_height,
            overlay_alpha=0.58,
            fill_color=(245, 245, 245),
        )

        title_y = dialog_y + top_pad + title_block["line_height"]
        self._draw_text_lines(
            canvas,
            title_block["lines"],
            dialog_x + left_pad,
            title_y,
            font,
            title_scale,
            (0, 0, 0),
            title_thickness,
            line_spacing=title_spacing,
        )

        stage_y = title_y + title_block["text_height"] + title_gap
        self._draw_text_lines(
            canvas,
            stage_block["lines"],
            dialog_x + left_pad,
            stage_y,
            font,
            stage_scale,
            (40, 40, 40),
            stage_thickness,
            line_spacing=stage_spacing,
        )

        bar_x = dialog_x + left_pad
        bar_y = stage_y + stage_block["text_height"] + progress_gap
        progress = max(0, min(100, int(self.reset_progress)))
        fill_w = int((bar_w * progress) / 100)

        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (220, 220, 220), -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 30, 30), 2)
        if fill_w > 0:
            cv2.rectangle(
                canvas,
                (bar_x + 2, bar_y + 2),
                (bar_x + fill_w - 2, bar_y + bar_h - 2),
                (67, 125, 22),
                -1,
            )

        pct_text = f"{progress}%"
        pct_size = cv2.getTextSize(pct_text, font, 0.9, 2)[0]
        pct_x = bar_x + (bar_w - pct_size[0]) // 2
        pct_y = bar_y + bar_h - 8
        cv2.putText(canvas, pct_text, (pct_x, pct_y), font, 0.9, (255, 255, 255), 2)

        helper_y = bar_y + bar_h + helper_gap + helper_block["line_height"]
        self._draw_text_lines(
            canvas,
            helper_block["lines"],
            dialog_x + left_pad,
            helper_y,
            font,
            helper_scale,
            (30, 30, 30),
            helper_thickness,
            line_spacing=helper_spacing,
        )

        return canvas

    def draw_db_block_dialog(self, canvas):
        """Draw blocking dialog shown while PostgreSQL is unavailable."""
        if not self.db_blocking:
            return canvas

        font = cv2.FONT_HERSHEY_SIMPLEX
        message = (
            self.db_block_message
            or "PostgreSQL is disconnected. Start postgres and wait for automatic reconnect."
        )
        title_scale = 1.0
        title_thickness = 3
        title_spacing = 42
        body_scale = 0.8
        body_thickness = 2
        body_spacing = 36
        icon_diameter = 68
        left_pad = 45
        right_pad = 40
        text_gap = 28
        top_pad = 34
        title_gap = 24
        bottom_pad = 34

        title_block = self._prepare_wrapped_text_block(
            "Database connection required",
            font,
            title_scale,
            title_thickness,
            max_width=max(320, int(self.width * 0.56)),
            line_spacing=title_spacing,
            max_lines_per_paragraph=2,
        )
        body_block = self._prepare_wrapped_text_block(
            [
                message,
                "User actions are locked until DB connection is restored.",
                "Auto-reconnect is running...",
            ],
            font,
            body_scale,
            body_thickness,
            max_width=max(340, int(self.width * 0.62)),
            line_spacing=body_spacing,
            max_lines_per_paragraph=3,
        )

        text_width = max(title_block["text_width"], body_block["text_width"])
        dialog_width = min(
            max(700, left_pad + icon_diameter + text_gap + text_width + right_pad),
            int(self.width * 0.85),
        )
        text_height = title_block["text_height"] + title_gap + body_block["text_height"]
        body_height = max(icon_diameter, text_height)
        dialog_height = max(250, top_pad + body_height + bottom_pad)

        canvas, dialog_x, dialog_y = self._draw_modal_frame(
            canvas,
            dialog_width,
            dialog_height,
            overlay_alpha=0.62,
        )

        icon_x = dialog_x + left_pad + (icon_diameter // 2)
        icon_y = dialog_y + top_pad + (icon_diameter // 2)
        cv2.circle(canvas, (icon_x, icon_y), icon_diameter // 2, (0, 0, 200), -1)
        cv2.circle(canvas, (icon_x, icon_y), icon_diameter // 2, (0, 0, 0), 2)
        cv2.putText(canvas, "!", (icon_x - 11, icon_y + 15), font, 1.8, (255, 255, 255), 4)

        text_x = dialog_x + left_pad + icon_diameter + text_gap
        title_y = dialog_y + top_pad + title_block["line_height"]
        self._draw_text_lines(
            canvas,
            title_block["lines"],
            text_x,
            title_y,
            font,
            title_scale,
            (0, 0, 0),
            title_thickness,
            line_spacing=title_spacing,
        )

        body_y = title_y + title_block["text_height"] + title_gap
        self._draw_text_lines(
            canvas,
            body_block["lines"],
            text_x,
            body_y,
            font,
            body_scale,
            (20, 20, 20),
            body_thickness,
            line_spacing=body_spacing,
        )

        return canvas
    
    def draw_reset_confirmation_dialog(self, canvas):
        """Draw reset confirmation dialog"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.95
        thickness = 2
        line_spacing = 38
        icon_diameter = 60
        left_pad = 45
        right_pad = 35
        text_gap = 28
        top_pad = 32
        button_gap = 28
        bottom_pad = 30

        body_block = self._prepare_wrapped_text_block(
            [
                "Warning: This will reset DB and delete images in historic, annotated, classified, and final_classification.",
                "Confirm reset operation?",
            ],
            font,
            font_scale,
            thickness,
            max_width=max(320, int(self.width * 0.52)),
            line_spacing=line_spacing,
            max_lines_per_paragraph=3,
        )

        button_width = 150
        button_height = 50
        button_spacing = 30
        dialog_width = min(
            max(560, left_pad + icon_diameter + text_gap + body_block["text_width"] + right_pad),
            int(self.width * 0.8),
        )
        body_height = max(icon_diameter, body_block["text_height"])
        dialog_height = top_pad + body_height + button_gap + button_height + bottom_pad

        canvas, dialog_x, dialog_y = self._draw_modal_frame(
            canvas,
            dialog_width,
            dialog_height,
            overlay_alpha=0.5,
        )

        icon_x = dialog_x + left_pad + (icon_diameter // 2)
        icon_y = dialog_y + top_pad + (icon_diameter // 2)
        cv2.circle(canvas, (icon_x, icon_y), icon_diameter // 2, (0, 0, 200), -1)
        cv2.circle(canvas, (icon_x, icon_y), icon_diameter // 2, (0, 0, 0), 2)
        cv2.putText(canvas, "!", (icon_x - 10, icon_y + 15), font, 2.0, (255, 255, 255), 4)

        text_x = dialog_x + left_pad + icon_diameter + text_gap
        text_y = dialog_y + top_pad + body_block["line_height"]
        self._draw_text_lines(
            canvas,
            body_block["lines"],
            text_x,
            text_y,
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            line_spacing=line_spacing,
        )

        buttons_y = dialog_y + dialog_height - button_height - bottom_pad
        cancel_x = dialog_x + (dialog_width // 2) - button_width - (button_spacing // 2)
        self.reset_cancel_button_rect = (cancel_x, buttons_y, button_width, button_height)
        confirm_x = dialog_x + (dialog_width // 2) + (button_spacing // 2)
        self.reset_confirm_button_rect = (confirm_x, buttons_y, button_width, button_height)
        self._draw_modal_button(canvas, self.reset_cancel_button_rect, "CANCEL", (150, 150, 150))
        self._draw_modal_button(canvas, self.reset_confirm_button_rect, "CONFIRM", (0, 0, 200))
        
        return canvas

    def draw_rebuild_confirmation_dialog(self, canvas):
        """Draw rebuild-from-historic confirmation dialog."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.82
        thickness = 2
        line_spacing = 36
        icon_diameter = 60
        left_pad = 45
        right_pad = 35
        text_gap = 28
        top_pad = 32
        button_gap = 28
        bottom_pad = 30

        body_block = self._prepare_wrapped_text_block(
            [
                "Rebuild the app database from historic images?",
                "This clears app DB tables before rebuilding from the historic folder.",
                "Classified and final folders will be preserved.",
                "Historic images will be preserved.",
            ],
            font,
            font_scale,
            thickness,
            max_width=max(360, int(self.width * 0.58)),
            line_spacing=line_spacing,
            max_lines_per_paragraph=3,
        )

        button_width = 170
        button_height = 52
        button_spacing = 30
        dialog_width = min(
            max(700, left_pad + icon_diameter + text_gap + body_block["text_width"] + right_pad),
            int(self.width * 0.86),
        )
        body_height = max(icon_diameter, body_block["text_height"])
        dialog_height = top_pad + body_height + button_gap + button_height + bottom_pad

        canvas, dialog_x, dialog_y = self._draw_modal_frame(
            canvas,
            dialog_width,
            dialog_height,
            overlay_alpha=0.5,
        )

        icon_x = dialog_x + left_pad + (icon_diameter // 2)
        icon_y = dialog_y + top_pad + (icon_diameter // 2)
        cv2.circle(canvas, (icon_x, icon_y), icon_diameter // 2, (0, 0, 200), -1)
        cv2.circle(canvas, (icon_x, icon_y), icon_diameter // 2, (0, 0, 0), 2)
        cv2.putText(canvas, "!", (icon_x - 10, icon_y + 15), font, 2.0, (255, 255, 255), 4)

        text_x = dialog_x + left_pad + icon_diameter + text_gap
        text_y = dialog_y + top_pad + body_block["line_height"]
        self._draw_text_lines(
            canvas,
            body_block["lines"],
            text_x,
            text_y,
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            line_spacing=line_spacing,
        )

        buttons_y = dialog_y + dialog_height - button_height - bottom_pad

        cancel_x = dialog_x + (dialog_width // 2) - button_width - (button_spacing // 2)
        self.rebuild_cancel_button_rect = (cancel_x, buttons_y, button_width, button_height)

        confirm_x = dialog_x + (dialog_width // 2) + (button_spacing // 2)
        self.rebuild_confirm_button_rect = (confirm_x, buttons_y, button_width, button_height)
        self._draw_modal_button(canvas, self.rebuild_cancel_button_rect, "CANCEL", (150, 150, 150))
        self._draw_modal_button(canvas, self.rebuild_confirm_button_rect, "REBUILD", (0, 0, 200))

        return canvas

    def draw_delete_confirmation_dialog(self, canvas):
        """Draw delete-piece confirmation dialog"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        jsn = self._get_current_historic_jsn() or "N/A"
        font_scale = 0.85
        thickness = 2
        line_spacing = 38
        icon_diameter = 60
        left_pad = 45
        right_pad = 35
        text_gap = 28
        top_pad = 32
        button_gap = 28
        bottom_pad = 30

        body_block = self._prepare_wrapped_text_block(
            [
                f"Delete current piece? JSN: {jsn}",
                "This will delete local and remote images permanently.",
            ],
            font,
            font_scale,
            thickness,
            max_width=max(320, int(self.width * 0.54)),
            line_spacing=line_spacing,
            max_lines_per_paragraph=3,
        )

        button_width = 150
        button_height = 50
        button_spacing = 30
        dialog_width = min(
            max(620, left_pad + icon_diameter + text_gap + body_block["text_width"] + right_pad),
            int(self.width * 0.82),
        )
        body_height = max(icon_diameter, body_block["text_height"])
        dialog_height = top_pad + body_height + button_gap + button_height + bottom_pad

        canvas, dialog_x, dialog_y = self._draw_modal_frame(
            canvas,
            dialog_width,
            dialog_height,
            overlay_alpha=0.5,
        )

        icon_x = dialog_x + left_pad + (icon_diameter // 2)
        icon_y = dialog_y + top_pad + (icon_diameter // 2)
        cv2.circle(canvas, (icon_x, icon_y), icon_diameter // 2, (60, 60, 60), -1)
        cv2.circle(canvas, (icon_x, icon_y), icon_diameter // 2, (0, 0, 0), 2)
        cv2.putText(canvas, "X", (icon_x - 12, icon_y + 15), font, 1.6, (255, 255, 255), 3)

        text_x = dialog_x + left_pad + icon_diameter + text_gap
        text_y = dialog_y + top_pad + body_block["line_height"]
        self._draw_text_lines(
            canvas,
            body_block["lines"],
            text_x,
            text_y,
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            line_spacing=line_spacing,
        )

        buttons_y = dialog_y + dialog_height - button_height - bottom_pad
        cancel_x = dialog_x + (dialog_width // 2) - button_width - (button_spacing // 2)
        self.delete_cancel_button_rect = (cancel_x, buttons_y, button_width, button_height)
        confirm_x = dialog_x + (dialog_width // 2) + (button_spacing // 2)
        self.delete_confirm_button_rect = (confirm_x, buttons_y, button_width, button_height)
        self._draw_modal_button(canvas, self.delete_cancel_button_rect, "CANCEL", (150, 150, 150))
        self._draw_modal_button(canvas, self.delete_confirm_button_rect, "CONFIRM", (0, 0, 200))

        return canvas
    
    def draw_no_images_dialog(self, canvas):
        """Draw no images available dialog"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        message_text = self.no_images_dialog_message or "No images available"
        font_scale = 0.9
        thickness = 2
        line_spacing = 38
        icon_diameter = 60
        left_pad = 45
        right_pad = 35
        text_gap = 28
        top_pad = 30
        button_gap = 26
        bottom_pad = 25

        body_block = self._prepare_wrapped_text_block(
            message_text,
            font,
            font_scale,
            thickness,
            max_width=max(280, int(self.width * 0.45)),
            line_spacing=line_spacing,
            max_lines_per_paragraph=4,
        )

        button_width = 120
        button_height = 50
        dialog_width = min(
            max(440, left_pad + icon_diameter + text_gap + body_block["text_width"] + right_pad),
            int(self.width * 0.72),
        )
        body_height = max(icon_diameter, body_block["text_height"])
        dialog_height = top_pad + body_height + button_gap + button_height + bottom_pad

        canvas, dialog_x, dialog_y = self._draw_modal_frame(
            canvas,
            dialog_width,
            dialog_height,
            overlay_alpha=0.5,
        )

        icon_x = dialog_x + left_pad + (icon_diameter // 2)
        icon_y = dialog_y + top_pad + (icon_diameter // 2)
        cv2.circle(canvas, (icon_x, icon_y), icon_diameter // 2, (0, 0, 200), -1)
        cv2.circle(canvas, (icon_x, icon_y), icon_diameter // 2, (0, 0, 0), 2)
        cv2.putText(canvas, "!", (icon_x - 10, icon_y + 15), font, 1.8, (255, 255, 255), 3)

        text_x = dialog_x + left_pad + icon_diameter + text_gap
        text_y = dialog_y + top_pad + body_block["line_height"]
        self._draw_text_lines(
            canvas,
            body_block["lines"],
            text_x,
            text_y,
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            line_spacing=line_spacing,
        )

        button_x = dialog_x + (dialog_width - button_width) // 2
        button_y = dialog_y + dialog_height - button_height - bottom_pad

        self.no_images_ok_button_rect = (button_x, button_y, button_width, button_height)
        self._draw_modal_button(canvas, self.no_images_ok_button_rect, "OK", (132, 36, 2), font_scale=0.9)

        return canvas
    
    def draw_next_button(self, canvas):
        """Draw next arrow button (right)"""
        button_width = 100
        button_height = 100
        margin = 0  # Attached to edge
        
        # Button on far right (vertical center)
        x_next = self.width - button_width - margin
        y_next = (self.height - button_height) // 2
        
        self.next_button_rect = (x_next, y_next, button_width, button_height)
        
        # Check if button is hovered or pressed
        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.next_button_rect)
        is_pressed = is_hovered and self.mouse_button_down
        
        # Scale button on hover
        scale_factor = 0.95 if is_pressed else (1.08 if is_hovered else 1.0)
        scaled_rect = self._scale_rect(self.next_button_rect, scale_factor)
        x_draw, y_draw, w_draw, h_draw = scaled_rect
        
        # Draw circle
        center_x = x_draw + w_draw // 2
        center_y = y_draw + h_draw // 2
        radius = int(40 * scale_factor)
        circle_color = (132, 36, 2)
        border_color = (0, 0, 0)
        border_width = 2
        cv2.circle(canvas, (center_x, center_y), radius, circle_color, -1)
        cv2.circle(canvas, (center_x, center_y), radius, border_color, border_width)
        
        # Draw right arrow (triangle)
        arrow_points = np.array([
            [center_x - 15, center_y - 25],
            [center_x + 20, center_y],
            [center_x - 15, center_y + 25]
        ], np.int32)
        cv2.fillPoly(canvas, [arrow_points], (255, 255, 255))
        
        return canvas
    
    def draw_prev_button(self, canvas):
        """Draw previous arrow button (left)"""
        button_width = 100
        button_height = 100
        margin = 0  # Attached to edge
        
        # Button on far left (vertical center)
        x_prev = margin
        y_prev = (self.height - button_height) // 2
        
        self.prev_button_rect = (x_prev, y_prev, button_width, button_height)
        
        # Check if button is hovered or pressed
        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.prev_button_rect)
        is_pressed = is_hovered and self.mouse_button_down
        
        # Scale button on hover
        scale_factor = 0.95 if is_pressed else (1.08 if is_hovered else 1.0)
        scaled_rect = self._scale_rect(self.prev_button_rect, scale_factor)
        x_draw, y_draw, w_draw, h_draw = scaled_rect
        
        # Draw circle
        center_x = x_draw + w_draw // 2
        center_y = y_draw + h_draw // 2
        radius = int(40 * scale_factor)
        circle_color = (132, 36, 2)
        border_color = (0, 0, 0)
        border_width = 2
        cv2.circle(canvas, (center_x, center_y), radius, circle_color, -1)
        cv2.circle(canvas, (center_x, center_y), radius, border_color, border_width)
        
        # Draw left arrow (triangle)
        arrow_points = np.array([
            [center_x + 15, center_y - 25],
            [center_x - 20, center_y],
            [center_x + 15, center_y + 25]
        ], np.int32)
        cv2.fillPoly(canvas, [arrow_points], (255, 255, 255))
        
        return canvas
    
    def draw_search_elements(self, canvas):
        """Draw search input field and search button"""
        # Position in upper right corner
        input_width = 320  # Width for exactly 21 numbers with smaller font
        input_height = 45
        button_size = 45  # Square button same height as input
        margin_right = 150
        margin_top = 30
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Input field position (right side with margin)
        x_input = self.width - input_width - button_size - margin_right
        y_input = margin_top
        
        self.search_input_rect = (x_input, y_input, input_width, input_height)
        
        # Draw input field
        bg_color = (255, 255, 200) if self.search_active else (255, 255, 255)
        cv2.rectangle(canvas, (x_input, y_input), (x_input + input_width, y_input + input_height),
                     bg_color, -1)  # White/yellow background
        cv2.rectangle(canvas, (x_input, y_input), (x_input + input_width, y_input + input_height),
                     (0, 0, 0), 2)  # Black border
        
        # Display search text or placeholder
        display_text = self.search_jsn if self.search_jsn else "Enter JSN..."
        text_color = (0, 0, 0) if self.search_jsn else (150, 150, 150)
        
        text_size = cv2.getTextSize(display_text, font, font_scale, thickness)[0]
        text_x = x_input + 10  # Left padding
        text_y = y_input + (input_height + text_size[1]) // 2
        
        # Truncate text if too long
        max_text_width = input_width - 20
        if text_size[0] > max_text_width:
            # Truncate from left to show most recent characters
            while text_size[0] > max_text_width and len(display_text) > 0:
                display_text = display_text[1:]
                text_size = cv2.getTextSize(display_text, font, font_scale, thickness)[0]
        
        cv2.putText(canvas, display_text, (text_x, text_y), font, font_scale,
                   text_color, thickness)
        
        # Show cursor if active
        if self.search_active:
            cursor_x = text_x + cv2.getTextSize(self.search_jsn, font, font_scale, thickness)[0][0] + 5
            cursor_y1 = y_input + 10
            cursor_y2 = y_input + input_height - 10
            cv2.line(canvas, (cursor_x, cursor_y1), (cursor_x, cursor_y2), (0, 0, 0), 2)
        
        # Draw suggestions dropdown if search is active and there are suggestions
        self.suggestion_rects = []
        if self.search_active and self.filtered_suggestions:
            suggestion_height = 35
            suggestion_y = y_input + input_height
            
            for idx, jsn_suggestion in enumerate(self.filtered_suggestions):
                # Background color for suggestion
                if idx == self.selected_suggestion_idx:
                    bg_color = (200, 220, 255)  # Light blue for selected
                else:
                    bg_color = (245, 245, 245)  # Light gray
                
                suggestion_rect = (x_input, suggestion_y, input_width, suggestion_height)
                self.suggestion_rects.append((suggestion_rect, jsn_suggestion))
                
                # Draw suggestion background
                cv2.rectangle(canvas, (x_input, suggestion_y), 
                            (x_input + input_width, suggestion_y + suggestion_height),
                            bg_color, -1)
                cv2.rectangle(canvas, (x_input, suggestion_y), 
                            (x_input + input_width, suggestion_y + suggestion_height),
                            (0, 0, 0), 1)
                
                # Draw suggestion text
                suggestion_text_y = suggestion_y + (suggestion_height + text_size[1]) // 2
                cv2.putText(canvas, jsn_suggestion, (text_x, suggestion_text_y), font, font_scale,
                           (0, 0, 0), thickness)
                
                suggestion_y += suggestion_height
        
        # Search button position (attached to right of input field)
        x_button = x_input + input_width
        y_button = margin_top
        
        self.search_button_rect = (x_button, y_button, button_size, button_size)
        
        # Draw search button background
        cv2.rectangle(canvas, (x_button, y_button), (x_button + button_size, y_button + button_size),
                     (0, 150, 0), -1)  # Green button
        cv2.rectangle(canvas, (x_button, y_button), (x_button + button_size, y_button + button_size),
                     (0, 0, 0), 2)
        
        # Draw magnifying glass (lupa)
        center_x = x_button + button_size // 2
        center_y = y_button + button_size // 2
        
        # Circle of the magnifying glass
        circle_radius = 12
        cv2.circle(canvas, (center_x - 3, center_y - 3), circle_radius, (255, 255, 255), 3)
        
        # Handle of the magnifying glass
        handle_start_x = center_x + 6
        handle_start_y = center_y + 6
        handle_end_x = center_x + 14
        handle_end_y = center_y + 14
        cv2.line(canvas, (handle_start_x, handle_start_y), (handle_end_x, handle_end_y), (255, 255, 255), 3)
        
        return canvas

    
    def show(self):
        """Show the display window occupying full screen"""
        if self.image is None:
            self.create_white_display()
        
        # Create window with WND_PROP_FULLSCREEN flag
        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Register mouse callback
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        cv2.imshow(self.window_name, self.image)

        
        import time
        self.last_refresh_time = time.time()
        
        # Loop until it's time to update
        while True:
            self._pump_historic_image_report_dialog()
            if hasattr(cv2, "waitKeyEx"):
                key_ex = cv2.waitKeyEx(100)
            else:
                key_ex = cv2.waitKey(100)
            key = (key_ex & 0xFF) if key_ex != -1 else -1

            if self._check_stats_long_press():
                return True

            if self.db_blocking:
                if time.time() - self.last_refresh_time >= self.refresh_interval:
                    return True
                continue

            left_arrow_keys = {2424832, 81, ord('a'), ord('A')}
            right_arrow_keys = {2555904, 83, ord('d'), ord('D')}
            up_arrow_keys = {2490368, 82}
            down_arrow_keys = {2621440, 84}

            if self.show_piece_number_dialog and not self.show_no_images_dialog and key_ex != -1:
                if key == 27:
                    self._emit_action("close_piece_number_dialog")
                    return True
                if key == 13:
                    self._emit_action("submit_piece_number_dialog")
                    return True
                if key == 8:
                    self._emit_action("piece_number_backspace")
                    return True
                if 48 <= key <= 57:
                    self._emit_action("piece_number_append_digit", digit=chr(key))
                    return True

            if self.show_piece_identifier_dialog and not self.show_no_images_dialog and key_ex != -1:
                if key == 27:
                    self._emit_action("close_piece_identifier_dialog")
                    return True
                if key == 13:
                    self._emit_action("save_piece_identifier_only")
                    return True
                if key == 8:
                    self._emit_action("piece_identifier_backspace")
                    return True
                if 48 <= key <= 57:
                    self._emit_action("piece_identifier_append_digit", digit=chr(key))
                    return True

            # Handle keyboard input when search is active
            if self.search_active and key_ex != -1:
                if key == 27:  # ESC key
                    self._emit_action("search_cancel")
                    return True
                elif key == 13:  # ENTER key
                    self._emit_action("search_submit")
                    return True
                elif key == 22:  # CTRL+V
                    self._paste_clipboard_into_search()
                    return True
                elif key == 8:  # BACKSPACE key
                    self._emit_action("search_backspace")
                    return True
                elif key_ex in up_arrow_keys:  # UP arrow key
                    self._emit_action("search_move_up")
                    return True
                elif key_ex in down_arrow_keys:  # DOWN arrow key
                    self._emit_action("search_move_down")
                    return True
                elif 48 <= key <= 57:  # Only numeric characters (0-9)
                    self._emit_action("search_append_digit", digit=chr(key))
                    return True

            # Historic navigation with keyboard arrows (left/right)
            if (
                key_ex != -1
                and self.historic_mode
                and not self.search_active
                and not self.sync_in_progress
                and not self.reset_in_progress
                and not self.show_stats_class_modal
                and not self.show_piece_date_dialog
                and not self.show_piece_number_dialog
                and not self.show_piece_identifier_dialog
                and not self.show_no_images_dialog
            ):
                if key in {3, ord('c'), ord('C')}:
                    self._copy_current_historic_jsn()
                    return True
                if key_ex in left_arrow_keys:
                    self._emit_action("prev_historic_batch")
                    return True
                if key_ex in right_arrow_keys:
                    self._emit_action("next_historic_batch")
                    return True
            
            # Check if it's time to update
            if time.time() - self.last_refresh_time >= self.refresh_interval:
                # If in historic mode, reload images to update counter
                if self.historic_mode:
                    now = time.time()
                    if now - self._last_historic_auto_refresh >= self.historic_auto_refresh_interval:
                        self.enter_historic_mode()
                        self._last_historic_auto_refresh = now
                return True  # Signal to update
        
    def close(self):
        """Close the display window"""
        self._close_historic_image_report_dialog()
        self._close_historic_verdict_analysis_dialog(confirm=False)

        def _stop_worker(process_attr, stop_attr):
            stop_event = getattr(self, stop_attr, None)
            if stop_event is not None:
                try:
                    stop_event.set()
                except Exception:
                    pass

            process = getattr(self, process_attr, None)
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

            setattr(self, process_attr, None)
            setattr(self, stop_attr, None)

        _stop_worker("download_process", "download_stop_event")
        _stop_worker("annotated_download_process", "annotated_download_stop_event")

        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        
    def set_color(self, color):
        """Change display color - color in BGR format (Blue, Green, Red)"""
        self.image = np.ones((self.height, self.width, 3), dtype=np.uint8) * np.array(color, dtype=np.uint8)

    def _get_background_canvas(self):
        """Return a writable background canvas using a cached resized template."""
        background_path = self.BACKGROUND_IMAGE_PATH
        target_size = (self.width, self.height)

        if self.file_manager.exists(background_path):
            current_mtime = None
            try:
                current_mtime = self.file_manager.getmtime(background_path)
            except Exception:
                pass

            needs_reload = (
                self._background_cache is None
                or self._background_cache_mtime != current_mtime
                or self._background_cache_size != target_size
            )
            if needs_reload:
                bg = self.file_manager.read_image(background_path)
                if bg is not None:
                    self._background_cache = cv2.resize(bg, target_size)
                    self._background_cache_mtime = current_mtime
                    self._background_cache_size = target_size

            if self._background_cache is not None:
                if (
                    self._canvas_buffer is None
                    or self._canvas_buffer.shape != self._background_cache.shape
                    or self._canvas_buffer.dtype != self._background_cache.dtype
                ):
                    self._canvas_buffer = np.empty_like(self._background_cache)
                np.copyto(self._canvas_buffer, self._background_cache)
                return self._canvas_buffer

        expected_shape = (self.height, self.width, 3)
        if (
            self._canvas_buffer is None
            or self._canvas_buffer.shape != expected_shape
            or self._canvas_buffer.dtype != np.uint8
        ):
            self._canvas_buffer = np.empty(expected_shape, dtype=np.uint8)
        self._canvas_buffer.fill(255)
        return self._canvas_buffer

    def _image_cache_key(self, img_path, target_size=None):
        if target_size is None:
            size_key = None
        elif isinstance(target_size, (tuple, list)):
            size_key = tuple(int(value) for value in target_size[:2])
        else:
            size_key = (int(target_size), int(target_size))
        return (os.fspath(img_path), size_key)

    def clear_cached_image(self, img_path):
        """Remove all cached sizes for one image path."""
        path_key = os.fspath(img_path)
        for cache_key in list(self._image_cache.keys()):
            if isinstance(cache_key, tuple) and cache_key[0] == path_key:
                self._image_cache.pop(cache_key, None)

    def _resize_for_cache(self, img, target_size):
        if img is None or target_size is None:
            return img
        if isinstance(target_size, (tuple, list)):
            target_w, target_h = [int(value) for value in target_size[:2]]
        else:
            target_w = target_h = int(target_size)
        if target_w <= 0 or target_h <= 0:
            return img
        if img.shape[1] == target_w and img.shape[0] == target_h:
            return img
        interpolation = (
            cv2.INTER_AREA
            if img.shape[0] > target_h or img.shape[1] > target_w
            else cv2.INTER_LINEAR
        )
        return cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    def _get_cached_image(self, img_path, target_size=None):
        """Read an image with a small LRU cache keyed by path, mtime, and rendered size."""
        cache_key = self._image_cache_key(img_path, target_size)
        try:
            current_mtime = self.file_manager.getmtime(img_path)
        except Exception:
            self.clear_cached_image(img_path)
            return None

        cached = self._image_cache.get(cache_key)
        if cached and cached[0] == current_mtime:
            self._image_cache.move_to_end(cache_key)
            return cached[1]

        img = self.file_manager.read_image(img_path)
        if img is None:
            self.clear_cached_image(img_path)
            return None

        img = self._resize_for_cache(img, target_size)
        self._image_cache[cache_key] = (current_mtime, img)
        self._image_cache.move_to_end(cache_key)
        while len(self._image_cache) > self._image_cache_max_items:
            self._image_cache.popitem(last=False)
        return img

    def _get_cached_model_overlays(self, image_paths):
        image_names = tuple(self.file_manager.basename(path) for path in image_paths or [])
        if not image_names:
            return {}

        now = time.monotonic()
        if (
            self._model_overlay_cache_key == image_names
            and now - self._model_overlay_cache_time < self._model_overlay_cache_ttl
        ):
            return self._model_overlay_cache_value

        try:
            overlays = self.get_model_overlays_for_images(image_names) or {}
        except Exception as exc:
            print(f"Error loading model overlays: {exc}")
            overlays = {}

        self._model_overlay_cache_key = image_names
        self._model_overlay_cache_value = overlays
        self._model_overlay_cache_time = now
        return overlays

    def _decode_overlay_coordinates(self, coordinates):
        if coordinates is None:
            return None
        if isinstance(coordinates, str):
            try:
                return json.loads(coordinates)
            except Exception:
                return None
        return coordinates

    def _as_float(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _bbox_points_from_dict(self, data):
        x1 = self._as_float(data.get("x1", data.get("xmin", data.get("left"))))
        y1 = self._as_float(data.get("y1", data.get("ymin", data.get("top"))))
        x2 = self._as_float(data.get("x2", data.get("xmax", data.get("right"))))
        y2 = self._as_float(data.get("y2", data.get("ymax", data.get("bottom"))))

        if None not in (x1, y1, x2, y2):
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        x = self._as_float(data.get("x"))
        y = self._as_float(data.get("y"))
        w = self._as_float(data.get("width", data.get("w")))
        h = self._as_float(data.get("height", data.get("h")))
        if None not in (x, y, w, h):
            return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        return None

    def _points_from_overlay_coordinates(self, coordinates, geometry_type):
        decoded = self._decode_overlay_coordinates(coordinates)
        if decoded is None:
            return None

        if isinstance(decoded, dict):
            for key in ("points", "polygon", "coordinates"):
                if key in decoded:
                    points = self._points_from_overlay_coordinates(decoded.get(key), geometry_type)
                    if points:
                        return points
            if "bbox" in decoded:
                points = self._points_from_overlay_coordinates(decoded.get("bbox"), "bbox")
                if points:
                    return points
            return self._bbox_points_from_dict(decoded)

        if not isinstance(decoded, (list, tuple)):
            return None

        if len(decoded) == 4 and all(not isinstance(item, (list, tuple, dict)) for item in decoded):
            x1, y1, x2, y2 = [self._as_float(item) for item in decoded]
            if None in (x1, y1, x2, y2):
                return None
            if x2 <= x1 or y2 <= y1:
                x2 = x1 + max(0.0, x2)
                y2 = y1 + max(0.0, y2)
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        if decoded and all(not isinstance(item, (list, tuple, dict)) for item in decoded):
            values = [self._as_float(item) for item in decoded]
            if any(value is None for value in values) or len(values) < 6 or len(values) % 2 != 0:
                return None
            return list(zip(values[0::2], values[1::2]))

        points = []
        for item in decoded:
            if isinstance(item, dict):
                x = self._as_float(item.get("x"))
                y = self._as_float(item.get("y"))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                x = self._as_float(item[0])
                y = self._as_float(item[1])
            else:
                x = y = None
            if x is None or y is None:
                return None
            points.append((x, y))

        if geometry_type == "bbox" and len(points) == 2:
            (x1, y1), (x2, y2) = points
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return points if len(points) >= 2 else None

    def _scale_overlay_points(self, points, overlay, target_w, target_h, fallback_source_w, fallback_source_h):
        flat_values = [value for point in points for value in point]
        normalized = flat_values and all(0.0 <= value <= 1.0 for value in flat_values)
        if normalized:
            return [
                (
                    int(round(max(0.0, min(1.0, x)) * target_w)),
                    int(round(max(0.0, min(1.0, y)) * target_h)),
                )
                for x, y in points
            ]

        source_w = self._as_float(overlay.get("image_width")) or fallback_source_w or target_w
        source_h = self._as_float(overlay.get("image_height")) or fallback_source_h or target_h
        source_w = max(1.0, float(source_w))
        source_h = max(1.0, float(source_h))
        return [
            (
                int(round(max(0.0, min(source_w, x)) * target_w / source_w)),
                int(round(max(0.0, min(source_h, y)) * target_h / source_h)),
            )
            for x, y in points
        ]

    def _overlay_color(self, class_name):
        label = str(class_name or "").strip().upper()
        if label == "OK":
            return (103, 122, 20)
        palette = [
            (49, 49, 255),
            (0, 180, 255),
            (64, 220, 64),
            (255, 160, 64),
            (220, 80, 220),
        ]
        return palette[sum(ord(ch) for ch in label) % len(palette)]

    def _draw_model_overlays(self, img, overlays, source_w, source_h):
        if img is None or not overlays:
            return img

        target_h, target_w = img.shape[:2]
        for overlay in overlays:
            geometry_type = str(overlay.get("geometry_type") or "bbox").strip().lower()
            if geometry_type == "classification":
                continue

            points = self._points_from_overlay_coordinates(
                overlay.get("coordinates"),
                geometry_type,
            )
            if not points:
                continue

            scaled_points = self._scale_overlay_points(
                points,
                overlay,
                target_w,
                target_h,
                source_w,
                source_h,
            )
            if len(scaled_points) < 2:
                continue

            color = self._overlay_color(overlay.get("class_name"))
            pts = np.array(scaled_points, dtype=np.int32).reshape((-1, 1, 2))
            line_thickness = 1
            xs = [point[0] for point in scaled_points]
            ys = [point[1] for point in scaled_points]

            if geometry_type == "bbox" and len(scaled_points) >= 4:
                cv2.rectangle(img, (min(xs), min(ys)), (max(xs), max(ys)), color, line_thickness)
                label_x, label_y = min(xs), max(ys)
            else:
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=line_thickness)
                label_x, label_y = min(xs), max(ys)

            class_name = str(overlay.get("class_name") or "").strip()
            confidence = overlay.get("confidence")
            label = class_name
            if confidence is not None:
                try:
                    label = f"{class_name} {float(confidence):.2f}"
                except (TypeError, ValueError):
                    pass
            if label:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                text_x = max(0, min(target_w - text_size[0] - 4, int(label_x)))
                text_y = max(
                    text_size[1] + 6,
                    min(target_h - 4, int(label_y) + text_size[1] + 8),
                )
                cv2.rectangle(
                    img,
                    (text_x, text_y - text_size[1] - 6),
                    (text_x + text_size[0] + 6, text_y + 3),
                    color,
                    -1,
                )
                cv2.putText(
                    img,
                    label,
                    (text_x + 3, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )
        return img

    def _normalize_grid_item(self, item):
        if isinstance(item, dict):
            img_name = str(item.get("img_name") or "").strip()
            img_path = item.get("path")
            if not img_name and img_path:
                img_name = self.file_manager.basename(img_path)
            return {
                "img_name": img_name,
                "path": img_path,
                "status": str(item.get("status") or "ready").strip().lower(),
                "source": item.get("source"),
                "prepared_image": item.get("prepared_image"),
                "error": item.get("error"),
            }

        img_path = item
        return {
            "img_name": self.file_manager.basename(img_path),
            "path": img_path,
            "status": "ready",
            "source": None,
            "prepared_image": None,
            "error": None,
        }

    def _draw_tile_placeholder(self, canvas, x, y, size, status):
        status_text = str(status or "").upper()
        if status_text == "LOADING":
            label = "LOADING"
            fill = (45, 45, 45)
            border = (0, 180, 255)
        elif status_text == "ERROR":
            label = "ERROR"
            fill = (35, 35, 55)
            border = (49, 49, 255)
        else:
            label = "MISSING"
            fill = (35, 35, 35)
            border = (120, 120, 120)

        cv2.rectangle(canvas, (x, y), (x + size, y + size), fill, -1)
        cv2.rectangle(canvas, (x, y), (x + size, y + size), border, 3)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = x + (size - text_size[0]) // 2
        text_y = y + (size + text_size[1]) // 2
        cv2.putText(
            canvas,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (235, 235, 235),
            thickness,
        )

        if status_text == "LOADING":
            dot_count = int(time.monotonic() * 4) % 4
            dots = "." * dot_count
            dot_size = cv2.getTextSize(dots, font, 0.9, thickness)[0]
            cv2.putText(
                canvas,
                dots,
                (x + (size - dot_size[0]) // 2, text_y + 36),
                font,
                0.9,
                (0, 180, 255),
                thickness,
            )

    def _extract_status_from_filename(self, img_name):
        base_name = os.path.splitext(str(img_name or ""))[0]
        if base_name.upper().endswith("_OK"):
            return "OK"
        if base_name.upper().endswith("_NOK"):
            return "NOK"
        return None

    def _draw_filename_status_badge(self, canvas, x, y, size, status_text):
        status_text = self._extract_status_from_filename(status_text) or status_text
        if status_text not in ("OK", "NOK"):
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.62
        thickness = 2
        padding_x = 10
        padding_y = 7
        margin = 8
        text_size = cv2.getTextSize(status_text, font, font_scale, thickness)[0]
        badge_w = text_size[0] + padding_x * 2
        badge_h = text_size[1] + padding_y * 2
        badge_x1 = x + size - badge_w - margin
        badge_y1 = y + margin
        badge_x2 = badge_x1 + badge_w
        badge_y2 = badge_y1 + badge_h
        fill = (49, 49, 255) if status_text == "NOK" else (103, 122, 20)

        cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), fill, -1)
        cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), (0, 0, 0), 1)
        cv2.putText(
            canvas,
            status_text,
            (badge_x1 + padding_x, badge_y2 - padding_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    def show_image_grid(self, image_paths, cols=4, rows=2, img_size=None, padding=None):
        """Show images without scaling, with fixed padding"""
        if img_size is None:
            img_size = self.DEFAULT_TILE_SIZE
        if padding is None:
            padding = self.DEFAULT_TILE_PADDING
        canvas = self._get_background_canvas()

        total_width = cols * img_size + (cols - 1) * padding
        total_height = rows * img_size + (rows - 1) * padding

        start_x = (self.width - total_width) // 2
        start_y = (self.height - total_height) // 2
        
        # Clear result buttons list at start
        self.result_buttons = []

        # Pre-blank the 7 active tile slots (skip last slot = bottom-right)
        for slot in range(cols * rows - 1):
            r = slot // cols
            c = slot % cols
            sx = start_x + c * (img_size + padding)
            sy = start_y + r * (img_size + padding)
            canvas[sy:sy + img_size, sx:sx + img_size] = 30  # dark gray

        for idx, img_path in enumerate(image_paths):
            if idx >= cols * rows:
                break

            tile_item = self._normalize_grid_item(img_path)
            img_filename = tile_item["img_name"]
            status = tile_item["status"]
            prepared_image = tile_item.get("prepared_image")
            raw_path = tile_item.get("path")

            img = None
            if status == "ready":
                if prepared_image is not None:
                    try:
                        img = prepared_image.copy()
                    except Exception:
                        img = None
                elif raw_path:
                    img = self._get_cached_image(raw_path, target_size=img_size)
                    if img is not None:
                        img = img.copy()

            row = idx // cols
            col = idx % cols

            x = start_x + col * (img_size + padding)
            y = start_y + row * (img_size + padding)

            if img is not None and (img.shape[0] != img_size or img.shape[1] != img_size):
                interpolation = cv2.INTER_AREA if (img.shape[0] > img_size or img.shape[1] > img_size) else cv2.INTER_LINEAR
                img = cv2.resize(img, (img_size, img_size), interpolation=interpolation)
            
            # Check if this image is being hovered or pressed
            x_draw, y_draw, size_draw = x, y, img_size
            if self.historic_mode:
                is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, (x, y, img_size, img_size))
                is_pressed = is_hovered and self.mouse_button_down
                
                # Calculate scale factor
                scale_factor = 0.95 if is_pressed else (1.08 if is_hovered else 1.0)
                
                if scale_factor != 1.0:
                    new_size = int(img_size * scale_factor)
                    x_draw = x - (new_size - img_size) // 2
                    y_draw = y - (new_size - img_size) // 2
                    size_draw = new_size
                    # Resize image to scaled size
                    if img is not None:
                        img = cv2.resize(img, (size_draw, size_draw))

            if img is None:
                self._draw_tile_placeholder(canvas, x_draw, y_draw, size_draw, status)
            else:
                canvas[y_draw:y_draw + size_draw, x_draw:x_draw + size_draw] = img

            if self.historic_mode:
                self._draw_filename_status_badge(
                    canvas,
                    x_draw,
                    y_draw,
                    size_draw,
                    img_filename,
                )

            # Show camera label above each image (normal + historic)
            label_source = raw_path or img_filename
            label_text = self._extract_camera_label(label_source)
            if label_text:
                self._draw_camera_label(canvas, x, y, img_size, label_text)
            
            # If we are in historic mode, show result below each image
            if self.historic_mode:
                # Extract filename from path
                result_text = self.get_result_for_image(img_filename)
                
                # Dibujar etiqueta debajo de la imagen
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                label_text = result_text  # Only the value, without "Result:"
                
                # Calculate text position (centered below scaled image)
                text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
                text_x = x_draw + (size_draw - text_size[0]) // 2
                text_y = y_draw + size_draw + 30  # 30 pixels below image
                
                # Draw background the width of the scaled image
                bg_x1 = x_draw
                bg_y1 = text_y - text_size[1] - 8
                bg_x2 = x_draw + size_draw
                bg_y2 = text_y + 8
                
                # Make sure it doesn't go outside canvas
                if bg_y2 < self.height and bg_x2 < self.width:
                    # Background color according to result
                    if result_text == "NOK":
                        bg_color = (49, 49, 255)  # #ff3131 in BGR (red)
                    else:
                        bg_color = (103, 122, 20)  # #147a67 in BGR (green)
                    
                    cv2.rectangle(canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
                    cv2.putText(canvas, label_text, (text_x, text_y), font, font_scale, 
                               (255, 255, 255), thickness)
                    
                    # Save button rectangle (image + text together) to detect clicks
                    # Use scaled coordinates if image was scaled
                    button_height = bg_y2 - y_draw
                    button_rect = (x_draw, y_draw, size_draw, button_height)
                    self.result_buttons.append((button_rect, img_filename, result_text))

            else:
                # Vista normal: extraer estado OK/NOK desde el nombre del archivo
                base_name = os.path.splitext(img_filename)[0]
                if base_name.endswith('_OK'):
                    result_text = 'OK'
                elif base_name.endswith('_NOK'):
                    result_text = 'NOK'
                else:
                    result_text = None

                if result_text is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    label_text = result_text

                    text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
                    text_x = x_draw + (size_draw - text_size[0]) // 2
                    text_y = y_draw + size_draw + 30

                    bg_x1 = x_draw
                    bg_y1 = text_y - text_size[1] - 8
                    bg_x2 = x_draw + size_draw
                    bg_y2 = text_y + 8

                    if bg_y2 < self.height and bg_x2 < self.width:
                        bg_color = (49, 49, 255) if result_text == 'NOK' else (103, 122, 20)
                        cv2.rectangle(canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
                        cv2.putText(canvas, label_text, (text_x, text_y), font,
                                    font_scale, (255, 255, 255), thickness)

        # Draw stats card in the last slot (main display and historic)
        stats_slot = cols * rows - 1
        stats_row = stats_slot // cols
        stats_col = stats_slot % cols
        stats_x = start_x + stats_col * (img_size + padding)
        stats_y = start_y + stats_row * (img_size + padding)
        self.stats_card_rect = (stats_x, stats_y, img_size, img_size)

        db_counts = self._get_piece_result_counts()
        ok_count = db_counts.get("OK", 0)
        nok_count = db_counts.get("NOK", 0)
        fok_count = db_counts.get("FOK", 0)
        fnok_count = db_counts.get("FNOK", 0)

        self._draw_stats_card(canvas, stats_x, stats_y, img_size, ok_count, nok_count, fok_count, fnok_count)

        # Normal mode: only HISTORIC button
        if not self.historic_mode:
            canvas = self.draw_historic_button(canvas)
            canvas = self.draw_import_button(canvas)
            canvas = self.draw_export_button(canvas)
            canvas = self.draw_exit_button(canvas)
        else:
            # Historic mode: show JSN in upper blue bar
            self.historic_jsn_rect = None
            self.piece_counter_rect = None
            self.piece_identifier_rect = None
            if self.historic_images and len(self.historic_images) > 0:
                current_batch = self.historic_images[self.historic_offset]
                is_incomplete = len(current_batch) < 7

                canvas = self.draw_historic_jsn_banner(canvas)
                canvas = self.draw_piece_identifier_badge(canvas)

                if is_incomplete:
                    incomplete_text = f"INCOMPLETE BATCH ({len(current_batch)}/7)"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    thickness = 3
                    text_size_bottom = cv2.getTextSize(incomplete_text, font, font_scale, thickness)[0]
                    text_x_bottom = (self.width - text_size_bottom[0]) // 2
                    text_y_bottom = self.height - 30
                    cv2.putText(
                        canvas,
                        incomplete_text,
                        (text_x_bottom, text_y_bottom),
                        font,
                        font_scale,
                        (0, 0, 255),
                        thickness,
                    )
            
            # Historic mode: navigation arrows, search elements and BACK button
            canvas = self.draw_prev_button(canvas)
            canvas = self.draw_next_button(canvas)
            canvas = self.draw_search_elements(canvas)
            canvas = self.draw_back_button(canvas)
            canvas = self.draw_info_icon(canvas)
            canvas = self.draw_trash_button(canvas)
            canvas = self.draw_image_report_button(canvas)
            canvas = self.draw_import_button(canvas)
            canvas = self.draw_export_button(canvas)
            canvas = self.draw_sync_button(canvas)
            canvas = self.draw_reset_button(canvas)
            
            # Draw piece date dialog if needed
            if self.show_piece_date_dialog:
                canvas = self.draw_piece_date_dialog(canvas)
            if self.show_piece_number_dialog:
                canvas = self.draw_piece_number_dialog(canvas)
            if self.show_piece_identifier_dialog:
                canvas = self.draw_piece_identifier_dialog(canvas)
            
            # Draw confirmation dialog if needed
            if self.show_reset_confirm:
                canvas = self.draw_reset_confirmation_dialog(canvas)
            elif self.show_delete_confirm:
                canvas = self.draw_delete_confirmation_dialog(canvas)

        if self.show_stats_class_modal:
            canvas = self.draw_stats_class_modal(canvas)

        if self.show_rebuild_confirm:
            canvas = self.draw_rebuild_confirmation_dialog(canvas)

        if self.show_no_images_dialog:
            canvas = self.draw_no_images_dialog(canvas)
        if self.sync_in_progress:
            canvas = self.draw_sync_progress(canvas)
        elif self.reset_in_progress:
            canvas = self.draw_reset_progress(canvas)
        else:
            canvas = self.draw_sync_message(canvas)
            canvas = self.draw_toast_message(canvas)
        if self.db_blocking:
            canvas = self.draw_db_block_dialog(canvas)

        self.image = canvas
        return self.show()  # Muestra el display y retorna True para actualizar


def check_historic_images():
    controller_check_historic_images()


if __name__ == "__main__":
    display = DisplayWindow()
    display.sync_images_by_status()

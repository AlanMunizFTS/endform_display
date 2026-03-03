
import cv2
import numpy as np
import os
import re
from collections import OrderedDict
from multiprocessing import Process
from db import PostgresDB, get_db_connection
from file_manager import FileManager
from utilities.log import get_logger, install_print_logger
from settings import get_sftp_settings
from paths_config import (
    HISTORIC_SUBDIR_NAME,
    STATUS_SYNC_DIRS,
    SYNC_IMAGES_BASE_DIR,
    TMP_DISPLAY_DIR,
)


# Standalone worker kept for compatibility with SFTP mode.
def _download_images_background_worker(hostname, port, username, password, remote_dir, historic_temp_dir, check_interval=30, verbose=False):
    """Background historic downloader. In local-only mode this worker is not used."""
    install_print_logger(reset=False)
    logger = get_logger()
    logger.info(
        "[HIST_SYNC_SSH] Background downloader skipped (local-only mode)",
        allow_repeat=True,
    )
    return

class DisplayWindow:
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
        self.remote_hist_dir = "/media/ssd/hist_display"  # Remote folder for history
        self.refresh_interval = refresh_interval  # Seconds between updates
        self.last_refresh_time = 0
        self.sftp_client = None  # SFTP client to upload images
        self.remote_controls_enabled = False
        self.sftp_credentials = sftp_credentials  # Credenciales SFTP para multiprocessing
        self.file_manager = file_manager or FileManager()
        self.filename_mapping = filename_mapping or {}  # Mapping of short names to original names
        self.historic_mode = False  # Indicates if we are in historic mode
        self.historic_offset = 0  # Offset to navigate through historic batches
        self.historic_images = []  # Complete list of historic images
        self.result_buttons = []  # List of result buttons [(rect, img_name, result_value), ...]
        self.temp_results = {}  # Dictionary for temporary changes {img_name: new_value}
        self.db = get_db_connection()
        self.download_process = None  # Process for background download
        self.historic_db_registered = False  # Tracks whether visible historic images were registered in DB.
        self.search_button_rect = None  # Search button rect
        self.search_input_rect = None  # Search input field rect
        self.search_jsn = ""  # Current JSN search term
        self.search_active = False  # Whether search input is active
        self.available_jsns = []  # List of all available JSNs
        self.filtered_suggestions = []  # Filtered suggestions based on input
        self.selected_suggestion_idx = -1  # Index of selected suggestion (-1 = none)
        self.suggestion_rects = []  # Rectangles for each suggestion
        self.reset_button_rect = None  # Reset button rect
        self.trash_button_rect = None  # Trash button rect
        self.sync_button_rect = None  # Sync button rect
        self.exit_button_rect = None  # Exit button rect
        self.start_stop_button_rect = None  # Start/Stop button rect
        self.remote_action_request = None  # "start" or "stop" requested by UI
        self.remote_requested = False  # Whether remote process is requested to run
        self.show_reset_confirm = False  # Show reset confirmation dialog
        self.reset_confirm_button_rect = None  # Confirm button rect
        self.reset_cancel_button_rect = None  # Cancel button rect
        self.show_delete_confirm = False  # Show delete-piece confirmation dialog
        self.delete_confirm_button_rect = None  # Confirm delete button rect
        self.delete_cancel_button_rect = None  # Cancel delete button rect
        self.show_no_images_dialog = False  # Show no images available dialog
        self.no_images_ok_button_rect = None  # OK button rect for no images dialog
        self.sync_message = ""
        self.sync_message_time = 0
        self.exit_requested = False
        self.trigger_active = False  # Trigger status for normal view
        self.connected_cameras = set()  # Unique connected cameras in normal view
        self.total_cameras = 7
        self.camera_icon = None
        self.camera_icon_size = None
        self._camera_icon_warned = False
        self.trash_icon = None
        self.trash_icon_size = None
        self._trash_icon_warned = False
        # Shift right-aligned status text/icon left (pixels)
        self.right_info_shift = 140
        self.connected_cameras = set()  # Unique connected cameras in normal view
        self.total_cameras = 7
        self.info_icon_rect = None  # Info icon rect (top right in historic mode)
        self.show_piece_date_dialog = False  # Show piece date dialog
        self.piece_date_dialog_close_rect = None  # Close button rect for date dialog
        self.mouse_x = 0  # Current mouse X position
        self.mouse_y = 0  # Current mouse Y position
        self.mouse_button_down = False  # Track if left mouse button is down
        self.historic_auto_refresh_interval = 2.0
        self._last_historic_auto_refresh = 0.0
        self._background_cache = None
        self._background_cache_mtime = None
        self._background_cache_size = (self.width, self.height)
        self._image_cache = OrderedDict()
        self._image_cache_max_items = 64
        self._db_result_cache = {}
        self._db_registered_images = set()
        self._historic_index_cache = None
        self._historic_index_mtime = None
        self._historic_index_last_scan = 0.0
        self.historic_index_rescan_interval = 1.5
        self._historic_jsn_cache = []
        self.set_sftp_client(sftp_client)

    def set_sftp_client(self, sftp_client):
        """Update active SFTP client and dependent UI/control state."""
        self.sftp_client = sftp_client
        self.remote_controls_enabled = bool(self.sftp_client)

        if not self.remote_controls_enabled:
            self.remote_action_request = None
            self.remote_requested = False
            self.trigger_active = False
            if hasattr(self, "connected_cameras"):
                self.connected_cameras = set()

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
    
    def _get_piece_date(self):
        """Get the date of the current historic piece from the first image."""
        if not self.historic_images or self.historic_offset >= len(self.historic_images):
            return "N/A"
        
        try:
            batch = self.historic_images[self.historic_offset]
            if not batch:
                return "N/A"
            
            # Get the first image filename
            first_image = batch[0]
            historic_dir = self.file_manager.join(str(TMP_DISPLAY_DIR), HISTORIC_SUBDIR_NAME)
            image_path = self.file_manager.join(historic_dir, first_image)
            
            # Get file modification time
            if self.file_manager.exists(image_path):
                import datetime
                mtime = self.file_manager.getmtime(image_path)
                dt = datetime.datetime.fromtimestamp(mtime)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                return "N/A"
        except Exception as e:
            print(f"Error getting piece date: {e}")
            return "N/A"
    def create_white_display(self):
        """Create a white display"""
        self.image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
    def _is_point_in_rect(self, x, y, rect):
        """Check if point (x, y) is inside rectangle rect (x, y, width, height)"""
        if rect is None:
            return False
        bx, by, bw, bh = rect
        return bx <= x <= bx + bw and by <= y <= by + bh
    
    def _apply_hover_color(self, base_color, is_hovered):
        """Apply hover effect to a color by brightening it"""
        if not is_hovered:
            return base_color
        # Brighten the color by increasing each channel by 40 (capped at 255)
        b, g, r = base_color
        return (
            min(b + 40, 255),
            min(g + 40, 255),
            min(r + 40, 255)
        )
    
    def _apply_pressed_color(self, base_color):
        """Apply pressed effect to a color by darkening it"""
        b, g, r = base_color
        return (
            max(b - 60, 0),
            max(g - 60, 0),
            max(r - 60, 0)
        )
    
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

    def mouse_callback(self, event, x, y, flags, param):
        """Callback to handle mouse events"""
        # Track mouse position and button state
        self.mouse_x = x
        self.mouse_y = y
        self.mouse_button_down = (flags & cv2.EVENT_FLAG_LBUTTON) != 0
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Piece date dialog close button (highest priority)
            if self.show_piece_date_dialog and self.piece_date_dialog_close_rect:
                bx, by, bw, bh = self.piece_date_dialog_close_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.show_piece_date_dialog = False
                return  # Exit early to prevent other clicks
            
            # No images dialog OK button (highest priority)
            if self.show_no_images_dialog and self.no_images_ok_button_rect:
                bx, by, bw, bh = self.no_images_ok_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.show_no_images_dialog = False
                return  # Exit early to prevent other clicks

            # Delete-piece confirmation buttons (high priority)
            if self.show_delete_confirm:
                # Confirm button
                if self.delete_confirm_button_rect:
                    bx, by, bw, bh = self.delete_confirm_button_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self.show_delete_confirm = False
                        self.perform_delete_current_piece()
                        return

                # Cancel button
                if self.delete_cancel_button_rect:
                    bx, by, bw, bh = self.delete_cancel_button_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self.show_delete_confirm = False
                        return
                # If dialog is shown, don't process other clicks
                return
            
            # Reset confirmation buttons (high priority)
            if self.show_reset_confirm:
                # Confirm button
                if self.reset_confirm_button_rect:
                    bx, by, bw, bh = self.reset_confirm_button_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self.show_reset_confirm = False
                        self.perform_reset()
                        return
                
                # Cancel button
                if self.reset_cancel_button_rect:
                    bx, by, bw, bh = self.reset_cancel_button_rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self.show_reset_confirm = False
                        return
                # If dialog is shown, don't process other clicks
                return
            
            # HISTORIC button - only to activate historic mode
            if self.save_button_rect and not self.historic_mode and not self.show_no_images_dialog:
                bx, by, bw, bh = self.save_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.enter_historic_mode()
                    return

            # EXIT button - only in normal mode
            if self.exit_button_rect and not self.historic_mode and not self.show_no_images_dialog:
                bx, by, bw, bh = self.exit_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.exit_requested = True
                    return

            # START/STOP button - only in normal mode
            if (
                self.remote_controls_enabled
                and self.start_stop_button_rect
                and not self.historic_mode
                and not self.show_no_images_dialog
            ):
                bx, by, bw, bh = self.start_stop_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    if self.remote_requested:
                        self.remote_action_request = "stop"
                        self.remote_requested = False
                    else:
                        self.remote_action_request = "start"
                        self.remote_requested = True
                    self.trigger_active = False
                    self.connected_cameras = set()
                    return
            
            # BACK button - exit historic mode
            if self.back_button_rect and self.historic_mode:
                bx, by, bw, bh = self.back_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.exit_historic_mode()
            
            # INFO icon - show piece date (historic mode)
            if self.info_icon_rect and self.historic_mode:
                bx, by, bw, bh = self.info_icon_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.show_piece_date_dialog = True
                    return
            
            # NEXT ARROW button (right) - advance in historic
            if self.next_button_rect and self.historic_mode:
                bx, by, bw, bh = self.next_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.next_historic_batch()
            
            # PREVIOUS ARROW button (left) - go back in historic
            if self.prev_button_rect and self.historic_mode:
                bx, by, bw, bh = self.prev_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.prev_historic_batch()
            
            # Search button - in historic mode
            if self.search_button_rect and self.historic_mode:
                bx, by, bw, bh = self.search_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.perform_jsn_search()
            
            # RESET button - in historic mode
            if self.reset_button_rect and self.historic_mode and not self.show_reset_confirm and not self.show_delete_confirm:
                bx, by, bw, bh = self.reset_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.show_reset_confirm = True
                    self.show_delete_confirm = False

            # TRASH button - in historic mode
            if self.trash_button_rect and self.historic_mode and not self.show_reset_confirm and not self.show_delete_confirm:
                bx, by, bw, bh = self.trash_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.show_delete_confirm = True
                    self.show_reset_confirm = False

            # SYNC button - in historic mode
            if self.sync_button_rect and self.historic_mode and not self.show_reset_confirm and not self.show_delete_confirm:
                bx, by, bw, bh = self.sync_button_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.sync_images_by_status()
                    self.sync_message = "Dataset correctly saved"
                    import time
                    self.sync_message_time = time.time()
            
            # Search input field - in historic mode
            if self.search_input_rect and self.historic_mode:
                bx, by, bw, bh = self.search_input_rect
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.search_active = True
                    # Collect available JSNs when activating search
                    self.collect_available_jsns()
                    self.update_suggestions()
                    return
            
            # Suggestion items - in historic mode
            clicked_on_suggestion = False
            if self.historic_mode and self.suggestion_rects:
                for idx, (rect, jsn_value) in enumerate(self.suggestion_rects):
                    bx, by, bw, bh = rect
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        # Select this JSN
                        self.search_jsn = jsn_value[:21]
                        self.perform_jsn_search()
                        self.search_active = False
                        self.filtered_suggestions = []
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
                    self.search_active = False
                    self.filtered_suggestions = []
            
            # Result buttons - in historic mode
            if self.historic_mode and self.result_buttons:
                for rect, img_name, result_value in self.result_buttons:
                    bx, by, bw, bh = rect
                    # Check against the scaled area (accounting for hover/press effects)
                    is_hovered = self._is_point_in_rect(x, y, rect)
                    
                    if is_hovered:
                        # Change value from OK to NOK (toggle)
                        new_value = "NOK" if result_value == "OK" else "OK"
                        # Save directly to database
                        self._update_result_in_db(img_name, new_value)
                        # Also save in temp_results to avoid redundant queries
                        self.temp_results[img_name] = new_value
                        break
    
    def _load_historic_index(self, force_rescan=False):
        """Load grouped historic images, using cache when folder mtime is unchanged."""
        import time
        from collections import defaultdict

        local_historic_dir = self.file_manager.join(str(TMP_DISPLAY_DIR), HISTORIC_SUBDIR_NAME)
        if not self.file_manager.exists(local_historic_dir):
            self._historic_index_cache = []
            self._historic_jsn_cache = []
            self._historic_index_mtime = None
            self._historic_index_last_scan = time.monotonic()
            return []

        current_mtime = None
        try:
            current_mtime = self.file_manager.getmtime(local_historic_dir)
        except Exception:
            pass

        use_cache = False
        if not force_rescan and self._historic_index_cache is not None:
            if current_mtime is not None and current_mtime == self._historic_index_mtime:
                use_cache = True
            elif (
                current_mtime is None
                and (time.monotonic() - self._historic_index_last_scan) < self.historic_index_rescan_interval
            ):
                use_cache = True

        if use_cache:
            return self._historic_index_cache

        image_extensions = (".png", ".jpg", ".jpeg", ".bmp")
        files = self.file_manager.listdir(local_historic_dir)
        images_with_jsn = [
            name
            for name in files
            if name.lower().endswith(image_extensions) and name.startswith("11861")
        ]

        jsn_groups = defaultdict(list)
        for img in images_with_jsn:
            jsn = img.split("_")[0]
            jsn_groups[jsn].append(img)

        sorted_jsns = sorted(jsn_groups.keys(), reverse=True)
        historic_images = []
        for jsn in sorted_jsns:
            group_images = jsn_groups[jsn]
            group_images.sort(key=self._custom_sort_key)
            historic_images.append(group_images)

        self._historic_index_cache = historic_images
        self._historic_jsn_cache = sorted_jsns
        self._historic_index_mtime = current_mtime
        self._historic_index_last_scan = time.monotonic()

        # Directory content changed; register new DB rows lazily per visible batch.
        self.historic_db_registered = False
        return historic_images

    def enter_historic_mode(self):
        """Activate historic mode and load list of images grouped by JSN from local directory."""
        # Preserve current JSN so background downloads don't move the view.
        current_jsn = None
        fallback_offset = self.historic_offset
        if self.historic_mode and self.historic_images:
            try:
                current_batch = self.historic_images[self.historic_offset]
                if current_batch:
                    current_jsn = current_batch[0].split("_")[0] if "_" in current_batch[0] else current_batch[0]
            except Exception:
                current_jsn = None

        try:
            self.historic_images = self._load_historic_index(force_rescan=False)

            if not self.historic_images:
                if not self.historic_mode:
                    self.show_no_images_dialog = True
                return

            # Activate historic mode only if not already active.
            if not self.historic_mode:
                self.historic_mode = True
                self.historic_offset = 0
            else:
                # If already in historic, keep the same JSN visible after refresh.
                if current_jsn:
                    found_idx = None
                    for idx, batch in enumerate(self.historic_images):
                        if not batch:
                            continue
                        batch_jsn = batch[0].split("_")[0] if "_" in batch[0] else batch[0]
                        if batch_jsn == current_jsn:
                            found_idx = idx
                            break
                    if found_idx is not None:
                        self.historic_offset = found_idx
                    else:
                        self.historic_offset = min(fallback_offset, len(self.historic_images) - 1)
                else:
                    self.historic_offset = min(fallback_offset, len(self.historic_images) - 1)

        except Exception as e:
            print(f"Error entering historic: {e}")

    def _custom_sort_key(self, filename):
        """Helper function to sort by type: side -> front -> diag"""
        lower_name = filename.lower()
        if 'side' in lower_name:
            return (0, filename)
        elif 'front' in lower_name:
            return (1, filename)
        elif 'diag' in lower_name:
            return (2, filename)
        else:
            return (3, filename)
    
    def exit_historic_mode(self):
        """Exit historic mode"""
        self.historic_mode = False
        self.historic_offset = 0
        self.historic_images = []
        self.search_jsn = ""
        self.search_active = False
        self.filtered_suggestions = []
        self.selected_suggestion_idx = -1
        self.show_reset_confirm = False
        self.show_delete_confirm = False
        self.show_piece_date_dialog = False
    
    def next_historic_batch(self):
        """Advance to next batch of historic images"""
        if not self.historic_images:
            return
        
        total_batches = len(self.historic_images)  # Each group is a batch
        self.historic_offset = (self.historic_offset + 1) % total_batches
    
    def prev_historic_batch(self):
        """Go back to previous batch of historic images"""
        if not self.historic_images:
            return
        
        # Don't allow going back if already at first batch
        if self.historic_offset == 0:
            return
        
        total_batches = len(self.historic_images)  # Each group is a batch
        self.historic_offset = self.historic_offset - 1
    
    def collect_available_jsns(self):
        """Collect all available JSN numbers from historic images"""
        if not self.historic_images:
            self.available_jsns = []
            return

        if self._historic_jsn_cache:
            self.available_jsns = list(self._historic_jsn_cache)
            return

        jsn_set = set()
        for batch in self.historic_images:
            if batch and len(batch) > 0:
                jsn = batch[0].split('_')[0] if '_' in batch[0] else ''
                if jsn:
                    jsn_set.add(jsn)
        
        # Sort JSNs in descending order (most recent first)
        self.available_jsns = sorted(list(jsn_set), reverse=True)
    
    def update_suggestions(self):
        """Update filtered suggestions based on current input"""
        if not self.search_jsn:
            # Show all JSNs if input is empty
            self.filtered_suggestions = self.available_jsns[:10]  # Limit to 10
        else:
            # Filter JSNs that contain the input anywhere in the string
            self.filtered_suggestions = [jsn for jsn in self.available_jsns if self.search_jsn in jsn][:10]
        
        self.selected_suggestion_idx = -1
    
    def perform_jsn_search(self):
        """Search for a specific JSN and jump to that batch."""
        if not self.search_jsn.strip():
            print("No JSN entered for search")
            return

        search_term = self.search_jsn.strip()

        # Search for JSN in historic_images
        for idx, batch in enumerate(self.historic_images):
            jsn = batch[0].split("_")[0] if "_" in batch[0] else ""
            if jsn == search_term:
                self.historic_offset = idx
                print(f"JSN {search_term} found at position {idx}")
                self.search_active = False
                self.filtered_suggestions = []
                self.search_jsn = ""
                return

        print(f"JSN {search_term} not found in historic images")
        self.search_active = False
        self.filtered_suggestions = []
        self.search_jsn = ""

    def _get_current_historic_jsn(self):
        """Return JSN for the current historic batch, or None if unavailable."""
        if not self.historic_images:
            return None
        if self.historic_offset < 0 or self.historic_offset >= len(self.historic_images):
            return None
        batch = self.historic_images[self.historic_offset]
        if not batch:
            return None
        first = batch[0]
        return first.split('_')[0] if '_' in first else first

    def perform_delete_current_piece(self):
        """Delete current historic piece (local + remote hist_display + DB records)."""
        jsn = self._get_current_historic_jsn()
        if not jsn:
            print("No historic piece selected for deletion")
            return

        print("\n" + "="*70)
        print(f"STARTING PIECE DELETE (JSN {jsn})")
        print("="*70)

        image_extensions = (".png", ".jpg", ".jpeg", ".bmp")
        local_historic_dir = self.file_manager.join(str(TMP_DISPLAY_DIR), HISTORIC_SUBDIR_NAME)

        # 1. Delete local historic files for JSN
        local_deleted = 0
        local_candidates = []
        if self.file_manager.exists(local_historic_dir):
            try:
                for name in self.file_manager.listdir(local_historic_dir):
                    if name.startswith(jsn) and name.lower().endswith(image_extensions):
                        local_candidates.append(self.file_manager.join(local_historic_dir, name))
                for path in local_candidates:
                    try:
                        self.file_manager.remove(path)
                        local_deleted += 1
                    except Exception as e:
                        print(f"Error deleting local file {path}: {e}")
                print(f"Local delete: {local_deleted}/{len(local_candidates)}")
            except Exception as e:
                print(f"Error reading local historic folder: {e}")
        else:
            print("Local historic folder does not exist")

        # 2. Delete remote hist_display files for JSN
        remote_deleted = 0
        if self.sftp_client:
            try:
                self.file_manager.sftp_chdir(self.sftp_client, self.remote_hist_dir)
                remote_files = self.file_manager.sftp_listdir(self.sftp_client)
                remote_candidates = [f for f in remote_files if f.startswith(jsn) and f.lower().endswith(image_extensions)]
                for remote_file in remote_candidates:
                    try:
                        file_path = f"{self.remote_hist_dir}/{remote_file}"
                        self.file_manager.sftp_remove(self.sftp_client, file_path)
                        remote_deleted += 1
                    except Exception as e:
                        print(f"Error deleting remote file {remote_file}: {e}")
                print(f"Remote delete: {remote_deleted}/{len(remote_candidates)}")
            except Exception as e:
                print(f"Error accessing remote historic folder: {e}")
        else:
            print("No SFTP connection available")

        # 3. Delete DB records for JSN
        if self.db:
            try:
                query_delete = "DELETE FROM img_results WHERE img_name LIKE %s"
                affected_rows = self.db.execute(query_delete, (f"{jsn}%",))
                print(f"Deleted {affected_rows} database records")
            except Exception as e:
                print(f"Error clearing database records: {e}")
        else:
            print("No database connection available")

        # 4. Clear temp results for this JSN
        if self.temp_results:
            self.temp_results = {k: v for k, v in self.temp_results.items() if not k.startswith(jsn)}
        self._db_registered_images = {name for name in self._db_registered_images if not name.startswith(jsn)}
        if self._db_result_cache:
            self._db_result_cache = {
                k: v for k, v in self._db_result_cache.items() if not k.startswith(jsn)
            }
        for path in local_candidates:
            self._image_cache.pop(path, None)

        self.historic_db_registered = False
        self._historic_index_cache = None
        self._historic_index_mtime = None
        self._historic_jsn_cache = []

        # Refresh historic list or exit if none left
        remaining_images = []
        if self.file_manager.exists(local_historic_dir):
            remaining_images = [
                f for f in self.file_manager.listdir(local_historic_dir)
                if f.lower().endswith(image_extensions) and f.startswith("11861")
            ]

        if not remaining_images:
            self.historic_images = []
            self.historic_offset = 0
            self.available_jsns = []
            self.filtered_suggestions = []
            self.exit_historic_mode()
            self.show_no_images_dialog = True
        else:
            self.enter_historic_mode()

        print("="*70)
        print("PIECE DELETE COMPLETED")
        print("="*70 + "\n")

    def perform_reset(self):
        """Reset everything: delete local historic folder, remote hist_display folder, and database records"""
        print("\n" + "="*70)
        print("STARTING COMPLETE RESET")
        print("="*70)

        # 1. Delete local historic folder
        local_historic_dir = self.file_manager.join(str(TMP_DISPLAY_DIR), HISTORIC_SUBDIR_NAME)
        if self.file_manager.exists(local_historic_dir):
            try:
                self.file_manager.rmtree(local_historic_dir)
                self.file_manager.makedirs(local_historic_dir, exist_ok=True)
                print("Local historic folder cleared")
            except Exception as e:
                print(f"Error clearing local historic folder: {e}")
        else:
            print("Local historic folder does not exist")

        # 2. Delete remote hist_display folder contents
        if self.sftp_client:
            try:
                self.file_manager.sftp_chdir(self.sftp_client, self.remote_hist_dir)
                remote_files = self.file_manager.sftp_listdir(self.sftp_client)

                if remote_files:
                    print(f"Deleting {len(remote_files)} files from remote server...")
                    deleted_count = 0
                    for remote_file in remote_files:
                        try:
                            file_path = f"{self.remote_hist_dir}/{remote_file}"
                            self.file_manager.sftp_remove(self.sftp_client, file_path)
                            deleted_count += 1
                        except Exception as e:
                            print(f"Error deleting {remote_file}: {e}")
                    print(f"Deleted {deleted_count}/{len(remote_files)} remote files")
                else:
                    print("Remote folder is already empty")
            except Exception as e:
                print(f"Error accessing remote folder: {e}")
        else:
            print("No SFTP connection available")

        # 3. Delete all records from database
        if self.db:
            try:
                query_delete = "DELETE FROM img_results"
                affected_rows = self.db.execute(query_delete)
                print(f"Deleted {affected_rows} records from database")
            except Exception as e:
                print(f"Error clearing database: {e}")
        else:
            print("No database connection available")

        # 4. Clear internal variables
        self.historic_images = []
        self.historic_offset = 0
        self.temp_results = {}
        self.available_jsns = []
        self.filtered_suggestions = []
        self.historic_db_registered = False
        self._db_registered_images.clear()
        self._historic_index_cache = None
        self._historic_index_mtime = None
        self._historic_jsn_cache = []
        self._db_result_cache.clear()
        self._image_cache.clear()

        print("="*70)
        print("RESET COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")

        # Exit historic mode
        self.exit_historic_mode()

    def start_historic_download_on_startup(self, local_path, check_interval=30):
        """Start historic sync/downloader. Local-only mode just ensures the folder exists."""
        # Local-only mode: there is no SFTP download process; just ensure local folder exists.
        if not self.sftp_client:
            historic_temp_dir = self.file_manager.join(local_path, HISTORIC_SUBDIR_NAME)
            self.file_manager.makedirs(historic_temp_dir, exist_ok=True)
            print("Local-only mode: background historic SFTP download is disabled")
            return

        if not self.sftp_credentials:
            print("Error: missing SFTP credentials for multiprocessing")
            return
        
        # Verificar si ya hay un proceso activo
        if self.download_process and self.download_process.is_alive():
            print("Background download process is already running")
            return
        
        try:
            # Create temp folder for historic
            historic_temp_dir = self.file_manager.join(local_path, HISTORIC_SUBDIR_NAME)
            self.file_manager.makedirs(historic_temp_dir, exist_ok=True)
            
            # Obtener credenciales
            hostname = self.sftp_credentials.get('hostname')
            port = self.sftp_credentials.get('port')
            username = self.sftp_credentials.get('username')
            password = self.sftp_credentials.get('password')
            
            
            # Iniciar descarga continua en proceso separado (BACKGROUND)
            self.download_process = Process(
                target=_download_images_background_worker,
                args=(hostname, port, username, password, self.remote_hist_dir, historic_temp_dir, check_interval)
            )
            self.download_process.daemon = True  # Proceso daemon se cierra cuando la app termina
            self.download_process.start()
                
        except Exception as e:
            print(f"Error starting background download: {e}")
            import traceback
            traceback.print_exc()
    
    def download_historic_batch(self, local_path, max_images=7):
        """Return currently selected historic batch from local cache."""
        if not self.historic_images:
            return []
        
        try:
            # Temp folder for historic
            historic_temp_dir = self.file_manager.join(local_path, HISTORIC_SUBDIR_NAME)
            
            # Obtener grupo actual para mostrar (ya ordenado por tipo)
            batch_images = self.historic_images[self.historic_offset]
            
            # Retornar rutas locales del grupo actual (EN VIVO - sin esperar descarga)
            downloaded_files = []
            for img in batch_images:
                local_file = self.file_manager.join(historic_temp_dir, img)
                if self.file_manager.exists(local_file):
                    downloaded_files.append(local_file)

            # Register only the visible batch to avoid large blocking scans.
            self._register_local_images_in_db(historic_temp_dir, image_names=batch_images)
            
            return downloaded_files
            
        except Exception as e:
            print(f"Error reading historic batch: {e}")
            return []
    
    def _register_local_images_in_db(self, historic_dir, image_names=None):
        """Register given local images in DB (or all from folder when image_names is None)."""
        try:
            # Check DB connection
            if not self.db:
                return
            
            if not self.file_manager.exists(historic_dir):
                return
            
            if image_names is None:
                image_extensions = (".png", ".jpg", ".jpeg", ".bmp")
                local_images = [
                    f for f in self.file_manager.listdir(historic_dir)
                    if f.lower().endswith(image_extensions)
                ]
            else:
                local_images = list(image_names)
            
            if not local_images:
                return

            pending = [img for img in local_images if img not in self._db_registered_images]
            if not pending:
                return

            # Fetch existing rows in one query.
            existing_rows = self.db.fetch(
                "SELECT img_name FROM img_results WHERE img_name = ANY(%s)",
                (pending,),
            )
            existing = {row["img_name"] for row in existing_rows} if existing_rows else set()

            images_to_insert = [img for img in pending if img not in existing]
            if images_to_insert:
                query_insert = "INSERT INTO img_results (img_name) VALUES (%s)"
                for img_name in images_to_insert:
                    try:
                        self.db.execute(query_insert, (img_name,))
                    except Exception as e:
                        print(f"Error inserting {img_name}: {e}")

            self._db_registered_images.update(pending)
            self.historic_db_registered = True
            
        except Exception as e:
            print(f"General error registering images in DB: {e}")
    
    def _update_result_in_db(self, img_name, new_value):
        """Update the result value in database"""
        try:
            query_update = "UPDATE img_results SET result = %s WHERE img_name = %s"
            self.db.execute(query_update, (new_value, img_name))
            self._db_result_cache[img_name] = new_value
        except Exception as e:
            print(f"Error updating result: {e}")
    
    def save_temp_results_to_db(self):
        """Save all temporary changes to database."""
        if not self.temp_results:
            print("No changes to save")
            return

        print(f"\n{'='*60}")
        print("SAVING CHANGES TO DATABASE")
        print(f"{'='*60}")
        print(f"Total changes: {len(self.temp_results)}")

        success_count = 0
        failed_count = 0

        for img_name, new_value in self.temp_results.items():
            try:
                self._update_result_in_db(img_name, new_value)
                success_count += 1
            except Exception as e:
                failed_count += 1
                print(f"Error saving {img_name}: {e}")

        print(f"{'='*60}")
        print(f"{success_count} changes saved successfully")
        if failed_count > 0:
            print(f"{failed_count} changes failed")
        print(f"{'='*60}\n")

        self.temp_results.clear()
        print("Temporary changes cleared")

    def sync_images_by_status(self, historic_dir=None, base_dir=None):
        """
        Sync images from the historic folder into position/status folders based on DB status.

        - Creates side_ok/side_nok, front_ok/front_nok, diag_ok/diag_nok if missing.
        - Reads img_name and result from img_results.
        - Extracts position (side/front/diag) from filename.
        - Copies images from historic_dir into the folder that matches position + status (OK/NOK).
        - If an image exists in the wrong status folder for that position, it is removed.
        """
        historic_dir = historic_dir or self.file_manager.join(str(TMP_DISPLAY_DIR), HISTORIC_SUBDIR_NAME)
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

        if not self.db:
            print("No database connection available")
            return

        if not self.file_manager.exists(historic_dir):
            print(f"Historic folder not found: {historic_dir}")
            return

        try:
            rows = self.db.fetch("SELECT img_name, result FROM img_results")
        except Exception as e:
            print(f"Error fetching image results: {e}")
            return

        if not rows:
            print("No image results found in database")
            return

        for row in rows:
            img_name = row.get("img_name") or row.get("name")
            status = row.get("result")

            if not img_name or status is None:
                continue

            status = str(status).strip().upper()
            if status not in ("OK", "NOK"):
                continue

            match = re.search(r"(side|front|diag)", img_name, re.IGNORECASE)
            if not match:
                continue
            position = match.group(1).lower()

            source_path = self.file_manager.join(historic_dir, img_name)
            if not self.file_manager.exists(source_path):
                continue

            target_dir = position_dirs[position][status]
            other_status = "NOK" if status == "OK" else "OK"
            other_dir = position_dirs[position][other_status]

            target_path = self.file_manager.join(target_dir, img_name)
            other_path = self.file_manager.join(other_dir, img_name)

            if self.file_manager.exists(other_path):
                try:
                    self.file_manager.remove(other_path)
                except Exception as e:
                    print(f"Error removing from wrong folder: {other_path} -> {e}")

            if not self.file_manager.exists(target_path):
                try:
                    self.file_manager.copy2(source_path, target_path)
                except Exception as e:
                    print(f"Error copying {img_name} to {target_dir}: {e}")
    
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
        if total_pieces > 0:
            current_piece = total_pieces - self.historic_offset
            # Clamp in case offset is out of range after refresh
            if current_piece < 1:
                current_piece = 1
            if current_piece > total_pieces:
                current_piece = total_pieces
        else:
            current_piece = 0
        counter_text = f"Pieces: {current_piece} of {total_pieces}"
        counter_font_scale = 0.9
        counter_thickness = 2
        counter_color = (0, 0, 0)  # Black text
        
        counter_size = cv2.getTextSize(counter_text, font, counter_font_scale, counter_thickness)[0]
        counter_x = x_draw + (w_draw - counter_size[0]) // 2 - self.right_info_shift
        counter_y = y_draw - 20  # 20 pixels above the button
        
        cv2.putText(canvas, counter_text, (counter_x, counter_y), font, counter_font_scale, 
                   counter_color, counter_thickness)
        
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

        # Place TRASH button to the left of SYNC (shifted further left)
        reset_width = 180
        sync_width = 180
        x_reset = self.width - reset_width - margin_right
        x_sync = x_reset - spacing - sync_width
        x_trash = x_sync - spacing - button_width
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

    def draw_start_stop_button(self, canvas):
        """Draw START/STOP button on canvas (normal mode)"""
        button_width = 180
        button_height = 60
        margin_bottom = 10

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        x_button = (self.width - button_width) // 2
        y_button = self.height - button_height - margin_bottom

        self.start_stop_button_rect = (x_button, y_button, button_width, button_height)
        
        # Check if button is hovered or pressed
        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.start_stop_button_rect)
        is_pressed = is_hovered and self.mouse_button_down
        
        # Scale button on hover
        scale_factor = 0.95 if is_pressed else (1.08 if is_hovered else 1.0)
        scaled_rect = self._scale_rect(self.start_stop_button_rect, scale_factor)
        x_draw, y_draw, w_draw, h_draw = scaled_rect

        if self.remote_requested:
            button_color = (0, 0, 200)
            text_label = "STOP"
        else:
            button_color = (0, 150, 0)
            text_label = "START"
        
        border_color = (0, 0, 0)
        border_width = 2

        cv2.rectangle(canvas, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw),
                     button_color, -1)
        cv2.rectangle(canvas, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw),
                     border_color, border_width)

        text_size = cv2.getTextSize(text_label, font, font_scale, thickness)[0]
        text_x = x_draw + (w_draw - text_size[0]) // 2
        text_y = y_draw + (h_draw + text_size[1]) // 2

        cv2.putText(canvas, text_label, (text_x, text_y), font, font_scale,
                   (255, 255, 255), thickness)

        return canvas

    def draw_trigger_status(self, canvas):
        """Draw trigger status in the top section (normal mode)."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        if self.trigger_active:
            text = "ACTIVATED"
            color = (0, 200, 0)
        elif self.remote_requested:
            text = "INITIATING"
            color = (0, 165, 255)
        else:
            text = "DEACTIVATED"
            color = (0, 0, 200)

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (self.width - text_size[0]) // 2
        text_y = 60
        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, color, thickness)
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

    def _load_camera_icon(self, size):
        if self.camera_icon is not None and self.camera_icon_size == size:
            return
        icon_path = "./resources/camara.png"
        if not self.file_manager.exists(icon_path):
            self.camera_icon = None
            self.camera_icon_size = size
            return
        icon = self.file_manager.read_image(icon_path, cv2.IMREAD_UNCHANGED)
        if icon is None:
            self.camera_icon = None
            self.camera_icon_size = size
            return
        if icon.shape[2] < 4:
            if not self._camera_icon_warned:
                print("[ICON] camara.png has no alpha channel (not transparent).")
                self._camera_icon_warned = True
            icon = cv2.cvtColor(icon, cv2.COLOR_BGR2BGRA)
            icon = self._apply_bg_key(icon)
        else:
            alpha = icon[:, :, 3]
            if np.all(alpha == 255):
                if not self._camera_icon_warned:
                    print("[ICON] camara.png alpha channel is fully opaque.")
                    self._camera_icon_warned = True
                icon = self._apply_bg_key(icon)
        icon = cv2.resize(icon, (size, size), interpolation=cv2.INTER_AREA)
        self.camera_icon = icon
        self.camera_icon_size = size

    def _load_trash_icon(self, size):
        if self.trash_icon is not None and self.trash_icon_size == size:
            return
        icon_path = "./resources/trash.png"
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

    def draw_camera_status(self, canvas):
        """Draw camera connection count with icon in the top section (normal mode)."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.3
        thickness = 3
        count = len(self.connected_cameras)
        text = f"{count}/{self.total_cameras}"
        color = (255, 255, 255)

        icon_size = 40
        self._load_camera_icon(icon_size)
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        padding_right = 30 + self.right_info_shift - 20
        gap = 8
        total_width = icon_size + gap + text_size[0]
        x_right = self.width - padding_right
        x_icon = x_right - total_width
        text_x = x_icon + icon_size + gap

        text_y = 66
        icon_y = text_y - icon_size + 10

        self._overlay_icon(canvas, self.camera_icon, x_icon, icon_y)
        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, color, thickness)
        return canvas

    def draw_sync_message(self, canvas):
        """Draw a short confirmation message after syncing"""
        import time

        if not self.sync_message or self.show_reset_confirm or self.show_delete_confirm:
            return canvas

        if time.time() - self.sync_message_time > 3:
            self.sync_message = ""
            return canvas

        # Dialog dimensions (based on warning dialog style)
        dialog_width = 700
        dialog_height = 220
        dialog_x = (self.width - dialog_width) // 2
        dialog_y = (self.height - dialog_height) // 2

        # Semi-transparent overlay
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

        # Dialog background
        cv2.rectangle(canvas, (dialog_x, dialog_y), (dialog_x + dialog_width, dialog_y + dialog_height),
                     (240, 240, 240), -1)
        cv2.rectangle(canvas, (dialog_x, dialog_y), (dialog_x + dialog_width, dialog_y + dialog_height),
                     (0, 0, 0), 3)

        # OK icon
        icon_x = dialog_x + 60
        icon_y = dialog_y + 80
        cv2.circle(canvas, (icon_x, icon_y), 30, (0, 150, 0), -1)
        cv2.circle(canvas, (icon_x, icon_y), 30, (0, 0, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        ok_text = "OK"
        ok_scale = 1.1
        ok_thickness = 3
        ok_size = cv2.getTextSize(ok_text, font, ok_scale, ok_thickness)[0]
        ok_x = icon_x - (ok_size[0] // 2)
        ok_y = icon_y + (ok_size[1] // 2)
        cv2.putText(canvas, ok_text, (ok_x, ok_y), font, ok_scale, (255, 255, 255), ok_thickness)

        # Message text
        text_x = dialog_x + 130
        text_y = dialog_y + 80

        line1 = "Dataset correctly"
        line2 = "saved"

        font_scale = 1.4
        thickness = 3

        cv2.putText(canvas, line1, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(canvas, line2, (text_x, text_y + 50), font, font_scale, (0, 0, 0), thickness)

        return canvas
    
    def draw_reset_confirmation_dialog(self, canvas):
        """Draw reset confirmation dialog"""
        # Dialog dimensions
        dialog_width = 600
        dialog_height = 250
        dialog_x = (self.width - dialog_width) // 2
        dialog_y = (self.height - dialog_height) // 2
        
        # Semi-transparent overlay
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
        
        # Dialog background
        cv2.rectangle(canvas, (dialog_x, dialog_y), (dialog_x + dialog_width, dialog_y + dialog_height),
                     (240, 240, 240), -1)
        cv2.rectangle(canvas, (dialog_x, dialog_y), (dialog_x + dialog_width, dialog_y + dialog_height),
                     (0, 0, 0), 3)
        
        # Warning icon (exclamation mark)
        icon_x = dialog_x + 50
        icon_y = dialog_y + 70
        cv2.circle(canvas, (icon_x, icon_y), 30, (0, 0, 200), -1)
        cv2.circle(canvas, (icon_x, icon_y), 30, (0, 0, 0), 2)
        
        # Exclamation mark
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "!", (icon_x - 10, icon_y + 15), font, 2.0, (255, 255, 255), 4)
        
        # Warning text
        text_x = dialog_x + 120
        text_y = dialog_y + 70
        
        warning_text1 = "Warning: This will"
        warning_text2 = "permanently delete all data"
        warning_text3 = "Confirm reset operation?"
        
        font_scale = 1.0
        thickness = 2
        
        cv2.putText(canvas, warning_text1, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(canvas, warning_text2, (text_x, text_y + 40), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(canvas, warning_text3, (text_x, text_y + 80), font, font_scale, (0, 0, 0), thickness)
        
        # Buttons
        button_width = 150
        button_height = 50
        button_spacing = 30
        buttons_y = dialog_y + dialog_height - button_height - 30
        
        # Cancel button (left)
        cancel_x = dialog_x + (dialog_width // 2) - button_width - (button_spacing // 2)
        self.reset_cancel_button_rect = (cancel_x, buttons_y, button_width, button_height)
        
        # Check if button is hovered or pressed
        is_cancel_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.reset_cancel_button_rect)
        is_cancel_pressed = is_cancel_hovered and self.mouse_button_down
        
        # Scale button on hover
        scale_factor = 0.95 if is_cancel_pressed else (1.08 if is_cancel_hovered else 1.0)
        scaled_rect = self._scale_rect(self.reset_cancel_button_rect, scale_factor)
        cancel_x_draw, buttons_y_draw, button_width_draw, button_height_draw = scaled_rect
        
        base_cancel_color = (150, 150, 150)
        cancel_color = (150, 150, 150)
        border_color_cancel = (0, 0, 0)
        border_width_cancel = 2

        cv2.rectangle(canvas, (cancel_x_draw, buttons_y_draw), (cancel_x_draw + button_width_draw, buttons_y_draw + button_height_draw),
                     cancel_color, -1)
        cv2.rectangle(canvas, (cancel_x_draw, buttons_y_draw), (cancel_x_draw + button_width_draw, buttons_y_draw + button_height_draw),
                     border_color_cancel, border_width_cancel)
        
        cancel_text = "CANCEL"
        font_scale_buttons = 0.7
        text_size = cv2.getTextSize(cancel_text, font, font_scale_buttons, 2)[0]
        text_x = cancel_x_draw + (button_width_draw - text_size[0]) // 2
        text_y = buttons_y_draw + (button_height_draw + text_size[1]) // 2
        cv2.putText(canvas, cancel_text, (text_x, text_y), font, font_scale_buttons, (255, 255, 255), 2)
        
        # Confirm button (right)
        confirm_x = dialog_x + (dialog_width // 2) + (button_spacing // 2)
        self.reset_confirm_button_rect = (confirm_x, buttons_y, button_width, button_height)
        
        # Check if button is hovered or pressed
        is_confirm_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.reset_confirm_button_rect)
        is_confirm_pressed = is_confirm_hovered and self.mouse_button_down
        
        # Scale button on hover
        scale_factor_confirm = 0.95 if is_confirm_pressed else (1.08 if is_confirm_hovered else 1.0)
        scaled_rect_confirm = self._scale_rect(self.reset_confirm_button_rect, scale_factor_confirm)
        confirm_x_draw, buttons_y_draw_c, button_width_draw_c, button_height_draw_c = scaled_rect_confirm
        
        base_confirm_color = (0, 0, 200)
        confirm_color = (0, 0, 200)
        border_color_confirm = (0, 0, 0)
        border_width_confirm = 2

        cv2.rectangle(canvas, (confirm_x_draw, buttons_y_draw_c), (confirm_x_draw + button_width_draw_c, buttons_y_draw_c + button_height_draw_c),
                     confirm_color, -1)
        cv2.rectangle(canvas, (confirm_x_draw, buttons_y_draw_c), (confirm_x_draw + button_width_draw_c, buttons_y_draw_c + button_height_draw_c),
                     border_color_confirm, border_width_confirm)
        
        confirm_text = "CONFIRM"
        text_size = cv2.getTextSize(confirm_text, font, font_scale_buttons, 2)[0]
        text_x = confirm_x_draw + (button_width_draw_c - text_size[0]) // 2
        text_y = buttons_y_draw_c + (button_height_draw_c + text_size[1]) // 2
        cv2.putText(canvas, confirm_text, (text_x, text_y), font, font_scale_buttons, (255, 255, 255), 2)
        
        return canvas

    def draw_delete_confirmation_dialog(self, canvas):
        """Draw delete-piece confirmation dialog"""
        # Dialog dimensions
        dialog_width = 720
        dialog_height = 300
        dialog_x = (self.width - dialog_width) // 2
        dialog_y = (self.height - dialog_height) // 2

        # Semi-transparent overlay
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

        # Dialog background
        cv2.rectangle(canvas, (dialog_x, dialog_y), (dialog_x + dialog_width, dialog_y + dialog_height),
                     (240, 240, 240), -1)
        cv2.rectangle(canvas, (dialog_x, dialog_y), (dialog_x + dialog_width, dialog_y + dialog_height),
                     (0, 0, 0), 3)

        # Warning icon (trash)
        icon_x = dialog_x + 50
        icon_y = dialog_y + 70
        cv2.circle(canvas, (icon_x, icon_y), 30, (60, 60, 60), -1)
        cv2.circle(canvas, (icon_x, icon_y), 30, (0, 0, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "X", (icon_x - 12, icon_y + 15), font, 1.6, (255, 255, 255), 3)

        # Warning text
        text_x = dialog_x + 130
        text_y = dialog_y + 70

        jsn = self._get_current_historic_jsn() or "N/A"
        warning_text1 = "Delete current piece?"
        warning_text2 = f"JSN: {jsn}"
        warning_text3 = "This will delete local and remote"
        warning_text4 = "images permanently"

        font_scale = 0.85
        thickness = 2

        cv2.putText(canvas, warning_text1, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(canvas, warning_text2, (text_x, text_y + 40), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(canvas, warning_text3, (text_x, text_y + 80), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(canvas, warning_text4, (text_x, text_y + 120), font, font_scale, (0, 0, 0), thickness)

        # Buttons
        button_width = 150
        button_height = 50
        button_spacing = 30
        buttons_y = dialog_y + dialog_height - button_height - 30

        # Cancel button (left)
        cancel_x = dialog_x + (dialog_width // 2) - button_width - (button_spacing // 2)
        self.delete_cancel_button_rect = (cancel_x, buttons_y, button_width, button_height)
        
        # Check if button is hovered or pressed
        is_cancel_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.delete_cancel_button_rect)
        is_cancel_pressed = is_cancel_hovered and self.mouse_button_down
        
        # Scale button on hover
        scale_factor = 0.95 if is_cancel_pressed else (1.08 if is_cancel_hovered else 1.0)
        scaled_rect = self._scale_rect(self.delete_cancel_button_rect, scale_factor)
        cancel_x_draw, buttons_y_draw, button_width_draw, button_height_draw = scaled_rect
        
        cancel_color = (150, 150, 150)
        border_color_cancel = (0, 0, 0)
        border_width_cancel = 2

        cv2.rectangle(canvas, (cancel_x_draw, buttons_y_draw), (cancel_x_draw + button_width_draw, buttons_y_draw + button_height_draw),
                     cancel_color, -1)
        cv2.rectangle(canvas, (cancel_x_draw, buttons_y_draw), (cancel_x_draw + button_width_draw, buttons_y_draw + button_height_draw),
                     border_color_cancel, border_width_cancel)
        
        cancel_text = "CANCEL"
        font_scale_buttons = 0.7
        text_size = cv2.getTextSize(cancel_text, font, font_scale_buttons, 2)[0]
        text_x = cancel_x_draw + (button_width_draw - text_size[0]) // 2
        text_y = buttons_y_draw + (button_height_draw + text_size[1]) // 2
        cv2.putText(canvas, cancel_text, (text_x, text_y), font, font_scale_buttons, (255, 255, 255), 2)
        
        # Confirm button (right)
        confirm_x = dialog_x + (dialog_width // 2) + (button_spacing // 2)
        self.delete_confirm_button_rect = (confirm_x, buttons_y, button_width, button_height)
        
        # Check if button is hovered or pressed
        is_confirm_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.delete_confirm_button_rect)
        is_confirm_pressed = is_confirm_hovered and self.mouse_button_down
        
        # Scale button on hover
        scale_factor_confirm = 0.95 if is_confirm_pressed else (1.08 if is_confirm_hovered else 1.0)
        scaled_rect_confirm = self._scale_rect(self.delete_confirm_button_rect, scale_factor_confirm)
        confirm_x_draw, buttons_y_draw_c, button_width_draw_c, button_height_draw_c = scaled_rect_confirm
        
        confirm_color = (0, 0, 200)
        border_color_confirm = (0, 0, 0)
        border_width_confirm = 2

        cv2.rectangle(canvas, (confirm_x_draw, buttons_y_draw_c), (confirm_x_draw + button_width_draw_c, buttons_y_draw_c + button_height_draw_c),
                     confirm_color, -1)
        cv2.rectangle(canvas, (confirm_x_draw, buttons_y_draw_c), (confirm_x_draw + button_width_draw_c, buttons_y_draw_c + button_height_draw_c),
                     border_color_confirm, border_width_confirm)
        
        confirm_text = "CONFIRM"
        text_size = cv2.getTextSize(confirm_text, font, font_scale_buttons, 2)[0]
        text_x = confirm_x_draw + (button_width_draw_c - text_size[0]) // 2
        text_y = buttons_y_draw_c + (button_height_draw_c + text_size[1]) // 2
        cv2.putText(canvas, confirm_text, (text_x, text_y), font, font_scale_buttons, (255, 255, 255), 2)

        return canvas
    
    def draw_no_images_dialog(self, canvas):
        """Draw no images available dialog"""
        # Dialog dimensions
        dialog_width = 500
        dialog_height = 200
        dialog_x = (self.width - dialog_width) // 2
        dialog_y = (self.height - dialog_height) // 2
        
        # Semi-transparent overlay
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
        
        # Dialog background
        cv2.rectangle(canvas, (dialog_x, dialog_y), (dialog_x + dialog_width, dialog_y + dialog_height),
                     (240, 240, 240), -1)
        cv2.rectangle(canvas, (dialog_x, dialog_y), (dialog_x + dialog_width, dialog_y + dialog_height),
                     (0, 0, 0), 3)
        
        # Warning icon (red, same as reset button)
        icon_x = dialog_x + 50
        icon_y = dialog_y + 70
        cv2.circle(canvas, (icon_x, icon_y), 30, (0, 0, 200), -1)  # Red
        cv2.circle(canvas, (icon_x, icon_y), 30, (0, 0, 0), 2)
        
        # Exclamation mark
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "!", (icon_x - 10, icon_y + 15), font, 1.8, (255, 255, 255), 3)

        # Message text to the right of the icon, not touching the border
        message_text = "No images available"
        font_scale = 0.9  # Smaller font
        thickness = 2
        text_size = cv2.getTextSize(message_text, font, font_scale, thickness)[0]
        # Place text to the right of the icon, with margin
        margin_right_of_icon = 20
        text_x = icon_x + 30 + margin_right_of_icon
        # Vertically center with the icon
        text_y = icon_y + text_size[1] // 2
        # Ensure text does not touch the right border
        max_text_x = dialog_x + dialog_width - 20 - text_size[0]
        if text_x > max_text_x:
            text_x = max_text_x
        cv2.putText(canvas, message_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

        # Confirm Button (blue, same as back button)
        button_width = 120
        button_height = 50
        button_x = dialog_x + (dialog_width - button_width) // 2
        button_y = dialog_y + dialog_height - button_height - 25

        self.no_images_ok_button_rect = (button_x, button_y, button_width, button_height)
        
        # Check if button is hovered or pressed
        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.no_images_ok_button_rect)
        is_pressed = is_hovered and self.mouse_button_down
        
        # Scale button on hover
        scale_factor = 0.95 if is_pressed else (1.08 if is_hovered else 1.0)
        scaled_rect = self._scale_rect(self.no_images_ok_button_rect, scale_factor)
        button_x_draw, button_y_draw, button_width_draw, button_height_draw = scaled_rect
        
        button_color = (132, 36, 2)
        border_color = (0, 0, 0)
        border_width = 2

        cv2.rectangle(canvas, (button_x_draw, button_y_draw), (button_x_draw + button_width_draw, button_y_draw + button_height_draw),
                 button_color, -1)  # Blue
        cv2.rectangle(canvas, (button_x_draw, button_y_draw), (button_x_draw + button_width_draw, button_y_draw + button_height_draw),
                 border_color, border_width)

        ok_text = "OK"
        font_scale_button = 0.9
        text_size_btn = cv2.getTextSize(ok_text, font, font_scale_button, 2)[0]
        text_x_btn = button_x_draw + (button_width_draw - text_size_btn[0]) // 2
        text_y_btn = button_y_draw + (button_height_draw + text_size_btn[1]) // 2
        cv2.putText(canvas, ok_text, (text_x_btn, text_y_btn), font, font_scale_button, (255, 255, 255), 2)

        return canvas
    
    def draw_save_changes_button(self, canvas):
        """Draw SAVE button on canvas"""
        button_width = 180
        button_height = 60
        margin = 30
        margin_top = 30
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # SAVE button (lower right corner)
        x_save = self.width - button_width - margin
        y_save = self.height - button_height - margin_top
        
        self.save_changes_button_rect = (x_save, y_save, button_width, button_height)
        
        # Check if button is hovered or pressed
        is_hovered = self._is_point_in_rect(self.mouse_x, self.mouse_y, self.save_changes_button_rect)
        is_pressed = is_hovered and self.mouse_button_down
        
        # Scale button on hover
        scale_factor = 0.95 if is_pressed else (1.08 if is_hovered else 1.0)
        scaled_rect = self._scale_rect(self.save_changes_button_rect, scale_factor)
        x_draw, y_draw, w_draw, h_draw = scaled_rect
        
        button_color = (132, 36, 2)
        border_color = (0, 0, 0)
        border_width = 2
        
        cv2.rectangle(canvas, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw), 
                     button_color, -1)  # Color #022484
        cv2.rectangle(canvas, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw), 
                     border_color, border_width)
        
        text_save = "SAVE"
        text_size_save = cv2.getTextSize(text_save, font, font_scale, thickness)[0]
        text_x_save = x_draw + (w_draw - text_size_save[0]) // 2
        text_y_save = y_draw + (h_draw + text_size_save[1]) // 2
        
        cv2.putText(canvas, text_save, (text_x_save, text_y_save), font, font_scale, 
                   (255, 255, 255), thickness)
        
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
            key = cv2.waitKey(100)
            
            # Handle keyboard input when search is active
            if self.search_active and key != -1:
                if key == 27:  # ESC key
                    self.search_active = False
                    self.filtered_suggestions = []
                    return True
                elif key == 13:  # ENTER key
                    # If a suggestion is selected, use it
                    if self.selected_suggestion_idx >= 0 and self.selected_suggestion_idx < len(self.filtered_suggestions):
                        self.search_jsn = self.filtered_suggestions[self.selected_suggestion_idx][:21]
                    self.perform_jsn_search()
                    return True
                elif key == 8:  # BACKSPACE key
                    self.search_jsn = self.search_jsn[:-1]
                    self.update_suggestions()
                    return True
                elif key == 0 or key == 2490368:  # UP arrow key
                    if self.filtered_suggestions:
                        self.selected_suggestion_idx = max(-1, self.selected_suggestion_idx - 1)
                    return True
                elif key == 1 or key == 2621440:  # DOWN arrow key
                    if self.filtered_suggestions:
                        self.selected_suggestion_idx = min(len(self.filtered_suggestions) - 1, self.selected_suggestion_idx + 1)
                    return True
                elif 48 <= key <= 57:  # Only numeric characters (0-9)
                    if len(self.search_jsn) < 21:
                        self.search_jsn += chr(key)
                        self.update_suggestions()
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
        cv2.destroyWindow(self.window_name)
        
    def set_color(self, color):
        """Change display color - color in BGR format (Blue, Green, Red)"""
        self.image = np.ones((self.height, self.width, 3), dtype=np.uint8) * np.array(color, dtype=np.uint8)

    def _get_background_canvas(self):
        """Return a writable background canvas using a cached resized template."""
        background_path = "./resources/base_screen.png"
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
                return self._background_cache.copy()

        return np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

    def _get_cached_image(self, img_path):
        """Read an image with a tiny LRU cache keyed by path + mtime."""
        try:
            current_mtime = self.file_manager.getmtime(img_path)
        except Exception:
            self._image_cache.pop(img_path, None)
            return None

        cached = self._image_cache.get(img_path)
        if cached and cached[0] == current_mtime:
            self._image_cache.move_to_end(img_path)
            return cached[1]

        img = self.file_manager.read_image(img_path)
        if img is None:
            self._image_cache.pop(img_path, None)
            return None

        self._image_cache[img_path] = (current_mtime, img)
        self._image_cache.move_to_end(img_path)
        while len(self._image_cache) > self._image_cache_max_items:
            self._image_cache.popitem(last=False)
        return img

    def show_image_grid(self, image_paths, cols=4, rows=2,
        img_size=360, padding=96):
        """Show images without scaling, with fixed padding"""
        canvas = self._get_background_canvas()

        total_width = cols * img_size + (cols - 1) * padding
        total_height = rows * img_size + (rows - 1) * padding

        start_x = (self.width - total_width) // 2
        start_y = (self.height - total_height) // 2
        
        # Clear result buttons list at start
        self.result_buttons = []

        for idx, img_path in enumerate(image_paths):
            if idx >= cols * rows:
                break

            img = self._get_cached_image(img_path)
            if img is None:
                continue

            # Normalize input image size for display tiles.
            if img.shape[0] != img_size or img.shape[1] != img_size:
                interpolation = cv2.INTER_AREA if (img.shape[0] > img_size or img.shape[1] > img_size) else cv2.INTER_LINEAR
                img = cv2.resize(img, (img_size, img_size), interpolation=interpolation)

            row = idx // cols
            col = idx % cols

            x = start_x + col * (img_size + padding)
            y = start_y + row * (img_size + padding)
            
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
                    img = cv2.resize(img, (size_draw, size_draw))

            canvas[y_draw:y_draw + size_draw, x_draw:x_draw + size_draw] = img

            # Show camera label above each image (normal + historic)
            label_text = self._extract_camera_label(img_path)
            if label_text:
                self._draw_camera_label(canvas, x, y, img_size, label_text)
            
            # If we are in historic mode, show result below each image
            if self.historic_mode:
                # Extract filename from path
                img_filename = self.file_manager.basename(img_path)
                
                # First check if there's a temporary change, if not query DB
                if img_filename in self.temp_results:
                    result_text = self.temp_results[img_filename]
                else:
                    cached_result = self._db_result_cache.get(img_filename)
                    if cached_result is not None:
                        result_text = cached_result
                    else:
                        # Query result from DB once, then cache it.
                        try:
                            query = "SELECT result FROM img_results WHERE img_name = %s"
                            result = self.db.fetch(query, (img_filename,))
                            
                            if result and len(result) > 0:
                                result_value = result[0]['result']
                                result_text = str(result_value) if result_value is not None else "N/A"
                            else:
                                result_text = "N/A"
                        except Exception as e:
                            result_text = "Error"
                            print(f"Error querying result for {img_filename}: {e}")
                        self._db_result_cache[img_filename] = result_text
                
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

        # Normal mode: only HISTORIC button
        if not self.historic_mode:
            if self.remote_controls_enabled:
                canvas = self.draw_trigger_status(canvas)
                canvas = self.draw_camera_status(canvas)
            canvas = self.draw_historic_button(canvas)
            if self.remote_controls_enabled:
                canvas = self.draw_start_stop_button(canvas)
            else:
                self.start_stop_button_rect = None
            canvas = self.draw_exit_button(canvas)
            
            # Draw no images dialog if needed
            if self.show_no_images_dialog:
                canvas = self.draw_no_images_dialog(canvas)
        else:
            # Historic mode: show JSN in upper blue bar
            if self.historic_images and len(self.historic_images) > 0:
                # Get JSN from current batch
                current_batch = self.historic_images[self.historic_offset]
                jsn = current_batch[0].split('_')[0] if '_' in current_batch[0] else 'Unknown'
                
                # Check if batch is incomplete
                is_incomplete = len(current_batch) < 7
                
                # Draw JSN in upper blue bar
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3
                
                # Only show JSN at top
                text = f"JSN: {jsn}"
                
                # Get text size to center it
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (self.width - text_size[0]) // 2
                text_y = 60  # Vertical position in blue bar
                
                # Draw text in white
                cv2.putText(canvas, text, (text_x, text_y), font, font_scale, 
                           (255, 255, 255), thickness)
                
                # If batch is incomplete, show message at bottom
                if is_incomplete:
                    incomplete_text = f"INCOMPLETE BATCH ({len(current_batch)}/7)"
                    
                    # Use same font and size as JSN
                    # Get text size to center it
                    text_size_bottom = cv2.getTextSize(incomplete_text, font, font_scale, thickness)[0]
                    text_x_bottom = (self.width - text_size_bottom[0]) // 2
                    text_y_bottom = self.height - 30  # Closer to bottom edge
                    
                    # Draw text in red
                    cv2.putText(canvas, incomplete_text, (text_x_bottom, text_y_bottom), font, 
                               font_scale, (0, 0, 255), thickness)
            
            # Historic mode: navigation arrows, search elements and BACK button
            # Only show left arrow if not at first batch
            if self.historic_offset > 0:
                canvas = self.draw_prev_button(canvas)
            canvas = self.draw_next_button(canvas)
            canvas = self.draw_search_elements(canvas)
            canvas = self.draw_back_button(canvas)
            canvas = self.draw_info_icon(canvas)
            canvas = self.draw_trash_button(canvas)
            canvas = self.draw_sync_button(canvas)
            canvas = self.draw_reset_button(canvas)
            canvas = self.draw_sync_message(canvas)
            
            # Draw piece date dialog if needed
            if self.show_piece_date_dialog:
                canvas = self.draw_piece_date_dialog(canvas)
            
            # Draw confirmation dialog if needed
            if self.show_reset_confirm:
                canvas = self.draw_reset_confirmation_dialog(canvas)
            elif self.show_delete_confirm:
                canvas = self.draw_delete_confirmation_dialog(canvas)

        self.image = canvas
        return self.show()  # Muestra el display y retorna True para actualizar


def check_historic_images():
    """Function to check how many images are in remote vs local historic."""
    import paramiko
    file_manager = FileManager()

    sftp_settings = get_sftp_settings()
    hostname = sftp_settings["hostname"]
    port = sftp_settings["port"]
    username = sftp_settings["username"]
    password = sftp_settings["password"]
    remote_hist_dir = "/media/ssd/hist_display"
    local_hist_dir = file_manager.join(str(TMP_DISPLAY_DIR), HISTORIC_SUBDIR_NAME)

    print("\n" + "="*70)
    print("HISTORIC IMAGES VERIFICATION")
    print("="*70)

    try:
        print("Connecting to SFTP server...")
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=hostname, port=port, username=username, password=password, timeout=10)
        sftp_client = ssh_client.open_sftp()
        print("Connection successful\n")

        image_extensions = (".png", ".jpg", ".jpeg", ".bmp")

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
        print("="*70)
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

        print("="*70)

        sftp_client.close()
        ssh_client.close()

    except paramiko.AuthenticationException:
        print("Error: Authentication failed")
    except paramiko.SSHException as e:
        print(f"SSH Error: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    display = DisplayWindow()
    display.sync_images_by_status()

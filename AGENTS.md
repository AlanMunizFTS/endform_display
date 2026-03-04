# AGENTS.md

## Purpose
This repository runs an OpenCV display app that:
- Pulls latest images from a remote SFTP source.
- Shows them in a 1920x1080 operator UI.
- Supports a historic review mode with DB-backed `OK/NOK` labeling.
- Syncs labeled images into local status folders (`side_*`, `front_*`, `diag_*`).

## Fast Navigation (Read This First)
Open these files first, in this order:
1. `main.py` - app bootstrap/entrypoint.
2. `main_controller.py` - runtime business/orchestration bridge.
3. `display_window.py` - UI rendering + interaction capture.
4. `sftp_app.py` - SSH/SFTP connection, remote process lifecycle, image download behavior.
5. `file_manager.py` - shared local/SFTP file I/O adapter used by display and SFTP flows.
6. `settings.py` - `.env` loader and validated DB/SFTP settings.
7. `paths_config.py` - canonical path constants.
8. `db.py` and `utilities/log.py` - DB and logging helpers.
9. `REPO_MAP.md` - canonical condensed repo map. Use this first for navigation.

If you only need orientation, do not recurse the image folders.

## Canonical Map
Use `REPO_MAP.md` as the single source of truth for structure, file routing, and scan priorities.

## Runtime Flow
1. `main.py` loads env/config and wires `DisplayWindow` + `MainController`.
2. `MainController` bootstraps DB from local historic cache:
   - Starts a background scan of `HISTORIC_LOCAL_DIR` on startup.
   - Inserts missing `img_results.img_name` rows with default `result='OK'`.
   - Blocks historic-mode entry until bootstrap finishes and shows a loading message if the operator tries early.
3. `MainController` manages SFTP connection via `SFTPApp` and runtime loop.
4. Normal mode:
   - Pulls a rotating batch from remote `/media/ssd/test_display`.
   - Mirrors each downloaded image to remote `/media/ssd/hist_display`.
5. Historic mode:
   - Reads local historic cache, groups by JSN, allows search/navigation.
   - Lets user assign/toggle `OK/NOK` and persists to Postgres.
6. Sync action:
   - Reads `img_results`, routes files into `*_ok` / `*_nok`, and removes mismatches.

## Task-to-File Routing
- Change UI layout, buttons, dialogs, draw logic:
  - `display_window.py` (`draw_*`, `mouse_callback`, `show_image_grid`).
- Change remote command start/stop behavior:
  - `main_controller.py` (`start_remote_process`, `stop_remote_process`) and `sftp_app.py`.
- Change image selection/rotation policy in live view:
  - `main_controller.py` (live image rotation/download helpers).
- Change local/SFTP file I/O wrappers (no business rules):
  - `file_manager.py`.
- Change historic filtering/grouping/search behavior:
  - `main_controller.py` (`enter_historic_mode`, `collect_available_jsns`, `perform_jsn_search`).
- Change DB schema usage or SQL:
  - `main_controller.py` DB helpers + `db.py`.
- Change path constants:
  - `paths_config.py`, then replace remaining hardcoded path literals in other modules.
- Validate classified folders against DB/historic (read-only):
  - `tests/test_sync_images_by_status.py`.

## Quick Commands
- Run app: `python main.py`
- Check remote historic folder: `python utilities/check_hist_display.py`
- Inspect logs: `Get-Content log.txt -Tail 200`
- Run unittest (read-only folder/DB validation): `.\venv\Scripts\python.exe -m unittest tests/test_sync_images_by_status.py`
- Run FileManager unittest: `.\venv\Scripts\python.exe -m unittest tests/test_file_manager.py`

## Known Constraints
- Credentials are loaded from `.env` through `settings.py`; required keys are DB and SFTP host/port/user/password values.
- Utility scripts and logger module now live under `utilities/`.
- Test coverage is limited; current coverage includes:
  - `tests/test_sync_images_by_status.py` (DB/folder consistency, read-only)
  - `tests/test_file_manager.py` (unit tests for file I/O adapter)
- UI assumes image tiles are already `360x360` in `show_image_grid`.
- Filename parsing assumes patterns that include JSN prefix and camera/position tokens (`side/front/diag`).
- Production safety: do not run `sync_images_by_status` unless explicitly requested.

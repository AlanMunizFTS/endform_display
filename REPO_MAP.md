# Repo Map

## Entry Points
- `main.py` - production app entrypoint (`main()`).
- `utilities/check_hist_display.py` - manual SFTP historic-folder inspection script.
- `display_window.py` (module `__main__`) - manual sync helper invocation.
- `utilities/db_folder_cleanup.py` - DB-driven cleanup utility for classified folders.
- `tests/test_sync_images_by_status.py` - read-only validation of DB status vs classified folders.
- `tests/test_file_manager.py` - unit tests for `file_manager.py`.

## Core Modules
- `main.py`
  - Orchestrates SFTP connection, UI lifecycle, remote process control, and refresh loop.
  - Loads SFTP credentials from `settings.py`.
- `display_window.py`
  - Contains almost all UI behavior plus historic mode, DB writes, and folder sync.
  - Starts a dedicated background SFTP downloader process for historic cache.
- `sftp_app.py`
  - Handles SSH/SFTP connect/disconnect, remote process streaming, and live image downloads.
  - Emits explicit connection success/failure logs.
- `db.py`
  - Postgres connection pool + execute/fetch helpers.
  - Loads DB credentials from `settings.py`.
  - Emits explicit DB connection success/failure logs.
- `settings.py`
  - Loads `.env` once and exposes validated getters:
    - `get_sftp_settings()`
    - `get_db_settings()`
- `file_manager.py`
  - Shared adapter for path/file operations, image read/write, and SFTP wrappers.
- `utilities/log.py`
  - Central logging and print redirection (`log.txt`).
- `paths_config.py`
  - Canonical path constants and status-folder mapping.

## Folder Layout
- `utilities/` - utility scripts and logging module.
- `resources/` - static UI assets.
- `tmp_display/` - current display images and historic cache root.
- `tmp_display/historic_prueba/` - local historic images used in historic mode.
- `side_ok/`, `side_nok/`, `front_ok/`, `front_nok/`, `diag_ok/`, `diag_nok/` - output folders for sync.
- `Especial_case/` - extra image set.
- `.env` - local credentials file (ignored by git).

## Heavy Folders (Avoid Full Scans)
- `venv/`
- `__pycache__/`
- `side_ok/`, `front_ok/`, `diag_ok/`
- `side_nok/`, `front_nok/`, `diag_nok/`

## Quick Task Routing
- UI/button/render changes: `display_window.py`.
- Remote process start/stop behavior: `main.py` + `sftp_app.py`.
- Live image selection/rotation: `sftp_app.py` (`download_images`).
- Local/SFTP file operation wrappers: `file_manager.py`.
- Historic grouping/search/delete/reset: `display_window.py`.
- DB/query behavior: `display_window.py` + `db.py`.
- Credential/env loading behavior: `settings.py` + `.env`.
- Logging behavior: `utilities/log.py`.
- Utility scripts: `utilities/check_hist_display.py`, `utilities/db_folder_cleanup.py`.
- Path normalization/constants: `paths_config.py`, then replace literals in other modules.

## Quick Commands
- Run app: `python main.py`
- Remote historic check: `python utilities/check_hist_display.py`
- View latest logs: `Get-Content log.txt -Tail 200`
- Run status-folder unittest: `.\venv\Scripts\python.exe -m unittest tests/test_sync_images_by_status.py`
- Run FileManager unittest: `.\venv\Scripts\python.exe -m unittest tests/test_file_manager.py`

# EndForm Display

## Architecture (SOLID Split)

- `main.py`
  - Bootstrap only: logger/env/config wiring and object composition.
- `main_controller.py`
  - Business/orchestration layer: SFTP lifecycle, reconnect policy, remote process control, live/historic flow, DB status updates, sync/reset/delete actions.
  - Runs startup historic DB bootstrap in background and gates historic entry until bootstrap completes.
  - Runs dataset sync asynchronously with progress updates and post-sync verification.
  - Holds runtime constants in `ControllerConfig` (camera IDs, image extensions, intervals, limits, remote command).
- `display_window.py`
  - View layer: rendering + input capture.
  - Shows sync progress modal (loader + percentage/stage) and completion/verification result message.
  - Supports historic navigation with keyboard left/right arrows.
  - Emits user actions to `MainController` and delegates business operations through thin compatibility wrappers.

## Environment Setup

This branch reads DB/SFTP credentials from `.env` through `settings.py`.

Required keys:
- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`

Optional SFTP keys (only required when enabled):
- `APP_SFTP_ENABLED` (`1`, `true`, `yes`, `on` to enable; default disabled)
- `SFTP_HOST`
- `SFTP_PORT`
- `SFTP_USERNAME`
- `SFTP_PASSWORD`

## SFTP Mode (Hybrid Fallback)

- `APP_SFTP_ENABLED` disabled or missing:
  - App runs local-only mode.
- `APP_SFTP_ENABLED` enabled with incomplete SFTP credentials:
  - App runs local-only mode (no startup crash).
- `APP_SFTP_ENABLED` enabled with valid credentials:
  - App tries SFTP at startup.
  - If SFTP is down/unreachable, app continues in local mode and retries every 10 seconds.
  - When SFTP reconnects, remote live tasks resume automatically.

Expected logs include:
- `[LOCAL] Running in local-only mode (SFTP disabled)`
- `[SSH] Initial SFTP connection failed, running local-only fallback`
- `[SSH] Reconnected successfully`

## Startup Historic DB Bootstrap

- On app startup, `MainController.initialize()` starts a background bootstrap for local `HISTORIC_LOCAL_DIR` (`tmp_display/historic`).
- For each local image filename not present in `img_results`, the app inserts a new row with:
  - `img_name = <filename>`
  - `result = 'OK'`
- Existing DB rows are not modified by this bootstrap step.
- The app window opens immediately; historic mode is blocked until bootstrap completes.
- If the user tries to enter historic mode while bootstrap is running, the UI shows a loading message.

## Dataset Sync Loader and Verification

- Pressing `SYNC` starts an asynchronous sync job (UI remains responsive).
- A loading modal is shown with:
  - Progress bar (`0-100%`)
  - Current stage text (`Saving dataset`, `Verifying classification`, `Completed`)
  - Current counter (`x/y`) while processing.
- During this sync, UI interactions are blocked by the modal until completion.
- After sync completes, the app verifies classification consistency using the same core logic as `tests/test_sync_images_by_status.py`.
- Completion messages:
  - Success: `Dataset completed and verified`
  - Verification issues: `Dataset completed but verification failed (N issues)`
  - Runtime failure: `Dataset sync failed: ...`

## Historic Keyboard Navigation

- In historic mode (when search input is not active), use keyboard arrows:
  - `Left Arrow`: previous historic batch
  - `Right Arrow`: next historic batch

## Quick Commands

- Activate virtualenv: `venv\Scripts\activate`
- Run app: `python main.py`
- Check remote historic folder: `python utilities/check_hist_display.py`
- Run status-sync unittest: `.\venv\Scripts\python.exe -m unittest tests/test_sync_images_by_status.py`
- Run FileManager unittest: `.\venv\Scripts\python.exe -m unittest tests/test_file_manager.py`
- Run all unittests: `.\venv\Scripts\python.exe -m unittest`
- View logs: `Get-Content log.txt -Tail 200`

## Database UI (pgAdmin)

- Start services: `docker compose up -d postgres pgadmin`
- Open browser: `http://localhost:5050`
- Login in pgAdmin:
  - Email: `PGADMIN_DEFAULT_EMAIL` from `.env` (default: `admin@endform.com`)
  - Password: `PGADMIN_DEFAULT_PASSWORD` from `.env` (default: `admin123`)
- Add server in pgAdmin:
  - Host: `postgres`
  - Port: `5432`
  - Username: `DB_USER` from `.env`
  - Password: `DB_PASSWORD` from `.env`
  - Database: `DB_NAME` from `.env`

## DB Folder Cleanup (side/front/diag ok/nok)

Deletes files from classification folders using image names from DB (`img_results`).

- Dry-run (no delete): `.\venv\Scripts\python.exe utilities/db_folder_cleanup.py`
- Delete matching files: `.\venv\Scripts\python.exe utilities/db_folder_cleanup.py --delete`
- Custom delete cap: `.\venv\Scripts\python.exe utilities/db_folder_cleanup.py --delete --limit 10`
- Custom base dir: `.\venv\Scripts\python.exe utilities/db_folder_cleanup.py --base-dir .\prueba --delete`

## Notes

- Connection success/failure for SSH and DB is logged to `log.txt`.
- Hard lock behavior for cleanup: by default, max deletions = DB row count (`N`).
- Target folders: `side_ok`, `side_nok`, `front_ok`, `front_nok`, `diag_ok`, `diag_nok`.

## Configuration new lap
-- GIT BASH --
- git fetch --all
- git branch -r
- git checkout -b feature/nueva-funcionalidad origin/feature/nueva-funcionalidad

-- PROJECT --
- .env (make sure you have the achieve)

--TERMINAL--
- venv\scripts\activate
- python -m venv venv   
- pip install -r requirements.txt 

## database form

--LOGIN--
user: admin@endform.com
pass: admin123

--FORM--
- Server Name: local-postgres 
- Host name/address: postgres
- Port: 5432
- Database: postgres 
- User: postgres 
- Password: MartinreaFT$2023#

python utilities\import_local_images_to_db.py

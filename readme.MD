# EndForm Display

## Environment Setup

This branch reads DB/SFTP credentials from `.env` through `settings.py`.

Required keys:
- `SFTP_HOST`
- `SFTP_PORT`
- `SFTP_USERNAME`
- `SFTP_PASSWORD`
- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`

## Quick Commands

- Activate virtualenv: `venv\Scripts\activate`
- Run app: `python main.py`
- Check remote historic folder: `python utilities/check_hist_display.py`
- Run status-sync unittest: `.\venv\Scripts\python.exe -m unittest tests/test_sync_images_by_status.py`
- Run FileManager unittest: `.\venv\Scripts\python.exe -m unittest tests/test_file_manager.py`
- Run all unittests: `.\venv\Scripts\python.exe -m unittest`
- View logs: `Get-Content log.txt -Tail 200`

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

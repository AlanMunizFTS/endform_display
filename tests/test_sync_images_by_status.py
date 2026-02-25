import re
import unittest
from collections import defaultdict
from pathlib import Path

IMPORT_ERRORS = []

try:
    from db import get_db_connection
except Exception as exc:  # pragma: no cover - environment dependent
    get_db_connection = None
    IMPORT_ERRORS.append(f"db import failed: {exc}")

from paths_config import (
    HISTORIC_SUBDIR_NAME,
    STATUS_SYNC_DIRS,
    SYNC_IMAGES_BASE_DIR,
    TMP_DISPLAY_DIR,
)


class TestSyncImagesByStatus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if IMPORT_ERRORS:
            raise unittest.SkipTest("; ".join(IMPORT_ERRORS))

        cls.db = get_db_connection()
        cls.historic_dir = Path(TMP_DISPLAY_DIR) / HISTORIC_SUBDIR_NAME
        cls.base_dir = Path(SYNC_IMAGES_BASE_DIR)

        cls.status_dirs = {
            (position, status): cls.base_dir / folder_name
            for position, statuses in STATUS_SYNC_DIRS.items()
            for status, folder_name in statuses.items()
        }

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "db") and cls.db:
            cls.db.close()

    def _fetch_db_rows(self):
        rows = self.db.fetch("SELECT img_name, result FROM img_results ORDER BY img_name")
        normalized = []
        for row in rows:
            img_name = row.get("img_name") or row.get("name")
            result = row.get("result")
            status = "" if result is None else str(result).strip().upper()
            normalized.append((img_name, status))
        return normalized

    def _list_historic_images(self):
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        if not self.historic_dir.exists():
            return []
        return sorted(
            p.name
            for p in self.historic_dir.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        )

    def test_historic_db_status_folders_are_consistent_without_duplicates(self):
        db_rows = self._fetch_db_rows()
        self.assertGreater(len(db_rows), 0, "img_results returned no rows")

        print(f"\nimg_results rows fetched: {len(db_rows)}")


        historic_images = self._list_historic_images()
        self.assertGreater(len(historic_images), 0, "No image files found in historic folder")
        print(f"\nhistoric image files found: {len(historic_images)}")

        db_status_by_image = defaultdict(set)
        for img_name, status in db_rows:
            if not img_name or status not in ("OK", "NOK"):
                continue
            db_status_by_image[img_name].add(status)

        self.assertGreater(
            len(db_status_by_image),
            0,
            "No valid DB rows with status OK/NOK were found",
        )

        expected_folder_by_image = {}
        missing_db_status = []
        conflicting_db_status = []
        invalid_position = []

        for img_name in historic_images:
            statuses = db_status_by_image.get(img_name, set())
            if not statuses:
                missing_db_status.append(img_name)
                continue
            if len(statuses) > 1:
                conflicting_db_status.append(f"{img_name}: {sorted(statuses)}")
                continue

            match = re.search(r"(side|front|diag)", img_name, re.IGNORECASE)
            if not match:
                invalid_position.append(img_name)
                continue

            position = match.group(1).lower()
            status = next(iter(statuses))
            expected_folder_by_image[img_name] = STATUS_SYNC_DIRS[position][status]

        self.assertFalse(
            missing_db_status,
            "Historic images missing DB status:\n" + "\n".join(missing_db_status[:50]),
        )
        self.assertFalse(
            conflicting_db_status,
            "Historic images with conflicting DB status:\n"
            + "\n".join(conflicting_db_status[:50]),
        )
        self.assertFalse(
            invalid_position,
            "Historic images with unknown position token (side/front/diag):\n"
            + "\n".join(invalid_position[:50]),
        )
        self.assertEqual(
            len(expected_folder_by_image),
            len(historic_images),
            "All historic images must map to a classification folder",
        )

        actual_locations = defaultdict(list)
        for (_, _), folder_path in self.status_dirs.items():
            if not folder_path.exists():
                continue
            for file_path in folder_path.iterdir():
                if file_path.is_file():
                    actual_locations[file_path.name].append(folder_path.name)

        duplicates = {
            img_name: sorted(folder_names)
            for img_name, folder_names in actual_locations.items()
            if len(folder_names) > 1
        }
        self.assertFalse(
            duplicates,
            f"Duplicate images found across status folders: {duplicates}",
        )

        missing = []
        wrong_folder = []
        for img_name, expected_folder in expected_folder_by_image.items():
            actual = actual_locations.get(img_name)
            if not actual:
                missing.append(f"{img_name} (expected in {expected_folder})")
                continue
            if actual[0] != expected_folder:
                wrong_folder.append(
                    f"{img_name} (expected {expected_folder}, found {actual[0]})"
                )

        self.assertFalse(
            missing,
            "Missing classified images:\n" + "\n".join(missing[:50]),
        )
        self.assertFalse(
            wrong_folder,
            "Images in wrong status folder:\n" + "\n".join(wrong_folder[:50]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

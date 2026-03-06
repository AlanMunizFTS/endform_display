import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from file_manager import FileManager


class TestFileManager(unittest.TestCase):
    def setUp(self):
        self.fm = FileManager()
        self.tmp = tempfile.TemporaryDirectory()
        self.base_dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_path_helpers(self):
        joined = self.fm.join(self.base_dir, "a", "b.txt")
        self.assertTrue(joined.endswith(os.path.join("a", "b.txt")))
        self.assertEqual(self.fm.basename(joined), "b.txt")

    def test_local_filesystem_wrappers(self):
        nested_dir = self.fm.join(self.base_dir, "folder", "nested")
        self.fm.makedirs(nested_dir, exist_ok=True)

        self.assertTrue(self.fm.exists(nested_dir))
        self.assertTrue(self.fm.is_dir(nested_dir))

        file_path = self.fm.join(nested_dir, "sample.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("hello")

        self.assertTrue(self.fm.is_file(file_path))
        self.assertIn("sample.txt", self.fm.listdir(nested_dir))
        self.assertIsInstance(self.fm.getmtime(file_path), float)
        self.assertEqual(self.fm.getsize(file_path), 5)

        copied = self.fm.join(nested_dir, "copy.txt")
        self.fm.copy2(file_path, copied)
        self.assertTrue(self.fm.exists(copied))

        self.fm.remove(file_path)
        self.assertFalse(self.fm.exists(file_path))

        top_dir = self.fm.join(self.base_dir, "folder")
        self.fm.rmtree(top_dir)
        self.assertFalse(self.fm.exists(top_dir))

    def test_list_files_by_extension(self):
        names = ["a.PNG", "b.jpg", "c.txt", "d.JPEG"]
        for name in names:
            path = self.fm.join(self.base_dir, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write("x")

        insensitive = self.fm.list_files_by_extension(
            self.base_dir, (".png", ".jpg", ".jpeg")
        )
        self.assertCountEqual(insensitive, ["a.PNG", "b.jpg", "d.JPEG"])

        sensitive = self.fm.list_files_by_extension(
            self.base_dir, (".PNG", ".jpg"), case_sensitive=True
        )
        self.assertCountEqual(sensitive, ["a.PNG", "b.jpg"])

    def test_read_and_write_image_wrappers(self):
        with patch("file_manager.cv2.imread", return_value="image") as imread_mock:
            result = self.fm.read_image("img.png", flags=123)
            self.assertEqual(result, "image")
            imread_mock.assert_called_once_with("img.png", 123)

        with patch("file_manager.cv2.imwrite", return_value=1) as imwrite_mock:
            self.assertTrue(self.fm.write_image("out.png", object()))
            imwrite_mock.assert_called_once()

        with patch("file_manager.cv2.imwrite", return_value=0):
            self.assertFalse(self.fm.write_image("out.png", object()))

    def test_sftp_wrappers(self):
        sftp = MagicMock()
        sftp.listdir.return_value = ["one.png"]
        sftp.stat.return_value = {"ok": True}

        self.fm.sftp_chdir(sftp, "/remote")
        sftp.chdir.assert_called_once_with("/remote")

        sftp.reset_mock()
        self.assertEqual(self.fm.sftp_listdir(sftp), ["one.png"])
        sftp.chdir.assert_not_called()
        sftp.listdir.assert_called_once_with()

        sftp.reset_mock()
        self.assertEqual(self.fm.sftp_listdir(sftp, "/remote2"), ["one.png"])
        sftp.chdir.assert_called_once_with("/remote2")
        sftp.listdir.assert_called_once_with()

        self.fm.sftp_remove(sftp, "/remote/file.txt")
        sftp.remove.assert_called_once_with("/remote/file.txt")

        self.fm.sftp_get(sftp, "a.png", "b.png")
        sftp.get.assert_called_once_with("a.png", "b.png")

        self.fm.sftp_put(sftp, "b.png", "a.png")
        sftp.put.assert_called_once_with("b.png", "a.png")

        self.assertEqual(self.fm.sftp_stat(sftp, "/remote/path"), {"ok": True})
        sftp.stat.assert_called_once_with("/remote/path")

        self.fm.sftp_mkdir(sftp, "/newdir")
        sftp.mkdir.assert_called_once_with("/newdir")


if __name__ == "__main__":
    unittest.main(verbosity=2)

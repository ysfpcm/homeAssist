import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import setup_kokoro


class DownloadFileTests(unittest.TestCase):
    def test_failed_download_is_removed_and_can_be_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "model"

            def fail_download(url, filename, reporthook):
                Path(filename).write_text("partial")
                raise OSError("download interrupted")

            def complete_download(url, filename, reporthook):
                Path(filename).write_text("complete")

            with contextlib.redirect_stdout(io.StringIO()):
                with patch.object(setup_kokoro.urllib.request, "urlretrieve", side_effect=fail_download):
                    with self.assertRaisesRegex(OSError, "download interrupted"):
                        setup_kokoro.download_file("https://example.invalid/model", str(destination))

                self.assertFalse(destination.exists())
                self.assertEqual(list(Path(directory).iterdir()), [])

                with patch.object(setup_kokoro.urllib.request, "urlretrieve", side_effect=complete_download):
                    setup_kokoro.download_file("https://example.invalid/model", str(destination))

            self.assertEqual(destination.read_text(), "complete")
            self.assertEqual(list(Path(directory).iterdir()), [destination])

    def test_failed_download_preserves_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "model"
            destination.write_text("previous")

            def fail_download(url, filename, reporthook):
                Path(filename).write_text("partial")
                raise OSError("download interrupted")

            with contextlib.redirect_stdout(io.StringIO()):
                with patch.object(setup_kokoro.urllib.request, "urlretrieve", side_effect=fail_download):
                    with self.assertRaisesRegex(OSError, "download interrupted"):
                        setup_kokoro.download_file("https://example.invalid/model", str(destination))

            self.assertEqual(destination.read_text(), "previous")
            self.assertEqual(list(Path(directory).iterdir()), [destination])


if __name__ == "__main__":
    unittest.main()

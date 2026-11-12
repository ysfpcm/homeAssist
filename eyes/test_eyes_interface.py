import contextlib
import io
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parent.parent


class WebcamCaptureTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(sys.modules, {"numpy": types.ModuleType("numpy")}):
            with patch("importlib.util.find_spec", return_value=None):
                with patch.object(sys, "path", sys.path[:]):
                    self.capture = runpy.run_path(str(ROOT / "eyes/eyes_interface.py"))[
                        "capture_frame_from_webcam"
                    ]
        self.camera = Mock()
        self.camera.isOpened.return_value = True
        self.cv2 = Mock()
        self.cv2.VideoCapture.return_value = self.camera

    def capture_frame(self):
        with patch.dict(self.capture.__globals__, {"HAS_VISION_LIBS": True, "cv2": self.cv2}):
            with contextlib.redirect_stdout(io.StringIO()):
                return self.capture(2)

    def test_releases_camera_when_open_fails(self):
        self.camera.isOpened.return_value = False

        self.assertIsNone(self.capture_frame())
        self.cv2.VideoCapture.assert_called_once_with(2)
        self.camera.release.assert_called_once_with()

    def test_releases_camera_when_read_raises(self):
        self.camera.read.side_effect = RuntimeError("camera disconnected")

        self.assertIsNone(self.capture_frame())
        self.camera.release.assert_called_once_with()

    def test_releases_camera_after_capture(self):
        frame = object()
        self.camera.read.return_value = (True, frame)

        self.assertIs(self.capture_frame(), frame)
        self.assertEqual(self.camera.read.call_count, 6)
        self.camera.release.assert_called_once_with()

    def test_releases_camera_when_final_read_fails(self):
        self.camera.read.side_effect = [(True, object())] * 5 + [(False, None)]

        self.assertIsNone(self.capture_frame())
        self.camera.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

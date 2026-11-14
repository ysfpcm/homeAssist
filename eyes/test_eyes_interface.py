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


class WebcamMonitorTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(sys.modules, {"numpy": types.ModuleType("numpy")}):
            with patch("importlib.util.find_spec", return_value=None):
                with patch.object(sys, "path", sys.path[:]):
                    self.monitor = runpy.run_path(str(ROOT / "eyes/eyes_interface.py"))[
                        "monitor_webcam_loop"
                    ]
        self.camera = Mock()
        self.camera.isOpened.return_value = True
        self.camera.read.return_value = (True, object())
        self.cv2 = Mock()
        self.cv2.VideoCapture.return_value = self.camera

    def monitor_camera(self, stop_event=None):
        with patch.dict(self.monitor.__globals__, {"HAS_VISION_LIBS": True, "cv2": self.cv2}):
            with contextlib.redirect_stdout(io.StringIO()):
                self.monitor(device_index=2, stop_event=stop_event)

    def test_releases_camera_when_open_fails(self):
        self.camera.isOpened.return_value = False

        self.monitor_camera()

        self.cv2.VideoCapture.assert_called_once_with(2)
        self.camera.release.assert_called_once_with()

    def test_releases_camera_when_initial_read_fails(self):
        self.camera.read.return_value = (False, None)

        self.monitor_camera()

        self.camera.release.assert_called_once_with()

    def test_releases_camera_when_initial_read_raises(self):
        self.camera.read.side_effect = RuntimeError("camera disconnected")

        with self.assertRaisesRegex(RuntimeError, "camera disconnected"):
            self.monitor_camera()

        self.camera.release.assert_called_once_with()

    def test_releases_camera_when_initial_processing_raises(self):
        self.cv2.cvtColor.side_effect = RuntimeError("camera processing failed")

        with self.assertRaisesRegex(RuntimeError, "camera processing failed"):
            self.monitor_camera()

        self.camera.release.assert_called_once_with()

    def test_releases_camera_when_monitoring_read_raises(self):
        self.camera.read.side_effect = [(True, object()), RuntimeError("camera disconnected")]

        with self.assertRaisesRegex(RuntimeError, "camera disconnected"):
            self.monitor_camera()

        self.camera.release.assert_called_once_with()

    def test_releases_camera_after_stop(self):
        stop_event = Mock()
        stop_event.is_set.return_value = True

        self.monitor_camera(stop_event)

        self.camera.release.assert_called_once_with()

    def test_releases_camera_when_monitoring_read_fails(self):
        self.camera.read.side_effect = [(True, object()), (False, None)]

        self.monitor_camera()

        self.camera.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

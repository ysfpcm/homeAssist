import builtins
import contextlib
import io
import os
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import Mock, patch

from brain import database
from brain.llm_client import BASE_SYSTEM_PROMPT


ROOT = Path(__file__).resolve().parent.parent


class ManualScriptTests(unittest.TestCase):
    def test_discovery_does_not_import_hardware_or_change_environment(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"main", "eyes.eyes_interface", "deepface", "cv2", "numpy"}:
                raise AssertionError(f"Unexpected import during discovery: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            with patch.dict(os.environ, {"TTS_PROVIDER": "original"}):
                original_path = sys.path[:]
                for path in ("brain/test_brain.py", "eyes/test_camera.py"):
                    runpy.run_path(str(ROOT / path))
                self.assertEqual(os.environ["TTS_PROVIDER"], "original")
                self.assertEqual(sys.path, original_path)

    def test_camera_reports_missing_vision_dependencies(self):
        script = runpy.run_path(str(ROOT / "eyes/test_camera.py"))
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "deepface":
                raise ImportError("deepface unavailable")
            return original_import(name, *args, **kwargs)

        output = io.StringIO()
        with patch("builtins.__import__", side_effect=guarded_import):
            with contextlib.redirect_stdout(output):
                script["main"]()
        self.assertIn("Missing vision libraries", output.getvalue())

    def test_camera_initializes_when_vision_dependencies_are_available(self):
        script = runpy.run_path(str(ROOT / "eyes/test_camera.py"))
        deepface = types.ModuleType("deepface")
        deepface.DeepFace = object()
        cv2 = types.ModuleType("cv2")
        camera = Mock()
        camera.isOpened.return_value = False
        cv2.VideoCapture = Mock(return_value=camera)
        vision = types.ModuleType("eyes.eyes_interface")
        vision.load_enrolled_faces = Mock(return_value=([], []))
        vision.get_cosine_distance = Mock()
        vision.THRESHOLD = 0.5
        modules = {
            "deepface": deepface,
            "cv2": cv2,
            "numpy": types.ModuleType("numpy"),
            "eyes.eyes_interface": vision,
        }

        output = io.StringIO()
        with patch.dict(sys.modules, modules), patch.object(sys, "path", sys.path[:]):
            with contextlib.redirect_stdout(output):
                script["main"]()
        vision.load_enrolled_faces.assert_called_once_with()
        cv2.VideoCapture.assert_called_once_with(0)
        self.assertIn("Could not open webcam", output.getvalue())

    def test_brain_script_initializes_terminal_history(self):
        script = runpy.run_path(str(ROOT / "brain/test_brain.py"))
        fake_main = types.ModuleType("main")
        fake_main.console_history = [{"role": "user", "content": "old"}]
        fake_main.process_message = lambda text, history: None

        with patch.dict(sys.modules, {"main": fake_main}):
            with patch.dict(os.environ, {"TTS_PROVIDER": "original"}):
                with patch.object(database, "init_db") as init_db:
                    with patch("builtins.input", return_value="exit"):
                        with contextlib.redirect_stdout(io.StringIO()):
                            script["run_tests"]()
                init_db.assert_called_once_with()
                self.assertEqual(os.environ["TTS_PROVIDER"], "terminal")
        self.assertEqual(fake_main.console_history, [{"role": "system", "content": BASE_SYSTEM_PROMPT}])


if __name__ == "__main__":
    unittest.main()

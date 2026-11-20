import contextlib
import io
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parent.parent


class WakeWordStartupTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(sys.modules, {"speech_recognition": types.ModuleType("speech_recognition")}):
            with patch("os.path.exists", return_value=False):
                module = runpy.run_path(str(ROOT / "ears_mouth/audio_interface.py"))
        self.listener = module["WakeWordListener"]()
        self.stream = Mock()
        self.audio = Mock()
        self.audio.open.return_value = self.stream
        self.model = Mock()
        self.model.models = {"hey_jarvis": object()}
        openwakeword = types.ModuleType("openwakeword")
        openwakeword.__path__ = []
        model_module = types.ModuleType("openwakeword.model")
        model_module.Model = Mock(return_value=self.model)
        pyaudio = types.ModuleType("pyaudio")
        pyaudio.PyAudio = Mock(return_value=self.audio)
        pyaudio.paInt16 = 8
        self.modules = {
            "openwakeword": openwakeword,
            "openwakeword.model": model_module,
            "pyaudio": pyaudio,
            "numpy": types.ModuleType("numpy"),
        }

    def start(self, thread):
        with patch.dict(sys.modules, self.modules), patch("threading.Thread", return_value=thread):
            with contextlib.redirect_stdout(io.StringIO()):
                self.listener.start()

    def test_open_failure_terminates_audio_and_allows_retry(self):
        self.audio.open.side_effect = OSError("microphone unavailable")
        retry_audio = Mock()
        retry_audio.open.return_value = self.stream
        self.modules["pyaudio"].PyAudio.side_effect = [self.audio, retry_audio]
        thread = Mock()

        self.start(thread)

        self.audio.terminate.assert_called_once_with()
        self.assertFalse(self.listener.is_listening)
        self.assertIsNone(self.listener.pa)
        self.assertIsNone(self.listener.model)
        self.start(thread)
        self.assertTrue(self.listener.is_listening)
        thread.start.assert_called_once_with()
        self.listener.stop()
        retry_audio.terminate.assert_called_once_with()

    def test_thread_start_failure_releases_stream_and_audio_once(self):
        thread = Mock()
        thread.start.side_effect = RuntimeError("thread unavailable")

        self.start(thread)

        self.stream.close.assert_called_once_with()
        self.audio.terminate.assert_called_once_with()
        self.assertFalse(self.listener.is_listening)
        self.assertIsNone(self.listener.audio_stream)
        self.assertIsNone(self.listener.pa)
        self.assertIsNone(self.listener.model)
        self.listener.stop()
        self.stream.close.assert_called_once_with()
        self.audio.terminate.assert_called_once_with()

    def test_cleanup_continues_if_stream_close_fails(self):
        self.stream.close.side_effect = OSError("stream already closed")
        thread = Mock()
        thread.start.side_effect = RuntimeError("thread unavailable")

        self.start(thread)
        self.listener.stop()

        self.stream.close.assert_called_once_with()
        self.audio.terminate.assert_called_once_with()
        self.assertIsNone(self.listener.audio_stream)
        self.assertIsNone(self.listener.pa)

    def test_listening_error_stops_and_releases_microphone(self):
        self.start(Mock())

        def stop_after_retry(*args, **kwargs):
            if self.stream.read.call_count == 1:
                raise OSError("microphone disconnected")
            self.listener.is_listening = False
            return b""

        self.stream.read.side_effect = stop_after_retry
        output = io.StringIO()
        with patch.dict(sys.modules, self.modules), contextlib.redirect_stdout(output):
            self.listener._listen_loop()

        self.stream.read.assert_called_once()
        self.stream.close.assert_called_once_with()
        self.audio.terminate.assert_called_once_with()
        self.assertFalse(self.listener.is_listening)
        self.assertIsNone(self.listener.audio_stream)
        self.assertIsNone(self.listener.model)
        self.assertIn("microphone disconnected", output.getvalue())

        retry_thread = Mock()
        self.start(retry_thread)
        self.assertTrue(self.listener.is_listening)
        retry_thread.start.assert_called_once_with()
        self.listener.stop()

    def test_model_error_stops_and_releases_microphone(self):
        self.start(Mock())
        self.stream.read.return_value = b"audio"
        self.modules["numpy"].frombuffer = Mock(return_value="samples")
        self.modules["numpy"].int16 = object()
        self.model.predict.side_effect = RuntimeError("prediction failed")

        output = io.StringIO()
        with patch.dict(sys.modules, self.modules), contextlib.redirect_stdout(output):
            self.listener._listen_loop()

        self.model.predict.assert_called_once_with("samples")
        self.stream.close.assert_called_once_with()
        self.audio.terminate.assert_called_once_with()
        self.assertFalse(self.listener.is_listening)
        self.assertIn("prediction failed", output.getvalue())

    def test_read_error_during_stop_does_not_report_failure(self):
        self.start(Mock())

        def stop_during_read(*args, **kwargs):
            self.listener.stop()
            raise OSError("stream closed")

        self.stream.read.side_effect = stop_during_read
        output = io.StringIO()
        with patch.dict(sys.modules, self.modules), contextlib.redirect_stdout(output):
            self.listener._listen_loop()

        self.stream.close.assert_called_once_with()
        self.audio.terminate.assert_called_once_with()
        self.assertNotIn("Wake word listener failed", output.getvalue())


if __name__ == "__main__":
    unittest.main()

import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from brain import database


ROOT = Path(__file__).resolve().parent.parent


def load_brain_cycle():
    modules = {}
    exports = {
        "speech_recognition": {"Recognizer": Mock},
        "ears_mouth.audio_interface": {
            name: Mock() for name in ("listen_and_transcribe", "speak", "WakeWordListener", "play_chime")
        },
        "brain.context_manager": {"get_home_context": Mock(), "get_live_traffic": Mock()},
        "arms.hardware_controller": {"execute_command": Mock()},
        "ears_mouth.messaging_client": {"TelegramBot": Mock()},
        "legs.legs_scheduler": {"ProactiveScheduler": Mock()},
        "eyes.eyes_interface": {
            name: Mock() for name in ("process_motion_event", "process_webcam_detection", "monitor_webcam_loop")
        },
    }
    for name, attributes in exports.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        modules[name] = module
    with patch.dict(sys.modules, modules):
        return runpy.run_path(str(ROOT / "main.py"))["run_brain_cycle"]


class BrainCycleTests(unittest.TestCase):
    def test_null_content_with_tool_call_executes_and_returns_response(self):
        run_brain_cycle = load_brain_cycle()
        tool_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "function": {"name": "get_active_notes", "arguments": {}}}],
        }
        query = Mock(side_effect=[tool_message, {"role": "assistant", "content": "No notes."}])
        history = [{"role": "user", "content": "What are my notes?"}]
        with patch.dict(run_brain_cycle.__globals__, {"query_brain": query}):
            with patch.object(database, "get_active_notes", return_value=[]) as get_notes:
                response = run_brain_cycle(history, tools=[{"type": "function"}])

        self.assertEqual(response, "No notes.")
        get_notes.assert_called_once_with(None)
        self.assertEqual(history[2]["tool_call_id"], "call_1")
        self.assertEqual(history[2]["content"], "No active notes found.")
        self.assertEqual(query.call_count, 2)

    def test_null_content_without_tool_calls_returns_empty_text(self):
        run_brain_cycle = load_brain_cycle()
        query = Mock(return_value={"role": "assistant", "content": None})
        with patch.dict(run_brain_cycle.__globals__, {"query_brain": query}):
            self.assertEqual(run_brain_cycle([]), "")

    def test_null_followup_content_returns_empty_text(self):
        run_brain_cycle = load_brain_cycle()
        query = Mock(side_effect=[
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "function": {"name": "get_active_notes", "arguments": {}}}
            ]},
            {"role": "assistant", "content": None},
        ])
        with patch.dict(run_brain_cycle.__globals__, {"query_brain": query}):
            with patch.object(database, "get_active_notes", return_value=[]):
                self.assertEqual(run_brain_cycle([]), "")

    def test_json_fallback_still_responds(self):
        run_brain_cycle = load_brain_cycle()
        query = Mock(return_value={
            "role": "assistant",
            "content": '```json\n{"name": "respond_to_user", "parameters": {"response_text": "Hello"}}\n```',
        })
        with patch.dict(run_brain_cycle.__globals__, {"query_brain": query}):
            self.assertEqual(run_brain_cycle([]), "Hello")


if __name__ == "__main__":
    unittest.main()

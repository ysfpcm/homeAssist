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
    def test_history_trimming_drops_complete_old_tool_turn(self):
        run_brain_cycle = load_brain_cycle()
        process_message = run_brain_cycle.__globals__["process_message"]
        history = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Old request"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "old", "function": {"name": "get_active_notes", "arguments": {}}}
            ]},
            {"role": "tool", "tool_call_id": "old", "name": "get_active_notes", "content": "[]"},
            {"role": "user", "content": "Please respond naturally to confirm what you did."},
            {"role": "assistant", "content": "No notes."},
        ]
        for number in range(5):
            history.extend([
                {"role": "user", "content": f"Question {number}"},
                {"role": "assistant", "content": f"Answer {number}"},
            ])

        query = Mock(return_value={"role": "assistant", "content": "Latest answer"})
        with patch.dict(run_brain_cycle.__globals__, {"query_brain": query, "get_home_context": Mock(return_value="")}):
            with patch.object(database, "get_active_notes", return_value=[]):
                self.assertEqual(process_message("Latest question", history), "Latest answer")

        self.assertEqual(history[1], {"role": "user", "content": "Question 0"})
        self.assertEqual(history[-2:], [
            {"role": "user", "content": "Latest question"},
            {"role": "assistant", "content": "Latest answer"},
        ])
        self.assertLessEqual(len(history), 16)

    def test_history_trimming_does_not_start_at_tool_confirmation(self):
        run_brain_cycle = load_brain_cycle()
        process_message = run_brain_cycle.__globals__["process_message"]
        call = {"id": "call_1", "function": {"name": "get_active_notes", "arguments": {}}}
        history = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Old request"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_1", "name": "get_active_notes", "content": "[]"},
            {"role": "user", "content": "Please respond naturally to confirm what you did."},
            {"role": "assistant", "content": "No notes."},
        ]
        for number in range(4):
            history.extend([
                {"role": "user", "content": f"Question {number}"},
                {"role": "assistant", "content": f"Answer {number}"},
            ])
        query = Mock(side_effect=[
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "assistant", "content": "Still no notes."},
        ])
        with patch.dict(run_brain_cycle.__globals__, {"query_brain": query, "get_home_context": Mock(return_value="")}):
            with patch.object(database, "get_active_notes", return_value=[]):
                self.assertEqual(process_message("Check again", history), "Still no notes.")

        self.assertEqual(history[1], {"role": "user", "content": "Question 0"})
        self.assertLessEqual(len(history), 16)

    def test_history_trimming_preserves_oversized_latest_tool_turn(self):
        run_brain_cycle = load_brain_cycle()
        process_message = run_brain_cycle.__globals__["process_message"]
        calls = [
            {"id": f"call_{number}", "function": {"name": "get_active_notes", "arguments": {}}}
            for number in range(14)
        ]
        query = Mock(side_effect=[
            {"role": "assistant", "content": None, "tool_calls": calls},
            {"role": "assistant", "content": "No notes."},
            {"role": "assistant", "content": "Hello."},
        ])
        history = [{"role": "system", "content": "system"}]
        with patch.dict(run_brain_cycle.__globals__, {"query_brain": query, "get_home_context": Mock(return_value="")}):
            with patch.object(database, "get_active_notes", return_value=[]):
                self.assertEqual(process_message("Check notes", history), "No notes.")

        self.assertEqual(history[1], {"role": "user", "content": "Check notes"})
        self.assertEqual(history[2]["tool_calls"], calls)
        self.assertEqual([message["tool_call_id"] for message in history[3:17]],
                         [call["id"] for call in calls])
        self.assertEqual(history[-1]["content"], "No notes.")
        with patch.dict(run_brain_cycle.__globals__, {"query_brain": query, "get_home_context": Mock(return_value="")}):
            with patch.object(database, "get_active_notes", return_value=[]):
                self.assertEqual(process_message("Hello", history), "Hello.")
        self.assertEqual(history[1:], [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello."},
        ])

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

    def test_response_tool_does_not_skip_other_tool_calls(self):
        response_call = {
            "id": "response_1",
            "function": {"name": "respond_to_user", "arguments": {"response_text": "I'll remember that."}},
        }
        save_call = {
            "id": "save_1",
            "function": {"name": "save_note", "arguments": {
                "category": "reminder", "target_person": None, "content": "Water the plants",
            }},
        }
        for calls in ([response_call, save_call], [save_call, response_call]):
            with self.subTest(first=calls[0]["id"]):
                run_brain_cycle = load_brain_cycle()
                query = Mock(side_effect=[
                    {"role": "assistant", "content": None, "tool_calls": calls},
                    {"role": "assistant", "content": "I've saved your reminder."},
                ])
                history = []
                with patch.dict(run_brain_cycle.__globals__, {"query_brain": query}):
                    with patch.object(database, "save_note", return_value=42) as save_note:
                        response = run_brain_cycle(history, tools=[{"type": "function"}])

                save_note.assert_called_once_with("reminder", None, "Water the plants")
                self.assertEqual(response, "I've saved your reminder.")
                self.assertEqual(query.call_count, 2)
                self.assertEqual(history[0]["tool_calls"], calls)
                self.assertEqual([message["tool_call_id"] for message in history[1:3]],
                                 [call["id"] for call in calls])
                tool_results = {message["tool_call_id"]: message["content"] for message in history[1:3]}
                self.assertEqual(tool_results["response_1"], "I'll remember that.")
                self.assertEqual(tool_results["save_1"], "Note saved successfully with ID 42.")


if __name__ == "__main__":
    unittest.main()

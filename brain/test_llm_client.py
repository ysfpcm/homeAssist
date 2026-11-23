import json
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from brain import llm_client


class OllamaClientTests(unittest.TestCase):
    def test_request_has_timeout_and_returns_response(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "message": {"role": "assistant", "content": "Hello"},
        }).encode("utf-8")

        with patch.object(llm_client.urllib.request, "urlopen", return_value=response) as urlopen:
            result = llm_client.query_brain([{"role": "user", "content": "Hi"}])

        self.assertEqual(result, {"role": "assistant", "content": "Hello"})
        self.assertGreater(urlopen.call_args.kwargs["timeout"], 0)

    def test_timeout_returns_recoverable_assistant_message(self):
        errors = (TimeoutError("timed out"), urllib.error.URLError(TimeoutError("timed out")))
        for error in errors:
            with self.subTest(error=error):
                with patch.object(llm_client.urllib.request, "urlopen", side_effect=error):
                    result = llm_client.query_brain([{"role": "user", "content": "Hi"}])

                self.assertEqual(result["role"], "assistant")
                self.assertIn("too long", result["content"])

    def test_connection_error_still_reports_unavailable_brain(self):
        with patch.object(llm_client.urllib.request, "urlopen", side_effect=urllib.error.URLError("refused")):
            result = llm_client.query_brain([{"role": "user", "content": "Hi"}])

        self.assertEqual(result["role"], "assistant")
        self.assertIn("check if Ollama is running", result["content"])

    def test_response_read_timeout_returns_recoverable_message(self):
        response = MagicMock()
        response.__enter__.return_value.read.side_effect = TimeoutError("timed out")
        with patch.object(llm_client.urllib.request, "urlopen", return_value=response):
            result = llm_client.query_brain([{"role": "user", "content": "Hi"}])

        self.assertEqual(result["role"], "assistant")
        self.assertIn("too long", result["content"])
        response.__exit__.assert_called_once()

    def test_invalid_response_returns_recoverable_assistant_message(self):
        responses = (
            b'{"message":',
            b'\xff',
            b'null',
            b'[]',
            b'{}',
            b'{"message": null}',
            b'{"message": {"content": "Hello"}}',
            b'{"message": {"role": "user", "content": "Hello"}}',
            b'{"message": {"role": "assistant", "content": 42}}',
        )
        for body in responses:
            with self.subTest(body=body):
                response = MagicMock()
                response.__enter__.return_value.read.return_value = body
                with patch.object(llm_client.urllib.request, "urlopen", return_value=response):
                    result = llm_client.query_brain([{"role": "user", "content": "Hi"}])

                self.assertEqual(result["role"], "assistant")
                self.assertIn("invalid response", result["content"])
                response.__exit__.assert_called_once()

    def test_tool_call_response_with_null_content_is_preserved(self):
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "get_active_notes", "arguments": {}}}],
        }
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({"message": message}).encode("utf-8")
        with patch.object(llm_client.urllib.request, "urlopen", return_value=response):
            result = llm_client.query_brain([{"role": "user", "content": "Hi"}])

        self.assertEqual(result, message)


if __name__ == "__main__":
    unittest.main()

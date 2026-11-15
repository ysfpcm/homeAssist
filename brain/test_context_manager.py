import json
import unittest
from unittest.mock import MagicMock, patch

from brain import context_manager


class TrafficTests(unittest.TestCase):
    def get_traffic(self, payload):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(payload).encode("utf-8")
        with patch.object(context_manager, "GOOGLE_MAPS_API_KEY", "test-key"):
            with patch.object(context_manager, "resolve_destination", return_value=("Store", "Store address")):
                with patch.object(context_manager.urllib.request, "urlopen", return_value=response):
                    return context_manager.get_live_traffic("Store")

    def test_reports_route_error_instead_of_successful_request_status(self):
        result = self.get_traffic({"status": "OK", "rows": [{"elements": [{"status": "ZERO_RESULTS"}]}]})

        self.assertEqual(result, "Traffic API Error: ZERO_RESULTS")

    def test_reports_top_level_error(self):
        self.assertEqual(self.get_traffic({"status": "REQUEST_DENIED"}), "Traffic API Error: REQUEST_DENIED")

    def test_returns_traffic_duration(self):
        result = self.get_traffic({
            "status": "OK",
            "rows": [{"elements": [{"status": "OK", "duration_in_traffic": {"text": "15 mins"}}]}],
        })

        self.assertEqual(result, "Live Traffic to Store (Store address): 15 mins")

    def test_falls_back_to_regular_duration(self):
        result = self.get_traffic({
            "status": "OK",
            "rows": [{"elements": [{"status": "OK", "duration": {"text": "12 mins"}}]}],
        })

        self.assertEqual(result, "Live Traffic to Store (Store address): 12 mins")


if __name__ == "__main__":
    unittest.main()

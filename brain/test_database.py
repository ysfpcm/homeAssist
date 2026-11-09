import contextlib
import io
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from brain import database


class DatabaseConnectionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.database_path = patch.object(
            database, "DB_FILE", os.path.join(self.temp_dir.name, "test.sqlite3")
        )
        self.database_path.start()
        self.addCleanup(self.database_path.stop)

        self.connections = []
        original_connection = database.get_db_connection

        def track_connection():
            connection = original_connection()
            self.connections.append(connection)
            return connection

        self.connection_patch = patch.object(
            database, "get_db_connection", side_effect=track_connection
        )
        self.connection_patch.start()
        self.addCleanup(self.connection_patch.stop)
        self.addCleanup(self.close_connections)
        database.init_db()

    def close_connections(self):
        for connection in self.connections:
            connection.close()

    def assert_connections_closed(self):
        for connection in self.connections:
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")

    def test_operations_close_connections_and_persist_data(self):
        self.assert_connections_closed()
        note_id = database.save_note("maintenance", "Marc", "Check the heater")
        self.assertEqual(database.get_active_notes("Marc")[0]["content"], "Check the heater")
        self.assertTrue(database.archive_note(note_id))
        self.assertEqual(database.get_active_notes(), [])
        database.add_face_profile("Marc", "owner", b"[1, 2]")
        self.assertEqual(database.get_face_profiles()[0]["name"], "Marc")
        database.log_event("test", "test", "Event details")
        self.assertTrue(any(event["event_type"] == "test" for event in database.get_recent_events()))
        self.assert_connections_closed()

    def test_connections_close_when_operations_fail(self):
        with self.assertRaises(sqlite3.IntegrityError):
            database.save_note(None, "Marc", "Invalid note")
        with contextlib.redirect_stdout(io.StringIO()):
            database.log_event(None, "test", "Invalid event")
        self.assert_connections_closed()
        self.assertEqual(database.get_active_notes(), [])
        self.assert_connections_closed()

    def test_active_notes_order_tied_timestamps_by_newest_id(self):
        latest = database.save_note("todo", "Marc", "latest timestamp")
        older = database.save_note("todo", "Marc", "older")
        newer = database.save_note("todo", "Marc", "newer")
        with contextlib.closing(database.get_db_connection()) as connection, connection:
            connection.execute("UPDATE memory_notes SET created_at = '2026-01-01 00:00:00'")
            connection.execute(
                "UPDATE memory_notes SET created_at = '2026-01-02 00:00:00' WHERE id = ?",
                (latest,),
            )

        expected = [latest, newer, older]
        self.assertEqual([note["id"] for note in database.get_active_notes()], expected)
        self.assertEqual([note["id"] for note in database.get_active_notes("Marc")], expected)
        self.assert_connections_closed()

    def test_recent_events_limit_prefers_newest_when_timestamps_tie(self):
        database.log_event("latest", "test", "")
        database.log_event("older", "test", "")
        database.log_event("newer", "test", "")
        with contextlib.closing(database.get_db_connection()) as connection, connection:
            connection.execute("UPDATE system_events SET timestamp = '2026-01-01 00:00:00'")
            connection.execute(
                "UPDATE system_events SET timestamp = '2026-01-02 00:00:00' WHERE event_type = 'latest'"
            )

        self.assertEqual(
            [event["event_type"] for event in database.get_recent_events(2)],
            ["latest", "newer"],
        )
        self.assert_connections_closed()


if __name__ == "__main__":
    unittest.main()

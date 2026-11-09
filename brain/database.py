import sqlite3
import os
from contextlib import closing

# Go up one directory so gemma.db stays in the main project root folder
DB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gemma.db")

def get_db_connection():
    """Establish and return a connection to the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database tables if they do not exist."""
    print(f"[System - Database]: Initializing database at {DB_FILE}...")
    with closing(get_db_connection()) as conn, conn:
        cursor = conn.cursor()
        
        # 1. Facial Profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS facial_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                role TEXT NOT NULL,
                encoding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 2. System Events Log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                source TEXT NOT NULL,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 3. Long-Term Memory Notes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                target_person TEXT,
                content TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
    print("[System - Database]: Initialization complete.")

# --- MEMORY NOTES API ---

def save_note(category, target_person, content):
    """Save a memory/task note to the database."""
    target_person_clean = target_person.strip() if target_person else None
    with closing(get_db_connection()) as conn, conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO memory_notes (category, target_person, content)
            VALUES (?, ?, ?)
        ''', (category, target_person_clean, content))
        conn.commit()
        note_id = cursor.lastrowid
    log_event("note_saved", "database", f"Saved note ID {note_id} for target '{target_person_clean}': {content}")
    return note_id

def get_active_notes(target_person=None):
    """Retrieve active notes. Optionally filtered by target person."""
    with closing(get_db_connection()) as conn, conn:
        cursor = conn.cursor()
        if target_person:
            cursor.execute('''
                SELECT * FROM memory_notes 
                WHERE is_active = 1 
                  AND (LOWER(target_person) = LOWER(?) OR target_person IS NULL OR target_person = '')
                ORDER BY created_at DESC, id DESC
            ''', (target_person.strip(),))
        else:
            cursor.execute('''
                SELECT * FROM memory_notes 
                WHERE is_active = 1 
                ORDER BY created_at DESC, id DESC
            ''')
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def archive_note(note_id):
    """Archive a note by setting its is_active status to 0."""
    with closing(get_db_connection()) as conn, conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE memory_notes 
            SET is_active = 0, updated_at = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (note_id,))
        conn.commit()
        success = cursor.rowcount > 0
    if success:
        log_event("note_archived", "database", f"Archived note ID {note_id}")
    return success

# --- FACIAL PROFILES API ---

def add_face_profile(name, role, encoding_bytes):
    """Add or update a face profile with its 128-float encoding blob."""
    with closing(get_db_connection()) as conn, conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO facial_profiles (name, role, encoding)
            VALUES (?, ?, ?)
        ''', (name.strip(), role.strip(), encoding_bytes))
        conn.commit()
    log_event("face_profile_updated", "database", f"Registered facial profile for {name} ({role})")

def get_face_profiles():
    """Retrieve all facial profiles and encodings."""
    with closing(get_db_connection()) as conn, conn:
        cursor = conn.cursor()
        cursor.execute('SELECT name, role, encoding FROM facial_profiles')
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

# --- SYSTEM EVENTS API ---

def log_event(event_type, source, details):
    """Write an event entry to the system events log."""
    try:
        with closing(get_db_connection()) as conn, conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO system_events (event_type, source, details)
                VALUES (?, ?, ?)
            ''', (event_type, source, details))
            conn.commit()
    except Exception as e:
        print(f"[Database Error]: Failed to log event: {e}")

def get_recent_events(limit=10):
    """Retrieve recent logged events."""
    with closing(get_db_connection()) as conn, conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM system_events 
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

if __name__ == "__main__":
    init_db()

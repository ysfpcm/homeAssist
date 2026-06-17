import threading
import time
import sys
import os
from datetime import datetime

# Add project root directory to path to support brain package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import database

class ProactiveScheduler:
    def __init__(self, brain_query_callback, announcement_callback):
        """
        brain_query_callback: func(conversation_history) -> returns assistant_message dict
        announcement_callback: func(text) -> speaks/sends proactive alert
        """
        self.brain_query = brain_query_callback
        self.announce = announcement_callback
        self.is_running = False
        self.lock = threading.Lock()

    def start(self):
        """Start the background scheduler thread."""
        self.is_running = True
        self.thread = threading.Thread(target=self._scheduler_loop, daemon=True, name="GemmaLegsScheduler")
        self.thread.start()
        print("[System - Legs]: Background proactive scheduler started.")
        return self.thread

    def stop(self):
        """Stop the background scheduler thread."""
        self.is_running = False

    def _scheduler_loop(self):
        while self.is_running:
            try:
                self._check_database_triggers()
            except Exception as e:
                print(f"[Legs Scheduler Error]: {e}")
            
            time.sleep(30)

    def _check_database_triggers(self):
        """Check the database for any pending event triggers that require alerts."""
        pass

    def trigger_face_detected_alert(self, detection_result):
        """
        Manually invoked when the Eyes interface detects a recognized face with active notes.
        Allows immediate announcement of notes.
        """
        name = detection_result["name"]
        role = detection_result["role"]
        notes = detection_result["notes"]
        
        if not notes:
            msg = f"Gemma detected {name} ({role}) entering."
            print(f"[System - Legs]: {msg}")
            
            history = [
                {"role": "system", "content": f"You are Gemma. A camera just recognized {name} ({role}) entering. Generate a natural, brief welcome greeting."},
                {"role": "user", "content": f"{name} is home."}
            ]
            response = self.brain_query(history, trigger_tools=False)
            if response:
                self.announce(response)
            return

        print(f"[System - Legs]: Found active notes for {name}. Asking Brain to deliver them.")
        notes_str = "\n".join([f"- {n}" for n in notes])
        
        prompt = f"Gemma, you just recognized {name} ({role}) on the camera. There are active notes in the database for them:\n{notes_str}\n\nPlease speak to them directly or summarize these notes for them naturally."
        
        history = [
            {"role": "system", "content": "You are Gemma. You are addressing a recognized person at the house to deliver notes stored for them."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.brain_query(history, trigger_tools=False)
        if response:
            self.announce(response)

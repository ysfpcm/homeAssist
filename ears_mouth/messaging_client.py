import json
import urllib.request
import urllib.parse
import urllib.error
import threading
import time
import os
import sys

# Add project root directory to path to support brain package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TELEGRAM_BOT_TOKEN = ""
if os.path.exists(".env"):
    try:
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("TELEGRAM_BOT_TOKEN="):
                    TELEGRAM_BOT_TOKEN = line.strip().split("=", 1)[1].strip('"').strip("'")
    except Exception as e:
        print(f"[Warning]: Failed to read .env file: {e}")

if not TELEGRAM_BOT_TOKEN:
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

class TelegramBot:
    def __init__(self, token=None):
        self.token = token or TELEGRAM_BOT_TOKEN
        self.api_url = f"https://api.telegram.org/bot{self.token}"
        self.last_update_id = 0
        self.is_running = False

    def is_configured(self):
        """Check if a valid token is set."""
        if not self.token:
            return False
        placeholder_keywords = ["your_telegram_bot_token", "token_here", "placeholder"]
        token_lower = self.token.lower()
        if any(keyword in token_lower for keyword in placeholder_keywords):
            return False
        return True

    def _send_request(self, method, data=None):
        """Helper to send POST requests to Telegram Bot API."""
        url = f"{self.api_url}/{method}"
        payload = json.dumps(data).encode('utf-8') if data else None
        headers = {'Content-Type': 'application/json'} if data else {}
        
        req = urllib.request.Request(url, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            err = e.read().decode('utf-8')
            print(f"[Telegram Error]: HTTP {e.code} - {err}")
            if e.code in (401, 404):
                raise ValueError(f"Invalid Telegram Bot Token (HTTP {e.code})")
            return None
        except Exception as e:
            print(f"[Telegram Error]: Connection failed: {e}")
            return None

    def get_updates(self):
        """Fetch incoming messages using long polling."""
        payload = {
            "offset": self.last_update_id + 1,
            "timeout": 10,
            "allowed_updates": ["message"]
        }
        res = self._send_request("getUpdates", payload)
        if res and res.get("ok"):
            return res.get("result", [])
        return []

    def send_message(self, chat_id, text):
        """Send a message to a specific chat."""
        payload = {
            "chat_id": chat_id,
            "text": text
        }
        return self._send_request("sendMessage", payload)

    def poll_loop(self, message_processor):
        """The main polling loop. Passes incoming text to the message_processor callback."""
        print("[System - Telegram]: Polling loop started successfully.")
        self.is_running = True
        
        while self.is_running:
            try:
                updates = self.get_updates()
                for update in updates:
                    self.last_update_id = update["update_id"]
                    message = update.get("message")
                    if not message:
                        continue
                    
                    chat_id = message["chat"]["id"]
                    text = message.get("text")
                    user = message.get("from", {})
                    username = user.get("first_name", "User")
                    
                    if not text:
                        continue
                        
                    print(f"\n[Telegram - Received from {username}]: {text}")
                    
                    from brain.database import log_event
                    log_event("telegram_message_received", f"Telegram (User: {username})", text)
                    
                    response_text = message_processor(text, chat_id)
                    
                    self.send_message(chat_id, response_text)
                    log_event("telegram_message_sent", "Gemma", response_text)
                    
            except ValueError as e:
                if "Invalid Telegram Bot Token" in str(e):
                    print(f"[System - Telegram]: Fatal token error: {e}. Stopping polling.")
                    self.is_running = False
                    break
                else:
                    print(f"[Telegram Bot Poll Loop Error]: {e}")
                    time.sleep(5)
            except Exception as e:
                print(f"[Telegram Bot Poll Loop Error]: {e}")
                time.sleep(5)
            time.sleep(0.5)

    def start_polling_in_background(self, message_processor):
        """Spawns the polling loop in a daemon thread."""
        if not self.is_configured():
            print("[Warning - Telegram]: No TELEGRAM_BOT_TOKEN configured. Telegram interface will be inactive.")
            return None
        
        thread = threading.Thread(
            target=self.poll_loop, 
            args=(message_processor,), 
            daemon=True,
            name="TelegramBotThread"
        )
        thread.start()
        return thread

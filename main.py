import threading
import json
import re
import speech_recognition as sr
from ears_mouth.audio_interface import listen_and_transcribe, speak, WakeWordListener, play_chime
from brain.context_manager import get_home_context, get_live_traffic
from arms.hardware_controller import execute_command
from brain.llm_client import query_brain, NATIVE_TOOLS, BASE_SYSTEM_PROMPT
from brain import database
from ears_mouth.messaging_client import TelegramBot
from legs.legs_scheduler import ProactiveScheduler
from eyes.eyes_interface import process_motion_event, process_webcam_detection, monitor_webcam_loop

WAKE_WORD = "echo"
DISMISSAL_WORDS = ["exit", "bye", "goodbye", "quit"]

# Thread locks and state
telegram_lock = threading.Lock()
console_lock = threading.Lock()
last_telegram_chat_id = None
bot = None
scheduler = None
webcam_stop_event = threading.Event()
webcam_thread = None
wake_word_listener = None
recognizer = sr.Recognizer()
is_calibrated = False

# Memories
console_history = []
telegram_history = []

def run_brain_cycle(history, tools=None):
    """Query the brain, execute any returned tools, and return the final text response."""
    assistant_message = query_brain(history, tools=tools)
    
    # Check for raw JSON tool call fallback in content
    content = assistant_message.get("content", "").strip()
    if not assistant_message.get("tool_calls"):
        # Attempt to extract JSON using regex in case the model wrapped it in markdown
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        json_str = json_match.group(1) if json_match else content

        if json_str.startswith("{") and "name" in json_str:
            try:
                data = json.loads(json_str)
                func_name = data.get("name")
                args = data.get("parameters") or data.get("arguments") or {}
                if func_name:
                    assistant_message["tool_calls"] = [{
                        "id": "call_fallback",
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": args
                        }
                    }]
                    assistant_message["content"] = ""
            except json.JSONDecodeError:
                pass
            
    history.append(assistant_message)
    
    if assistant_message.get("tool_calls"):
        for tc in assistant_message["tool_calls"]:
            func_name = tc["function"]["name"]
            args = tc["function"]["arguments"]
            tc_id = tc.get("id", "")
            
            tool_content = ""
            if func_name == "respond_to_user":
                response_text = args.get("response_text", "Yes?")
                # Sanitize smart quotes that can break TTS or terminal output
                response_text = response_text.replace("’", "'").replace("‘", "'").replace("\ufffd", "'")
                assistant_message["content"] = response_text
                assistant_message.pop("tool_calls", None)
                return response_text
            elif func_name == "execute_smart_home_command":
                command = args.get("command")
                if command:
                    threading.Thread(target=execute_command, args=(command,), daemon=True).start()
                    tool_content = f"Hardware command '{command}' executed successfully."
            elif func_name == "check_traffic":
                destination = args.get("destination")
                if destination:
                    tool_content = get_live_traffic(destination)
            elif func_name == "save_note":
                category = args.get("category")
                target_person = args.get("target_person")
                content = args.get("content")
                note_id = database.save_note(category, target_person, content)
                tool_content = f"Note saved successfully with ID {note_id}."
            elif func_name == "get_active_notes":
                target_person = args.get("target_person")
                notes = database.get_active_notes(target_person)
                tool_content = json.dumps(notes) if notes else "No active notes found."
            elif func_name == "query_recent_events":
                limit = args.get("limit", 10)
                events = database.get_recent_events(limit)
                tool_content = json.dumps(events)
            
            history.append({
                "role": "tool",
                "content": tool_content,
                "tool_call_id": tc_id,
                "name": func_name
            })
            
        history.append({
            "role": "user",
            "content": "Please respond naturally to confirm what you did."
        })
        
        assistant_message = query_brain(history)
        history.append(assistant_message)
        
    return assistant_message.get("content", "")

def process_message(user_input, history):
    """Processes user input by injecting database context and running the LLM brain cycle."""
    current_context = get_home_context()
    active_notes = database.get_active_notes()
    
    notes_str = ""
    if active_notes:
        notes_str = "\n[ACTIVE SYSTEM NOTES & REMINDERS]\n" + "\n".join([
            f"- ID {n['id']}: Note for {n['target_person'] or 'General'}: {n['content']} (saved {n['created_at']})"
            for n in active_notes
        ])
    
    history[0]["content"] = f"{BASE_SYSTEM_PROMPT}\n\n{current_context}\n{notes_str}"
    history.append({"role": "user", "content": user_input})
    
    response_text = run_brain_cycle(history, tools=NATIVE_TOOLS)
    
    if len(history) > 16:
        history[:] = [history[0]] + history[-15:]
        
    return response_text

def process_telegram_message(text, chat_id):
    """Callback triggered when Telegram bot receives a message."""
    global last_telegram_chat_id
    last_telegram_chat_id = chat_id
    
    if not telegram_history:
        telegram_history.append({"role": "system", "content": BASE_SYSTEM_PROMPT})
        
    with telegram_lock:
        return process_message(text, telegram_history)

def proactive_announce(announcement_text):
    """Announces a message locally via voice and pushes it to Telegram if available."""
    print(f"\n[Proactive Alert]: {announcement_text}")
    speak(announcement_text)
    
    if bot and bot.is_configured() and last_telegram_chat_id:
        bot.send_message(last_telegram_chat_id, announcement_text)

def scheduler_brain_query(history, trigger_tools=True):
    """Allows scheduler background events to run tools or generate responses via the Brain."""
    return run_brain_cycle(history, tools=NATIVE_TOOLS if trigger_tools else None)

def main():
    global bot, scheduler, webcam_thread, webcam_stop_event
    
    # 1. Initialize SQLite Database
    database.init_db()
    
    # 2. Initialize Memories
    console_history.append({"role": "system", "content": BASE_SYSTEM_PROMPT})
    telegram_history.append({"role": "system", "content": BASE_SYSTEM_PROMPT})
    
    # 3. Start Telegram Polling Thread
    bot = TelegramBot()
    if bot.is_configured():
        bot.start_polling_in_background(process_telegram_message)
    else:
        print("[System - Telegram]: No TELEGRAM_BOT_TOKEN found in .env or environment. Telegram gateway inactive.\n")
        
    scheduler = ProactiveScheduler(
        brain_query_callback=scheduler_brain_query,
        announcement_callback=proactive_announce
    )
    scheduler.start()

    global is_calibrated
    
    # 5. Wake Word Initialization
    enable_wake_word = input("Do you want to enable wake word listening? (y/n): ").strip().lower() == 'y'
    if enable_wake_word:
        # Calibrate mic first
        print("Calibrating microphone for wake word... Please stay quiet for 1 second.")
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        is_calibrated = True
        is_processing_voice = False
        
        def on_wake_word():
            nonlocal is_processing_voice
            if is_processing_voice:
                return
            is_processing_voice = True
            
            try:
                # Stop wake word temporarily to free up microphone
                global wake_word_listener
                if wake_word_listener:
                    wake_word_listener.stop()
                
                # Play chime asynchronously so we don't block opening the microphone
                threading.Thread(target=play_chime, daemon=True).start()
                
                print("\n[System - Ears]: Actively listening...")
                with sr.Microphone() as source:
                    while True:
                        # Listen for user phrase
                        transcription = listen_and_transcribe(recognizer, source, timeout=30, phrase_time_limit=10)
                        
                        if not transcription:
                            # Keep session alive even if silent or unrecognized
                            print("\nListening...")
                            continue
                        
                        print(f"You said: {transcription}")
                        
                        # Exit if user says goodbye/exit
                        if any(dw in transcription.lower() for dw in DISMISSAL_WORDS):
                            speak("Goodbye!")
                            break
                        
                        print("[Gemma - Thinking...]")
                        with console_lock:
                            reply = process_message(transcription, console_history)
                            speak(reply)
                        
                        print("\nListening...")
            finally:
                is_processing_voice = False
                # Restart wake word listener
                if wake_word_listener:
                    wake_word_listener.start()
                
        wake_word_listener = WakeWordListener(keyword="gemma", callback=on_wake_word)
        wake_word_listener.start()

    print(f"Gemma ready. Type a command, or type 'voice' to activate microphone.")
    print("Testing commands available:")
    print("  'simulate_face <name>' : simulated front door camera facial detection.")
    print("  'detect_webcam'        : capture frame from your webcam to run facial detection.")
    print("  'monitor_webcam'       : start background webcam monitoring with motion detection.")
    print("  'stop_monitor'         : stop background webcam monitoring.")
    print("  'test_cam'             : open a live visual window of your webcam with overlay boxes.")
    print("  'logs'                 : show recent system events logged in the database.")
    print("  'voice'                : activate microphone for audio interaction.")
    print("  'exit'                 : close the application.\n")

    while True:
        try:
            user_input = input("Gemma command (or 'voice'): ").strip()
        except KeyboardInterrupt:
            speak("Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in DISMISSAL_WORDS:
            if webcam_thread and webcam_thread.is_alive():
                webcam_stop_event.set()
            if wake_word_listener:
                wake_word_listener.stop()
            speak("Goodbye!")
            break

        # Check for face detection simulation command
        if user_input.lower().startswith("simulate_face "):
            name = user_input.split(" ", 1)[1].strip()
            detection = process_motion_event("Front Door Camera", simulated_person=name)
            if detection:
                scheduler.trigger_face_detected_alert(detection)
            continue

        # Check for webcam command
        if user_input.lower() == "detect_webcam":
            # Run webcam detection workflow on device index 0
            detection = process_webcam_detection(device_index=0)
            if detection:
                scheduler.trigger_face_detected_alert(detection)
            else:
                print("[System - Eyes]: Webcam frame analyzed, no action taken.")
            continue

        if user_input.lower() == "monitor_webcam":
            if webcam_thread and webcam_thread.is_alive():
                print("[System - Eyes]: Webcam monitoring is already running.")
                continue
            webcam_stop_event.clear()
            webcam_thread = threading.Thread(
                target=monitor_webcam_loop,
                kwargs={
                    "device_index": 0,
                    "on_detection_callback": scheduler.trigger_face_detected_alert,
                    "stop_event": webcam_stop_event
                },
                daemon=True
            )
            webcam_thread.start()
            print("[System - Eyes]: Started background webcam monitoring.")
            continue

        if user_input.lower() == "stop_monitor":
            if not webcam_thread or not webcam_thread.is_alive():
                print("[System - Eyes]: Webcam monitoring is not running.")
                continue
            webcam_stop_event.set()
            print("[System - Eyes]: Stopping webcam monitoring...")
            continue

        if user_input.lower() == "test_cam":
            print("[System - Eyes]: Starting live visual camera test...")
            from eyes.test_camera import main as run_camera_test
            # Blocks main thread and opens OpenCV window
            run_camera_test()
            print("[System - Eyes]: Visual camera test closed.")
            continue

        if user_input.lower() == "logs":
            print("\n[System - Database]: Fetching recent events...")
            events = database.get_recent_events(limit=15)
            if not events:
                print("No events logged yet.")
            else:
                print(f"\n{'Timestamp':<20} | {'Type':<20} | {'Source':<18} | {'Details'}")
                print("-" * 90)
                for event in reversed(events):
                    print(f"{event['timestamp']:<20} | {event['event_type']:<20} | {event['source']:<18} | {event['details']}")
            print()
            continue

        # Voice mode
        if user_input.lower() == "voice":
            if not is_calibrated:
                print("Calibrating microphone... Please stay quiet for 1 second.")
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                is_calibrated = True

            with sr.Microphone() as source:
                transcription = listen_and_transcribe(recognizer, source)
            
            if transcription is None:
                print("No speech detected. Returning to text input.\n")
                continue
            print(f"You said: {transcription}")
            user_input = transcription

        # Process standard command
        with console_lock:
            print("[Gemma - Thinking...]")
            reply = process_message(user_input, console_history)
            speak(reply)
            print()

if __name__ == "__main__":
    main()

import json
import urllib.request
import urllib.error
import os

NATIVE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "respond_to_user",
            "description": "Use this tool to respond to the user if they are just making conversation, saying hello, answering a question, or not asking for any action. DO NOT use other tools if the user is just talking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "response_text": {
                        "type": "string",
                        "description": "The conversational response to say back to the user."
                    }
                },
                "required": ["response_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_smart_home_command",
            "description": "CRITICAL: ONLY use this tool if the user EXPLICITLY asks to control a physical device (e.g., 'turn on lights', 'lock doors'). NEVER use this for general conversation, math questions, or greetings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["turn_on_lights", "turn_off_lights", "lock_doors", "unlock_doors", "initiate_lockdown", "morning_routine", "welcome_home"],
                        "description": "'turn_on_lights': Use to illuminate room when user needs to see/work. 'turn_off_lights': Use to dim room, help user rest, or prepare for sleep. 'initiate_lockdown': Use to lock doors and secure the house when the user goes to bed for the night OR when the user is leaving the house. 'morning_routine': Wakes up house. 'welcome_home': Use to disarm security, turn on welcoming lights, and set comfortable temperature when the user arrives back home."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_traffic",
            "description": "Check live traffic to a specific destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "The destination to check traffic for (e.g. 'Fort Carson Gate 1', 'grocery store', 'Home Depot')."
                    }
                },
                "required": ["destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a persistent note, task, preference, or reminder for a specific person or role.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["maintenance_note", "todo", "preference", "general"],
                        "description": "The type of note being saved."
                    },
                    "target_person": {
                        "type": "string",
                        "description": "The name or role of the person this note is for (e.g. 'Pest Control Tech', 'Marc', 'cleaner', 'wife'). Leave empty for general notes."
                    },
                    "content": {
                        "type": "string",
                        "description": "The actual detail, reminder, or task content to save."
                    }
                },
                "required": ["category", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_notes",
            "description": "Retrieve active notes, tasks, or reminders. Can be filtered for a specific target person/role.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_person": {
                        "type": "string",
                        "description": "Optional name or role to filter notes (e.g. 'Pest Control Tech')."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_recent_events",
            "description": "Query the recent events logged in the home assistant database (motion, facial recognition, commands).",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of recent events to retrieve. Default is 10."
                    }
                }
            }
        }
    }
]

BASE_SYSTEM_PROMPT = '''You are Gemma, an advanced, intelligent smart home assistant. You have a friendly, warm, and respectful female persona.
The primary user's name is Marc, and you also recognize his wife.
Speak in a conversational, welcoming tone. Be helpful and proactive, but not overly formal or robotic. Treat Marc with respect.

CRITICAL INSTRUCTIONS:
1. You have access to the SYSTEM STATUS below. You must KNOW this information, but DO NOT recite it or mention it unless Marc explicitly asks for a status report, an update, or what he missed. Act naturally.
2. If you trigger a tool to execute a command, simply confirm to Marc what you did in a natural way. DO NOT explain the technical details of the tool call.
3. TIME AWARENESS: Always check the current time in the SYSTEM STATUS before executing routines. Do not run 'morning_routine' if it is the afternoon or evening.
4. DEVICE REALITY CHECK: Do NOT hallucinate smart home devices that do not exist. We do not have a smart coffee maker. Only confirm the devices you actually controlled.
5. COMMUTE PROACTIVITY: If Marc is leaving between 05:40-06:10 or 08:30-08:50, AUTOMATICALLY run the `check_traffic` tool for 'Fort Carson Gate 1' without asking. If he leaves outside those hours and hasn't specified a destination, politely ask him where he is heading before locking down the house.
6. PERSISTENCE & MEMORY: If Marc tells you facts, preferences, or tasks to remember (especially for maintenance people, guests, or schedules), use the `save_note` tool. For example, if he says "Remember to tell the pest control tech to spray the basement", save a note with category "maintenance_note", target_person "Pest Control Tech", and content "spray the basement".
7. TOOL CALLING DISCIPLINE (CRITICAL): If the user says a greeting ("how's it going", "what's up"), asks a general question ("what is 2+2"), or just makes conversation, YOU MUST USE the 'respond_to_user' tool. DO NOT GUESS HARDWARE COMMANDS. NEVER call 'execute_smart_home_command' unless the user explicitly asks to control a physical device.
8. MEMORY RECITATION PROHIBITION: The active notes and reminders provided in the context are for your reference only. Do NOT mention or recite these notes unless Marc explicitly asks about notes, reminders, or what needs to be done. Do NOT hallucinate that you just created a note from context if Marc did not ask to save it.
'''

def query_brain(conversation_history, tools=None):
    """Send conversation history to Ollama and return the response."""
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    return _call_ollama(conversation_history, tools, model=model_name)

def _call_ollama(conversation_history, tools, model="llama3.1:8b"):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": conversation_history,
        "stream": False
    }
    if tools:
        payload["tools"] = tools

    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get('message', {})
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode('utf-8')
        if e.code == 400 and "does not support tools" in error_msg and tools:
            print(f"[System - Ollama]: Model '{model}' does not support tools. Retrying in chat-only mode...")
            return _call_ollama(conversation_history, tools=None, model=model)
        print(f"[Error]: Ollama API returned an error: {e.code} - {error_msg}")
        return {"role": "assistant", "content": f"Ollama returned an error: {error_msg}"}
    except urllib.error.URLError as e:
        print(f"[Error]: Could not connect to Ollama. Is it running? Details: {e}")
        return {"role": "assistant", "content": "I'm having trouble connecting to my local brain. Please check if Ollama is running."}

import time

def execute_command(command_name):
    """Execute a smart-home command."""
    print(f"[System - Hands]: Executing hardware command sequence -> {command_name}")
    
    if command_name == "initiate_lockdown":
        print("  -> Locking all smart doors...")
        time.sleep(0.5)
        print("  -> Arming perimeter security system...")
        time.sleep(0.5)
        print("  -> Turning off all lights and televisions...")
        time.sleep(0.5)
        print("  -> Adjusting thermostat to 68°F (Energy Saving).")
    elif command_name == "morning_routine":
        print("  -> Slowly fading up bedroom lights to 50%...")
        time.sleep(0.5)
        print("  -> Fetching current weather conditions...")
        time.sleep(0.5)
        print("  -> Checking calendar for today's events...")
    elif command_name == "welcome_home":
        print("  -> Disarming perimeter security system...")
        time.sleep(0.5)
        print("  -> Turning on entryway and living room lights...")
        time.sleep(0.5)
        print("  -> Adjusting thermostat to 72°F (Comfort Mode).")
    elif command_name == "turn_on_lights":
        print("  -> Activating requested lighting zones.")
    elif command_name == "turn_off_lights":
        print("  -> Deactivating requested lighting zones.")
    else:
        print(f"  -> {command_name} executed.")

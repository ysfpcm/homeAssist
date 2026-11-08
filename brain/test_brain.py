import os

def run_tests():
    os.environ["TTS_PROVIDER"] = "terminal"

    from brain import database
    from main import process_message, console_history
    from brain.llm_client import BASE_SYSTEM_PROMPT

    print("==================================================")
    print(" BRAIN TESTING ENVIRONMENT")
    print("==================================================")
    print("Type a message to test how Gemma's brain responds.")
    print("This will bypass microphones, wake words, and TTS.")
    print("Type 'exit' to quit.")
    print("==================================================\n")

    # Initialize the database so context fetching works
    database.init_db()

    # Initialize the baseline history with the system prompt
    console_history.clear()
    console_history.append({"role": "system", "content": BASE_SYSTEM_PROMPT})

    while True:
        try:
            user_input = input("Test Query: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break
                
            print("[\"Gemma\" is thinking...]")
            response = process_message(user_input, console_history)
            
            # Note: process_message automatically calls speak(), which in terminal mode just prints.
            # But we can also print the direct return value to be sure.
            print(f"-> Returned Value: {response}\n")
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    run_tests()

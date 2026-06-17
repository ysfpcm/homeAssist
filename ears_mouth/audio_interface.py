import os
import speech_recognition as sr
from abc import ABC, abstractmethod

def load_env_file():
    """Manually load .env file if it exists, without requiring python-dotenv."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    try:
                        key, val = line.split("=", 1)
                        val = val.strip().strip("'\"")
                        os.environ[key.strip()] = val
                    except ValueError:
                        pass

# Load environmental variables
load_env_file()

class TTSProvider(ABC):
    @abstractmethod
    def speak(self, text: str):
        pass

class TerminalTTS(TTSProvider):
    """Fallback TTS that just prints to the terminal."""
    def speak(self, text: str):
        print(f"[Assistant]: {text}")



class KokoroTTS(TTSProvider):
    """
    Local high-quality Text-to-Speech using Kokoro-82M via kokoro-onnx.
    To use this:
    1. pip install kokoro-onnx sounddevice soundfile
    2. Download 'kokoro-v1.0.onnx' (or 'kokoro-v1.0.int8.onnx') and 'voices-v1.0.bin' from:
       https://github.com/thewh1teagle/kokoro-onnx/releases
    3. Put them in 'models/kokoro/' and set the paths below.
    """
    def __init__(self, model_path="models/kokoro/kokoro-v1.0.int8.onnx", voices_path="models/kokoro/voices-v1.0.bin", voice="af_sarah", speed=1.0):
        self.model_path = model_path
        self.voices_path = voices_path
        self.voice = voice
        self.speed = speed
        self.kokoro = None
        self._initialized = False
        self._cache = {}

    def _initialize(self):
        if self._initialized:
            return True
        import os
        
        # Check if configured model_path exists. If not, check for the alternate ONNX version (quantized vs full).
        if not os.path.exists(self.model_path):
            if "int8.onnx" in self.model_path:
                alt_path = self.model_path.replace(".int8.onnx", ".onnx")
            else:
                alt_path = self.model_path.replace(".onnx", ".int8.onnx")
            if os.path.exists(alt_path):
                self.model_path = alt_path

        if not os.path.exists(self.model_path) or not os.path.exists(self.voices_path):
            print(f"\n[KokoroTTS Error]: Model files not found at:\n"
                  f"  - Model: {os.path.abspath(self.model_path)}\n"
                  f"  - Voices: {os.path.abspath(self.voices_path)}\n"
                  f"Please run 'setup_kokoro.py' or download them from https://github.com/thewh1teagle/kokoro-onnx/releases\n"
                  f"and place them in the correct directory.")
            return False
        try:
            from kokoro_onnx import Kokoro  # type: ignore
            self.kokoro = Kokoro(self.model_path, self.voices_path)
            self._initialized = True
            
            # Pre-warm cache for common phrases to make them play instantly
            try:
                for phrase in ["Yes?", "Goodbye!"]:
                    samples, sample_rate = self.kokoro.create(phrase, voice=self.voice, speed=self.speed, lang="en-us")
                    self._cache[(phrase, self.voice, self.speed)] = (samples, sample_rate)
            except Exception as e:
                print(f"[KokoroTTS Warning]: Failed to pre-warm cache: {e}")
                
            return True
        except ImportError:
            print("\n[KokoroTTS Error]: kokoro-onnx is not installed.\n"
                  "Please run: pip install kokoro-onnx sounddevice soundfile")
            return False
        except Exception as e:
            print(f"\n[KokoroTTS Error]: Failed to initialize: {e}")
            return False

    def speak(self, text: str):
        print(f"[KokoroTTS - Speaking]: {text}")
        if not self._initialize():
            return

        try:
            import sounddevice as sd
            cache_key = (text, self.voice, self.speed)
            if cache_key in self._cache:
                samples, sample_rate = self._cache[cache_key]
            else:
                samples, sample_rate = self.kokoro.create(text, voice=self.voice, speed=self.speed, lang="en-us")
                # Cache short responses to avoid future latency
                if len(text) < 100:
                    self._cache[cache_key] = (samples, sample_rate)
                    
            sd.play(samples, sample_rate)
            sd.wait()
        except Exception as e:
            print(f"[KokoroTTS Playback Error]: {e}")



class PyTTSx3TTS(TTSProvider):
    """
    Local fast offline TTS using the system's native text-to-speech engine (SAPI5/NSSpeech).
    Generates and plays audio instantly with practically 0% CPU footprint.
    """
    def __init__(self):
        self._engine = None
        self._initialized = False

    def _initialize(self):
        if self._initialized:
            return True
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            # Set a natural speaking rate
            self._engine.setProperty('rate', 175)
            self._initialized = True
            return True
        except Exception as e:
            print(f"[PyTTSx3 Error]: Failed to initialize: {e}")
            return False

    def speak(self, text: str):
        print(f"[PyTTSx3 - Speaking]: {text}")
        if not self._initialize():
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            print(f"[PyTTSx3 Playback Error]: {e}")



# Select TTS provider based on environment variable
provider_name = os.getenv("TTS_PROVIDER", "terminal").lower()

if provider_name == "kokoro":
    current_tts_provider = KokoroTTS()
elif provider_name == "pyttsx3":
    current_tts_provider = PyTTSx3TTS()
else:
    current_tts_provider = TerminalTTS()

def set_tts_provider(provider: TTSProvider):
    global current_tts_provider
    current_tts_provider = provider

def listen_and_transcribe(recognizer, source, timeout=7, phrase_time_limit=10):
    """Listen to the microphone and return the transcribed text."""
    try:
        print("Listening...")
        audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        user_phrase = recognizer.recognize_google(audio, language="en-US")
        return user_phrase
    except sr.WaitTimeoutError:
        return None
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Speech recognition service error: {e}")
        return None

def speak(text):
    """Speak the given text aloud using the configured TTS provider."""
    current_tts_provider.speak(text)

def play_chime():
    """Play a short, pleasant double-beep chime using winsound on Windows."""
    try:
        import winsound
        import time
        winsound.Beep(1200, 100)
        time.sleep(0.05)
        winsound.Beep(1600, 150)
    except Exception as e:
        print(f"[System - Audio]: Chime failed: {e}")

class WakeWordListener:
    """Listens for a wake word in the background using openWakeWord."""
    def __init__(self, keyword="jarvis", callback=None):
        self.keyword = keyword
        self.callback = callback
        self.is_listening = False
        self.model = None
        self.pa = None
        self.audio_stream = None
        self.thread = None
        self.resolved_model_name = None
        self.sample_rate = 16000
        self.chunk_size = 1280

    def start(self):
        import os
        try:
            import openwakeword
            from openwakeword.model import Model
            import pyaudio  # type: ignore
            import numpy as np
        except ImportError:
            print("[System - Ears]: openwakeword, pyaudio, or numpy not installed. Wake word disabled.")
            return

        # Resolve wake word model (built-in vs custom ONNX file in models/wakeword/)
        custom_model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "wakeword"
        )
        custom_model_path = os.path.join(custom_model_dir, f"{self.keyword}.onnx")
        
        mapped_keyword = self.keyword
        if mapped_keyword == "jarvis":
            mapped_keyword = "hey_jarvis"
            
        model_to_load = mapped_keyword
        if os.path.exists(custom_model_path):
            model_to_load = custom_model_path
            print(f"[System - Ears]: Loading custom wake word model from {custom_model_path}")
        else:
            built_in_models = ["hey_jarvis", "alexa", "hey_mycroft", "hey_email", "hey_google", "ok_google", "hey_siri"]
            if mapped_keyword not in built_in_models:
                print(f"[System - Ears]: Custom model '{self.keyword}.onnx' not found in models/wakeword/.")
                print(f"[System - Ears]: Falling back to built-in 'hey_jarvis' wake word listener.")
                model_to_load = "hey_jarvis"
            else:
                print(f"[System - Ears]: Loading built-in wake word model: '{mapped_keyword}'")

        try:
            # We explicitly pass the target model to load so we don't waste memory loading all models
            # and force the use of ONNX Runtime since it is already installed for Kokoro TTS.
            self.model = Model(wakeword_models=[model_to_load], inference_framework="onnx")
            self.resolved_model_name = list(self.model.models.keys())[0]
            
            self.pa = pyaudio.PyAudio()
            self.audio_stream = self.pa.open(
                rate=self.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            self.is_listening = True
            
            import threading
            self.thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.thread.start()
            print(f"[System - Ears]: Wake word listener active (Listening for: {self.keyword})")
        except Exception as e:
            print(f"[System - Ears]: Failed to start wake word listener: {e}")

    def stop(self):
        self.is_listening = False
        if self.audio_stream:
            try:
                self.audio_stream.close()
            except:
                pass
        if self.pa:
            try:
                self.pa.terminate()
            except:
                pass
        self.model = None
            
    def _listen_loop(self):
        import numpy as np
        import threading
        while self.is_listening:
            try:
                # Read raw PCM data from audio stream
                raw_data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                if not raw_data:
                    continue
                
                # Convert buffer to numpy array
                audio_data = np.frombuffer(raw_data, dtype=np.int16)
                
                # Get prediction score
                prediction = self.model.predict(audio_data)
                score = prediction.get(self.resolved_model_name, 0.0)
                
                # If score is high enough, trigger callback
                if score > 0.5:
                    print(f"\n[System - Ears]: Wake word '{self.keyword}' detected! (Score: {score:.2f})")
                    self.is_listening = False  # Stop listening immediately to prevent multiple triggers
                    if self.callback:
                        # Run the callback in a separate thread so it doesn't block this loop.
                        # This allows self.stop() inside the callback to cleanly stop this thread
                        # before a new wake word listener starts.
                        threading.Thread(target=self.callback, daemon=True).start()
            except Exception as e:
                if not self.is_listening:
                    break

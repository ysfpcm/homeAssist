import os
import sys
import urllib.request

MODEL_DIR = "models/kokoro"
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

def download_file(url, dest_path):
    print(f"Downloading {os.path.basename(url)}...")
    def report_hook(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = min(100, read_so_far * 100 // total_size)
            sys.stdout.write(f"\rProgress: [{percent:3d}%] {read_so_far // (1024*1024)}MB / {total_size // (1024*1024)}MB")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\rDownloaded {read_so_far // (1024*1024)}MB")
            sys.stdout.flush()
            
    urllib.request.urlretrieve(url, dest_path, reporthook=report_hook)
    print("\nDownload complete!\n")

def main():
    print("=== Kokoro TTS Local Setup Helper ===")
    
    # 1. Create directory
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created directory: {MODEL_DIR}")
        
    model_path = os.path.join(MODEL_DIR, "kokoro-v1.0.int8.onnx")
    voices_path = os.path.join(MODEL_DIR, "voices-v1.0.bin")
    
    # 2. Download files if missing
    if not os.path.exists(model_path):
        try:
            download_file(MODEL_URL, model_path)
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Please download it manually from https://github.com/thewh1teagle/kokoro-onnx/releases")
            return
    else:
        print("Model file already exists.")
        
    if not os.path.exists(voices_path):
        try:
            download_file(VOICES_URL, voices_path)
        except Exception as e:
            print(f"Error downloading voices file: {e}")
            print("Please download it manually from https://github.com/thewh1teagle/kokoro-onnx/releases")
            return
    else:
        print("Voices file already exists.")
        
    # 3. Create or update .env file
    env_path = ".env"
    has_tts_provider = False
    
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Check if TTS_PROVIDER is already there
        for i, line in enumerate(lines):
            if line.startswith("TTS_PROVIDER="):
                lines[i] = "TTS_PROVIDER=kokoro\n"
                has_tts_provider = True
                break
                
        if not has_tts_provider:
            lines.append("\nTTS_PROVIDER=kokoro\n")
            
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print("Updated .env file to set TTS_PROVIDER=kokoro.")
    else:
        # Create .env from .env.example if it exists, otherwise write new
        if os.path.exists(".env.example"):
            with open(".env.example", "r", encoding="utf-8") as f:
                content = f.read()
            content = content.replace("TTS_PROVIDER=terminal", "TTS_PROVIDER=kokoro")
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(content)
            print("Created .env file from .env.example and configured TTS_PROVIDER=kokoro.")
        else:
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("TTS_PROVIDER=kokoro\n")
            print("Created a new .env file with TTS_PROVIDER=kokoro.")
            
    print("\n=== Setup Complete! ===")
    print("To install the necessary python packages, run:")
    print("  pip install kokoro-onnx sounddevice soundfile")
    print("\nThen run the main application:")
    print("  python main.py")

if __name__ == "__main__":
    main()

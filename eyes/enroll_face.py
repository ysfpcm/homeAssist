import os
import sys
import json

# Add project root directory to path to support brain package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.database import add_face_profile, init_db

try:
    from deepface import DeepFace
    import cv2
    HAS_VISION_LIBS = True
except ImportError:
    HAS_VISION_LIBS = False

def enroll_face_from_file(image_path, name, role):
    """Process an image file, extract face encodings using DeepFace, and store in SQLite."""
    if not HAS_VISION_LIBS:
        print("[Error]: Missing libraries. Run: pip install deepface opencv-python")
        return False

    if not os.path.exists(image_path):
        print(f"[Error]: File not found: {image_path}")
        return False

    print(f"Loading image {image_path}...")
    try:
        # enforce_detection=True throws an exception if no face is found
        results = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=True)
        
        if len(results) == 0:
            print(f"[Error]: No faces detected in {image_path}")
            return False
            
        if len(results) > 1:
            print(f"[Warning]: Multiple faces detected in {image_path}. Enrolling first face found.")
            
        encoding_list = results[0]["embedding"]
        encoding_bytes = json.dumps(encoding_list).encode('utf-8')
        
        add_face_profile(name, role, encoding_bytes)
        print(f"[Success]: Enrolled '{name}' as '{role}' with face encoding.")
        return True
    except ValueError as ve:
        print(f"[Error]: Face detection failed: {ve}")
        return False
    except Exception as e:
        print(f"[Error] Failed to process image: {e}")
        return False

def enroll_mock(name, role):
    """Helper to enroll a mock face encoding for local testing."""
    import random
    mock_encoding = [random.uniform(-1.0, 1.0) for _ in range(128)]
    encoding_bytes = json.dumps(mock_encoding).encode('utf-8')
    add_face_profile(name, role, encoding_bytes)
    print(f"[Mock Success]: Registered mock face encoding for '{name}' as '{role}'.")

def main():
    init_db()  # Ensure database and tables exist
    
    # We require 4 arguments in all cases: script name + name + role + (path OR --mock)
    if len(sys.argv) < 4:
        print("Usage:")
        print("  Real Image: python enroll_face.py <name> <role> <path_to_image>")
        print("  Mock Face:  python enroll_face.py <name> <role> --mock")
        print("\nExamples:")
        print("  python enroll_face.py Marc owner photos/marc.jpg")
        print("  python enroll_face.py \"Pest Control Tech\" maintenance --mock")
        sys.exit(1)
        
    name = sys.argv[1]
    role = sys.argv[2]
    option = sys.argv[3]

    if option == "--mock":
        enroll_mock(name, role)
    else:
        if not HAS_VISION_LIBS:
            print("\n[Warning]: Vision libraries (deepface / cv2) are not installed.")
            print("To enroll mock faces for testing instead, use: --mock")
            print("Otherwise, install dependencies:")
            print("  pip install deepface opencv-python\n")
            sys.exit(1)
        enroll_face_from_file(option, name, role)

if __name__ == "__main__":
    main()

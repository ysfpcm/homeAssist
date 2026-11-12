import os
import sys
import json
import numpy as np
import time

# Add project root directory to path to support brain package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import database

import importlib.util
has_deepface = importlib.util.find_spec("deepface") is not None
has_cv2 = importlib.util.find_spec("cv2") is not None
HAS_VISION_LIBS = has_deepface and has_cv2

if has_cv2:
    import cv2

THRESHOLD = 0.40  # Cosine distance threshold for Facenet

def get_cosine_distance(source, test):
    a = np.matmul(np.transpose(source), test)
    b = np.sum(np.multiply(source, source))
    c = np.sum(np.multiply(test, test))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def load_enrolled_faces():
    """Retrieve enrolled face profiles and deserialize encoding vectors from SQLite."""
    profiles = database.get_face_profiles()
    known_encodings = []
    known_metadata = []
    
    for profile in profiles:
        encoding_bytes = profile.get("encoding")
        if encoding_bytes:
            try:
                encoding_list = json.loads(encoding_bytes.decode('utf-8'))
                if HAS_VISION_LIBS:
                    encoding_array = np.array(encoding_list)
                    known_encodings.append(encoding_array)
                else:
                    known_encodings.append(encoding_list)
                known_metadata.append({
                    "name": profile["name"],
                    "role": profile["role"]
                })
            except Exception as e:
                print(f"[Eyes Error]: Failed to load encoding for {profile['name']}: {e}")
                
    return known_encodings, known_metadata

def recognize_face_in_image(image_path):
    """Detect and match faces in an image file. Return list of matched profile dicts."""
    if not HAS_VISION_LIBS:
        print("[Eyes - Warning]: Vision libraries not available. Cannot process real images.")
        return []

    if not os.path.exists(image_path):
        print(f"[Eyes - Error]: Image file not found: {image_path}")
        return []

    known_encodings, known_metadata = load_enrolled_faces()
    if not known_encodings:
        print("[Eyes - Warning]: No enrolled faces in the database.")
        return []

    try:
        from deepface import DeepFace
        unknown_results = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)
        matches_found = []
        
        # Load image to get dimensions for fallback filtering
        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2] if img is not None else (0, 0)
        
        for unknown_res in unknown_results:
            facial_area = unknown_res.get("facial_area", {})
            if img_w > 0 and img_h > 0:
                if (facial_area.get("x") == 0 and 
                    facial_area.get("y") == 0 and 
                    facial_area.get("w") == img_w and 
                    facial_area.get("h") == img_h):
                    continue
                    
            unknown_enc = unknown_res["embedding"]
            best_match_idx = None
            best_distance = float('inf')
            
            for i, known_enc in enumerate(known_encodings):
                dist = get_cosine_distance(np.array(known_enc), np.array(unknown_enc))
                if dist < best_distance:
                    best_distance = dist
                    best_match_idx = i
                    
            if best_distance <= THRESHOLD and best_match_idx is not None:
                meta = known_metadata[best_match_idx]
                matches_found.append({
                    "name": meta["name"],
                    "role": meta["role"],
                    "confidence": float(1.0 - best_distance)
                })
            else:
                matches_found.append({
                    "name": "Stranger",
                    "role": "visitor",
                    "confidence": 0.0
                })
        return matches_found
    except Exception as e:
        print(f"[Eyes - Error]: Failed to analyze image: {e}")
        return []

def capture_frame_from_webcam(device_index=0):
    """Capture a single frame from the local webcam using OpenCV and return the image frame."""
    if not HAS_VISION_LIBS:
        print("[Eyes - Error]: OpenCV/vision libraries not installed.")
        return None
    try:
        print("[System - Eyes]: Opening webcam...")
        cap = cv2.VideoCapture(device_index)
        try:
            if not cap.isOpened():
                print("[Eyes - Error]: Could not open webcam.")
                return None
            # Allow webcam to auto-exposure/focus (take 5 frames and throw away)
            for _ in range(5):
                cap.read()
            ret, frame = cap.read()
            if not ret:
                print("[Eyes - Error]: Failed to read frame from webcam.")
                return None
            return frame
        finally:
            cap.release()
    except Exception as e:
        print(f"[Eyes - Error]: Failed to capture from webcam: {e}")
        return None

def recognize_face_from_webcam(device_index=0):
    """Captures a frame from the webcam, runs face recognition, and matches against DB."""
    if not HAS_VISION_LIBS:
        print("[Eyes - Error]: OpenCV/deepface not installed.")
        return []
    frame = capture_frame_from_webcam(device_index)
    if frame is None:
        return []
    
    known_encodings, known_metadata = load_enrolled_faces()
    if not known_encodings:
        print("[Eyes - Warning]: No enrolled faces in the database.")
        return []
        
    try:
        print("[System - Eyes]: Extracting face encodings from webcam frame...")
        # DeepFace can process the BGR frame directly
        from deepface import DeepFace
        unknown_results = DeepFace.represent(img_path=frame, model_name="Facenet", enforce_detection=False)
        matches_found = []
        h_frame, w_frame = frame.shape[:2]
        
        for unknown_res in unknown_results:
            facial_area = unknown_res.get("facial_area", {})
            if (facial_area.get("x") == 0 and 
                facial_area.get("y") == 0 and 
                facial_area.get("w") == w_frame and 
                facial_area.get("h") == h_frame):
                continue
                
            unknown_enc = unknown_res["embedding"]
            best_match_idx = None
            best_distance = float('inf')
            
            for i, known_enc in enumerate(known_encodings):
                dist = get_cosine_distance(np.array(known_enc), np.array(unknown_enc))
                if dist < best_distance:
                    best_distance = dist
                    best_match_idx = i
                    
            if best_distance <= THRESHOLD and best_match_idx is not None:
                meta = known_metadata[best_match_idx]
                matches_found.append({
                    "name": meta["name"],
                    "role": meta["role"],
                    "confidence": float(1.0 - best_distance)
                })
            else:
                matches_found.append({
                    "name": "Stranger",
                    "role": "visitor",
                    "confidence": 0.0
                })
        return matches_found
    except Exception as e:
        print(f"[Eyes - Error]: Failed to run recognition on webcam frame: {e}")
        return []

def monitor_webcam_loop(device_index=0, motion_threshold=10000, cooldown_seconds=10, on_detection_callback=None, stop_event=None):
    """
    Continuously capture frames from the webcam.
    Detect motion by calculating the difference between successive frames.
    When motion is detected, run face recognition and trigger the callback.
    """
    if not HAS_VISION_LIBS:
        print("[Eyes - Error]: Vision libraries (opencv/deepface) not installed.")
        return
        
    print("[System - Eyes]: Initializing live webcam monitoring...")
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print("[Eyes - Error]: Could not open webcam.")
        return

    # Read first frame to initialize background
    ret, prev_frame = cap.read()
    if not ret:
        print("[Eyes - Error]: Failed to read from webcam.")
        cap.release()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    last_detection_time = 0
    
    print("[System - Eyes]: Webcam monitoring active. Listening for motion...")
    
    while stop_event is None or not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[Eyes - Error]: Failed to read frame during monitoring.")
            break

        # Motion detection logic (Difference between consecutive frames)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        frame_delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Calculate amount of motion
        motion_pixel_count = np.sum(thresh == 255)
        
        # If motion exceeds threshold and cooldown has passed
        if motion_pixel_count > motion_threshold:
            current_time = time.time()
            if current_time - last_detection_time > cooldown_seconds:
                print(f"\n[System - Eyes]: Motion detected! Running face recognition... (delta: {motion_pixel_count})")
                last_detection_time = current_time
                
                try:
                    from deepface import DeepFace
                    unknown_results = DeepFace.represent(img_path=frame, model_name="Facenet", enforce_detection=False)
                    known_encodings, known_metadata = load_enrolled_faces()
                    h_frame, w_frame = frame.shape[:2]
                    
                    if known_encodings and unknown_results:
                        matches_found = []
                        for unknown_res in unknown_results:
                            facial_area = unknown_res.get("facial_area", {})
                            if (facial_area.get("x") == 0 and 
                                facial_area.get("y") == 0 and 
                                facial_area.get("w") == w_frame and 
                                facial_area.get("h") == h_frame):
                                continue
                                
                            unknown_enc = unknown_res["embedding"]
                            best_match_idx = None
                            best_distance = float('inf')
                            
                            for i, known_enc in enumerate(known_encodings):
                                dist = get_cosine_distance(np.array(known_enc), np.array(unknown_enc))
                                if dist < best_distance:
                                    best_distance = dist
                                    best_match_idx = i
                                    
                            if best_distance <= THRESHOLD and best_match_idx is not None:
                                meta = known_metadata[best_match_idx]
                                matches_found.append({
                                    "name": meta["name"],
                                    "role": meta["role"],
                                    "confidence": float(1.0 - best_distance)
                                })
                        
                        if matches_found:
                            best_match = matches_found[0]
                            recognized_person = best_match["name"]
                            role = best_match["role"]
                            
                            result = _generate_detection_result(recognized_person, role, f"Webcam Device {device_index}")
                            if result and on_detection_callback:
                                # Trigger alert callback (runs in background)
                                on_detection_callback(result)
                        else:
                            print("[System - Eyes]: Motion detected but no enrolled faces recognized.")
                            database.log_event("motion_detected", f"Webcam Device {device_index}", "Motion detected, no enrolled faces recognized.")
                    else:
                        print("[System - Eyes]: Motion detected, but no faces found or no enrolled faces in DB.")
                        database.log_event("motion_detected", f"Webcam Device {device_index}", "Motion detected, no faces found.")
                except Exception as e:
                    print(f"[Eyes - Error]: Error during motion face recognition: {e}")

        prev_gray = gray
        # Sleep a bit to not hog CPU entirely for video capture loop
        time.sleep(0.1)

    cap.release()
    print("[System - Eyes]: Webcam monitoring stopped.")

def process_motion_event(camera_name, simulated_person=None):
    """
    Called when a camera detects motion.
    Analyzes the frame and checks if there are notes for the recognized person.
    """
    print(f"\n[System - Eyes]: Processing motion event on '{camera_name}'...")
    
    recognized_person = None
    role = "visitor"
    
    if simulated_person:
        profiles = database.get_face_profiles()
        matched = [p for p in profiles if p["name"].lower() == simulated_person.lower()]
        if matched:
            recognized_person = matched[0]["name"]
            role = matched[0]["role"]
        else:
            recognized_person = simulated_person
            role = "visitor"
    else:
        # Real camera frame stream capture would go here
        pass

    return _generate_detection_result(recognized_person, role, camera_name)

def process_webcam_detection(device_index=0):
    """
    Triggers a real-time webcam frame grab, processes the face matching, 
    and checks if active notes exist for the recognized person.
    """
    if not HAS_VISION_LIBS:
        print("[Eyes - Error]: Vision libraries (opencv/deepface) not installed.")
        return None

    matches = recognize_face_from_webcam(device_index)
    if not matches:
        print("[System - Eyes]: No faces detected in webcam frame.")
        return None

    # Focus on the first face recognized
    best_match = matches[0]
    recognized_person = best_match["name"]
    role = best_match["role"]

    return _generate_detection_result(recognized_person, role, f"Webcam Device {device_index}")

def _generate_detection_result(recognized_person, role, source_name):
    if not recognized_person:
        print("[System - Eyes]: No recognizable person detected.")
        database.log_event("motion_detected", source_name, "Motion detected, no faces recognized.")
        return None

    print(f"[System - Eyes]: Recognized '{recognized_person}' ({role}) from source '{source_name}'.")
    database.log_event("face_recognized", source_name, f"Recognized {recognized_person} ({role})")

    # Fetch notes for this person
    notes = database.get_active_notes(target_person=recognized_person)
    notes_summary = []
    for note in notes:
        notes_summary.append(note["content"])
        database.log_event("note_retrieved", "eyes", f"Fetched note for {recognized_person}: '{note['content']}'")

    return {
        "name": recognized_person,
        "role": role,
        "notes": notes_summary
    }

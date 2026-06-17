import os
import sys
import json
import time

# Add project root directory to path to support brain package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from deepface import DeepFace
    import cv2
    import numpy as np
    HAS_VISION_LIBS = True
except ImportError:
    HAS_VISION_LIBS = False

from eyes.eyes_interface import load_enrolled_faces, get_cosine_distance, THRESHOLD

def main():
    if not HAS_VISION_LIBS:
        print("[Error]: Missing vision libraries. Make sure deepface, opencv-python, and numpy are installed.")
        return

    print("Loading enrolled faces from database...")
    known_encodings, known_metadata = load_enrolled_faces()
    print(f"Loaded {len(known_encodings)} face profiles.")

    print("\nStarting video stream... (An OpenCV window will pop up shortly)")
    print("Press 'q' inside the video window to quit.")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error]: Could not open webcam.")
        return

    last_recognition_time = 0
    recognition_interval = 1.5  # Run face recognition every 1.5 seconds to keep stream smooth
    matches_found = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error]: Failed to read frame.")
            break

        current_time = time.time()
        # Run recognition on interval
        if current_time - last_recognition_time > recognition_interval:
            last_recognition_time = current_time
            try:
                # enforce_detection=False prevents crash if no face is in frame
                results = DeepFace.represent(img_path=frame, model_name="Facenet", enforce_detection=False)
                
                matches_found = []
                h_frame, w_frame = frame.shape[:2]
                
                if results and known_encodings:
                    for res in results:
                        # Facial area is returned as a dict
                        facial_area = res["facial_area"]
                        # Filter out full-frame fallback when no face is detected
                        if (facial_area["x"] == 0 and 
                            facial_area["y"] == 0 and 
                            facial_area["w"] == w_frame and 
                            facial_area["h"] == h_frame):
                            continue
                        unknown_enc = res["embedding"]
                        
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
                                "facial_area": facial_area,
                                "distance": best_distance
                            })
                        else:
                            matches_found.append({
                                "name": "Stranger",
                                "role": "visitor",
                                "facial_area": facial_area,
                                "distance": best_distance
                            })
            except Exception as e:
                # Ignore recognition errors during active stream
                pass

        # Draw results on frame
        for match in matches_found:
            area = match["facial_area"]
            x, y, w, h = area["x"], area["y"], area["w"], area["h"]
            name = match["name"]
            role = match["role"]
            dist = match["distance"]

            color = (0, 255, 0) if name != "Stranger" else (0, 0, 255)
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # Label
            label = f"{name} ({role}) - dist: {dist:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show frame in a GUI window
        cv2.imshow("Gemma Live Face Recognition Test (Press 'q' to Quit)", frame)

        # Break loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")

if __name__ == "__main__":
    main()

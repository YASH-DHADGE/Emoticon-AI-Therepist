from deepface import DeepFace
import cv2
import time
import os
import traceback

current_mood = 'neutral'  # Global variable to store the latest detected mood
detection_active = False  # Global flag to control detection

def get_current_mood():
    global current_mood
    return current_mood

def start_detection():
    global detection_active
    detection_active = True

def stop_detection():
    global detection_active
    detection_active = False

class EmotionDetector:
    """
    A class to handle face detection and emotion analysis.
    """
    def __init__(self):
        """
        Initializes the face detector.
        """
        # Fix for haarcascade path
        haarcascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(haarcascade_path)

    def detect_emotions(self, frame):
        global current_mood
        """
        Detects faces and analyzes emotions in a single frame.

        Args:
            frame: The video frame (in BGR format) to analyze.

        Returns:
            A list of dictionaries, where each dictionary contains 'box' and 'emotions'
            for a detected face. Returns an empty list if no faces are found.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        print(f"[DEBUG] Number of faces detected: {len(faces)}")
        results = []
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            try:
                # DeepFace expects RGB
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                print(f"[DEBUG] face_rgb shape: {face_rgb.shape}, dtype: {face_rgb.dtype}")
                analysis = DeepFace.analyze(
                    img_path=face_rgb,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='skip',
                    silent=True
                )
                print(f"[DEBUG] DeepFace analysis result: {analysis}")
                if analysis:
                    detected_mood = analysis[0]['dominant_emotion']
                    current_mood = detected_mood  # Update global mood
                    results.append({
                        'box': (x, y, w, h),
                        'emotions': analysis[0]['emotion'],
                        'dominant_emotion': detected_mood
                    })
            except Exception as e:
                print(f"Could not analyze face: {e}")
                traceback.print_exc()
        if not results:
            current_mood = 'none'  # No face detected
        return results

def draw_results(frame, results):
    """Draws emotion detection results on a frame."""
    if not results:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return

    res = results[0]
    x, y, w, h = res['box']
    dominant_emotion = res['dominant_emotion']
    emotions = res['emotions']

    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    

    y_offset = 60
    for emotion, score in emotions.items():
        
        y_offset += 20

def real_time_emotion_detection():
    """Demonstrates real-time emotion detection using a webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = EmotionDetector()
    last_check_time = 0
    check_interval = 0.3  # seconds
    last_results = []
    global detection_active
    detection_active = True  # Start detection by default for demo

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        now = time.time()
        if detection_active and (now - last_check_time >= check_interval):
            last_results = detector.detect_emotions(frame)
            last_check_time = now

        draw_results(frame, last_results)
        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_emotion_detection()
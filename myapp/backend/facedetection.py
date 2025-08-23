import cv2
import time
import os
import traceback

# Global variables
current_mood = 'neutral'  # Global variable to store the latest detected mood
detection_active = False  # Global flag to control detection
_deepface = None  # Lazy loaded DeepFace module

def _get_deepface():
    """Lazy import DeepFace to avoid startup issues"""
    global _deepface
    if _deepface is None:
        try:
            from deepface import DeepFace
            _deepface = DeepFace
            print("[DEBUG] DeepFace imported successfully")
        except Exception as e:
            print(f"Error importing DeepFace: {e}")
            raise ImportError(f"Could not import DeepFace: {e}")
    return _deepface

def get_current_mood():
    global current_mood
    return current_mood

def start_detection():
    global detection_active
    detection_active = True
    print("[DEBUG] Detection started")

def stop_detection():
    global detection_active
    detection_active = False
    print("[DEBUG] Detection stopped")

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
        
        # Verify cascade loaded correctly
        if self.face_cascade.empty():
            print("[ERROR] Could not load face cascade classifier")
            # Try alternative path
            alt_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(alt_path)
            if self.face_cascade.empty():
                raise RuntimeError("Could not load face cascade classifier")

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
        # Get DeepFace when actually needed
        try:
            DeepFace = _get_deepface()
        except ImportError as e:
            print(f"[ERROR] Cannot perform emotion detection: {e}")
            return []

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
                    # Handle both list and dict responses from DeepFace
                    if isinstance(analysis, list):
                        analysis_result = analysis[0]
                    else:
                        analysis_result = analysis
                        
                    detected_mood = analysis_result['dominant_emotion']
                    current_mood = detected_mood  # Update global mood
                    
                    results.append({
                        'box': (x, y, w, h),
                        'emotions': analysis_result['emotion'],
                        'dominant_emotion': detected_mood
                    })
                    
            except Exception as e:
                print(f"Could not analyze face: {e}")
                traceback.print_exc()
                
        if not results:
            current_mood = 'none'  # No face detected
            
        return results

    def detect_emotion_from_image(self, image_path=None, image_array=None):
        """
        Detect emotion from a static image.
        
        Args:
            image_path: Path to image file
            image_array: Numpy array of image
            
        Returns:
            Dictionary with emotion analysis results
        """
        try:
            DeepFace = _get_deepface()
            
            if image_path:
                result = DeepFace.analyze(img_path=image_path, 
                                        actions=['emotion'], 
                                        enforce_detection=False,
                                        silent=True)
            elif image_array is not None:
                result = DeepFace.analyze(img=image_array, 
                                        actions=['emotion'], 
                                        enforce_detection=False,
                                        silent=True)
            else:
                return None
                
            if isinstance(result, list):
                result = result[0]
                
            emotions = result.get('emotion', {})
            dominant_emotion = result.get('dominant_emotion', 'neutral')
            confidence = emotions.get(dominant_emotion, 0.0)
            
            return {
                'emotion': dominant_emotion,
                'confidence': confidence,
                'all_emotions': emotions
            }
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'all_emotions': {}
            }

def draw_results(frame, results):
    """Draws emotion detection results on a frame."""
    if not results:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return

    res = results[0]
    x, y, w, h = res['box']
    dominant_emotion = res['dominant_emotion']
    emotions = res['emotions']

    # Draw rectangle around face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Draw dominant emotion
    cv2.putText(frame, f"Emotion: {dominant_emotion}", 
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Draw emotion scores
    y_offset = 60
    for emotion, score in emotions.items():
        text = f"{emotion}: {score:.2f}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 20

def real_time_emotion_detection():
    """Demonstrates real-time emotion detection using a webcam."""
    # Initialize DeepFace at runtime
    try:
        _get_deepface()
    except ImportError as e:
        print(f"Cannot start real-time detection: {e}")
        return
        
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

    print("Starting real-time emotion detection. Press 'q' to quit.")
    
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
        
        # Display current mood in corner
        cv2.putText(frame, f"Current Mood: {current_mood}", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Emotion Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            start_detection()
        elif key == ord('p'):
            stop_detection()

    cap.release()
    cv2.destroyAllWindows()

# Test function to verify everything works
def test_emotion_detector():
    """Test function to verify the detector works without GUI"""
    try:
        print("Testing EmotionDetector...")
        detector = EmotionDetector()
        
        # Try to capture a frame
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No camera available for testing")
            return False
            
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            results = detector.detect_emotions(frame)
            print(f"Test successful. Detected {len(results)} faces.")
            print(f"Current mood: {get_current_mood()}")
            return True
        else:
            print("Could not capture test frame")
            return False
            
    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test first, then run real-time detection
    if test_emotion_detector():
        real_time_emotion_detection()
    else:
        print("Test failed, skipping real-time detection")
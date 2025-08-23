import cv2
import numpy as np
from fer import FER
import logging

logger = logging.getLogger(__name__)

class EmotionDetector:
    def __init__(self):
        self.detector = FER(mtcnn=True)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_emotion(self, frame):
        try:
            # Convert frame to RGB (FER expects RGB)
            if isinstance(frame, str):  # If frame is a file path
                frame = cv2.imread(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect emotions
            result = self.detector.detect_emotions(frame_rgb)
            
            if result and len(result) > 0:
                # Get the dominant emotion
                emotions = result[0]['emotions']
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                return dominant_emotion
            return 'neutral'
            
        except Exception as e:
            logger.error(f"Error in emotion detection: {str(e)}")
            return 'neutral'
    
    def get_emotion_from_frame(self, frame_data):
        try:
            # Convert frame data to numpy array
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return self.detect_emotion(frame)
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return 'neutral'

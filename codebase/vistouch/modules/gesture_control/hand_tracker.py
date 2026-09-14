import cv2
import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    """
    Wrapper for MediaPipe GestureRecognizer to provide gesture labels and landmarks.
    """
    def __init__(self, model_path=None):
        if model_path is None:
            # Find model path relative to this file: ../../data/gesture_recognizer.task
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, 'data', 'gesture_recognizer.task')
        
        self.base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.GestureRecognizerOptions(
            base_options=self.base_options,
            running_mode=vision.RunningMode.IMAGE
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

    def process_frame(self, frame, timestamp_ms):
        """
        Processes a frame and returns the detected gesture and landmarks.
        """
        # Convert OpenCV BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Recognize gestures
        recognition_result = self.recognizer.recognize(mp_image)

        
        gesture_label = None
        confidence = 0.0
        landmarks = None

        if recognition_result.gestures:
            # Get the first detected gesture
            gesture = recognition_result.gestures[0][0]
            gesture_label = gesture.category_name
            confidence = gesture.score
        
        if recognition_result.hand_landmarks:
            # Get the first hand's landmarks
            landmarks = recognition_result.hand_landmarks[0]
            
        return gesture_label, confidence, landmarks

    def close(self):
        self.recognizer.close()

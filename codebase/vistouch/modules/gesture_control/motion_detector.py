import numpy as np
from collections import deque

class MotionDetector:
    """
    Detects predefined motion gestures from landmark sequences.
    Currently supports: swipe_left, swipe_right, swipe_up, swipe_down
    Requires three fingers to be extended for detection to avoid false positives.
    """
    def __init__(self, buffer_size=10, swipe_threshold=0.15, min_velocity=0.02):
        self.buffer_size = buffer_size
        self.swipe_threshold = swipe_threshold  # normalized displacement threshold
        self.min_velocity = min_velocity        # min velocity per frame to count as swipe
        
        # Buffers for landmark history and three-finger state
        self.landmark_buffer = deque(maxlen=buffer_size)
        self.three_finger_buffer = deque(maxlen=buffer_size)
        
    def update(self, landmarks):
        """
        Update detector with new landmarks.
        Returns: (motion_type, confidence) or (None, 0.0)
        motion_type: 'swipe_left', 'swipe_right', 'swipe_up', 'swipe_down', or None
        """
        if not landmarks:
            self.landmark_buffer.clear()
            self.three_finger_buffer.clear()
            return None, 0.0
            
        # Extract index fingertip (landmark 8) for motion tracking
        index_tip = landmarks[8]
        self.landmark_buffer.append((index_tip.x, index_tip.y))
        
        # Check if three fingers are extended
        three_finger = self._count_extended_fingers(landmarks) >= 3
        self.three_finger_buffer.append(three_finger)
        
        # Need sufficient history
        if len(self.landmark_buffer) < self.buffer_size:
            return None, 0.0
            
        # Only consider frames where three fingers were extended
        valid_frames = [
            (x, y) for (x, y), three_finger in 
            zip(self.landmark_buffer, self.three_finger_buffer) 
            if three_finger
        ]
        
        if len(valid_frames) < self.buffer_size * 0.7:  # Need 70% validity
            return None, 0.0
            
        # Calculate displacement and velocity
        dx = valid_frames[-1][0] - valid_frames[0][0]
        dy = valid_frames[-1][1] - valid_frames[0][1]
        dt = len(valid_frames)  # number of frames
        
        vx = dx / dt if dt > 0 else 0
        vy = dy / dt if dt > 0 else 0
        
        # Check for swipe
        motion = None
        confidence = 0.0
        
        # Horizontal swipe
        if abs(vx) > self.min_velocity and abs(dx) > self.swipe_threshold:
            if dx < 0:  # Swipe left (moving left in camera view = right on screen?)
                motion = 'swipe_left'
                confidence = min(abs(dx) / (self.swipe_threshold * 2), 1.0)
            else:  # Swipe right
                motion = 'swipe_right'
                confidence = min(abs(dx) / (self.swipe_threshold * 2), 1.0)
        # Vertical swipe
        elif abs(vy) > self.min_velocity and abs(dy) > self.swipe_threshold:
            if dy < 0:  # Swipe up (moving up in camera view)
                motion = 'swipe_up'
                confidence = min(abs(dy) / (self.swipe_threshold * 2), 1.0)
            else:  # Swipe down
                motion = 'swipe_down'
                confidence = min(abs(dy) / (self.swipe_threshold * 2), 1.0)
                
        return motion, confidence if motion else (None, 0.0)
    
    def _count_extended_fingers(self, landmarks):
        """Count number of extended fingers (0-5)"""
        if not landmarks:
            return 0
            
        wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
        tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        extended = 0
        
        for tip_idx in tips:
            tip = np.array([landmarks[tip_idx].x, landmarks[tip_idx].y, landmarks[tip_idx].z])
            # For thumb, compare to IP joint (3); for others, compare to PIP joint (tip_idx-2)
            pip_idx = 3 if tip_idx == 4 else tip_idx - 2
            pip = np.array([landmarks[pip_idx].x, landmarks[pip_idx].y, landmarks[pip_idx].z])
            
            # Finger is extended if tip is further from wrist than PIP joint
            if np.linalg.norm(tip - wrist) > np.linalg.norm(pip - wrist):
                extended += 1
                
        return extended
    
    def reset(self):
        """Clear detection buffers"""
        self.landmark_buffer.clear()
        self.three_finger_buffer.clear()
import numpy as np

def is_pinching(landmarks, threshold=0.05):
    """
    Determines if the hand is performing a 'pinch' gesture.
    A pinch is defined as the distance between the thumb tip (4) 
    and index fingertip (8) being below a certain threshold.
    """
    if not landmarks:
        return False
    
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
    index_tip = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
    
    distance = np.linalg.norm(thumb_tip - index_tip)
    return distance < threshold

def is_index_extended(landmarks):
    """
    Determines if the index finger is the most extended finger.
    Calculates distance from wrist (0) to fingertips (8, 12, 16, 20).
    """
    if not landmarks:
        return False
    
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    
    # Fingertips: Index(8), Middle(12), Ring(16), Pinky(20)
    tips = [8, 12, 16, 20]
    distances = []
    
    for tip_idx in tips:
        tip = np.array([landmarks[tip_idx].x, landmarks[tip_idx].y, landmarks[tip_idx].z])
        distances.append(np.linalg.norm(tip - wrist))
    
    # Index finger is distances[0]. 
    # Check if it is the maximum distance and significantly larger than the others.
    index_dist = distances[0]
    others = distances[1:]
    
    # Index must be the longest and at least 10% longer than the next longest finger
    return index_dist == max(distances) and index_dist > max(others) * 1.1

def is_thumb_active(landmarks):
    """
    Checks if the thumb is pointing towards the screen (horizontal/into Z).
    Separates 'active mode' from 'vertical' gestures like Thumb_Up/Down.
    """
    if not landmarks:
        return False
    
    # Thumb MCP (2) and Thumb Tip (4)
    mcp = np.array([landmarks[2].x, landmarks[2].y, landmarks[2].z])
    tip = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
    
    # Vector from MCP to Tip
    vector = tip - mcp
    
    # Check verticality: if Y component is dominant, it's likely Thumb_Up or Down
    y_mag = abs(vector[1])
    xz_mag = np.linalg.norm([vector[0], vector[2]])
    
    return xz_mag > y_mag * 0.5

def is_touching(landmarks, idx_a, idx_b, threshold=0.04):
    """
    Helper to check if two landmarks are touching.
    """
    if not landmarks:
        return False
    
    p1 = np.array([landmarks[idx_a].x, landmarks[idx_a].y, landmarks[idx_a].z])
    p2 = np.array([landmarks[idx_b].x, landmarks[idx_b].y, landmarks[idx_b].z])
    
    return np.linalg.norm(p1 - p2) < threshold


import inspect
GESTURE_RULES = {
    name: obj
    for name, obj in globals().items()
    if inspect.isfunction(obj) and not name.startswith('_')
}

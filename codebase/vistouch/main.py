import cv2
import time
import csv
import os
import numpy as np
from datetime import datetime
from modules.gesture_control.hand_tracker import HandTracker
from modules.gesture_control.gesture_rules import is_pinching, is_index_extended, is_thumb_active, is_touching
from modules.gesture_control.custom_gestures import CustomGestureManager
from modules.gesture_control.action_mapper import ActionMapper
from modules.gesture_control.motion_detector import MotionDetector

def main():
    # Initialize modules
    tracker = HandTracker()
    custom_manager = CustomGestureManager()
    mapper = ActionMapper()
    motion_detector = MotionDetector()
    
    cap = cv2.VideoCapture(0)
    
    # Configuration
    STABILITY_FRAMES = 5
    gesture_history = []
    prev_landmarks = None
    
    # Custom Enrollment State
    enroll_mode = False
    enroll_name = ""
    enroll_samples = []
    enroll_start_time = 0
    
    # Log setup
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'), exist_ok=True)
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'gesture_events.csv')
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'gesture', 'confidence'])

    print("VisTouch Gesture Control Started")
    print("Controls:")
    print("- 'q': Quit")
    print("- 'c': Enter Custom Gesture Enrollment Mode")
    print("- 'l': List enrolled gestures")
    print("- 'd': Delete a gesture")
    print("--------------------------------------------------")

    prev_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1) # Mirror for intuitive interaction
        h, w, _ = frame.shape
        curr_time_ms = int(time.time() * 1000)
        
        # 1. Hand Tracking
        gesture_label, confidence, landmarks = tracker.process_frame(frame, curr_time_ms)
        
        # 2. Refine Gesture (Reverting to Palm and Fist)
        final_gesture = gesture_label
        final_conf = confidence

        if landmarks:
            # Priority 1: Geometric Right-Click (Pinch)
            if is_pinching(landmarks):
                final_gesture = "Pinch"
                final_conf = 1.0
            # Priority 2: Motion gestures (swipes)
            else:
                motion_gesture, motion_conf = motion_detector.update(landmarks)
                if motion_gesture:
                    final_gesture = motion_gesture
                    final_conf = motion_conf
                else:
                    # Priority 3: Custom Gestures (static pose)
                    custom_label, custom_action, custom_conf = custom_manager.recognize(landmarks)
                    if custom_label and custom_conf > final_conf:
                        final_gesture = custom_label
                        final_conf = custom_conf

        # 3. Stability Layer
        gesture_history.append(final_gesture)
        if len(gesture_history) > STABILITY_FRAMES:
            gesture_history.pop(0)
        
        # Stable gesture is the one that has been consistent for N frames
        stable_gesture = None
        if len(gesture_history) == STABILITY_FRAMES and len(set(gesture_history)) == 1:
            stable_gesture = gesture_history[0]

        # 4. Execute Action
        if stable_gesture:
            # For motion gestures, we might want to execute immediately without stability?
            # But we'll keep stability for all gestures to avoid noise.
            # However, note: motion_detector.update already requires a sequence, so we can consider it stable.
            # We'll leave the stability layer as is.
            
            # Determine if we need to pass action config (for custom gestures with actions)
            if stable_gesture in custom_manager.gestures:
                # It's a registered custom gesture, get its action config
                action_config = custom_manager.get_gesture_action(stable_gesture)
                if action_config:
                    mapper.execute_action_from_config(action_config, landmarks, prev_landmarks)
                else:
                    # Fallback to default mapping (should not happen if we set action during enrollment)
                    mapper.execute(stable_gesture, landmarks, prev_landmarks)
            else:
                # Use the default mapping (for pinch, etc.)
                mapper.execute(stable_gesture, landmarks, prev_landmarks)
            
            # Log event
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().isoformat(), stable_gesture, final_conf])

        # 5. Handle Custom Enrollment / Management
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            for name, count, action in custom_manager.list_gestures():
                print(f"  {name}: {count} templates, action={action}")
        elif key == ord('d') and not enroll_mode:
            del_name = input("Name of gesture to delete: ").strip()
            if custom_manager.delete_gesture(del_name):
                print(f"Deleted '{del_name}'")
            else:
                print(f"'{del_name}' not found")
        elif key == ord('c') and not enroll_mode:
            enroll_mode = True
            enroll_name = input("Enter name for custom gesture: ")
            enroll_sigs = []
            enroll_start_time = time.time()
            print(f"Enrolling '{enroll_name}'... Hold pose for 3 seconds.")

        if enroll_mode:
            elapsed = time.time() - enroll_start_time
            cv2.putText(frame, f"ENROLLING {enroll_name}: {int(elapsed)/3:.0%}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if landmarks:
                sig = custom_manager._landmarks_to_signature(landmarks)
                enroll_sigs.append(sig)
                if len(enroll_sigs) > 1:
                    median_sig = np.median(np.array(enroll_sigs), axis=0)
                    dist = np.linalg.norm(sig - median_sig)
                    cv2.putText(frame, f"dist: {dist:.4f}", (10, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if elapsed >= 3.0:
                if len(enroll_sigs) >= 3:
                    median_sig = np.median(np.array(enroll_sigs), axis=0)
                    stable = [s for s in enroll_sigs if np.linalg.norm(s - median_sig) < 0.1]
                    if len(stable) >= 3:
                        if len(stable) > 10:
                            idxs = np.linspace(0, len(stable) - 1, 10, dtype=int)
                            stable = [stable[i] for i in idxs]
                        custom_manager.enroll_gesture(enroll_name, stable)
                        action_config = custom_manager.pick_action()
                        if action_config:
                            custom_manager.set_gesture_action(enroll_name, action_config)
                            print(f"Gesture '{enroll_name}' enrolled with {len(stable)} templates, action bound.")
                        else:
                            print(f"Gesture '{enroll_name}' enrolled with {len(stable)} templates, no action.")
                    else:
                        print(f"Pose too unstable. Got {len(stable)} stable frames, need 3. Try again.")
                else:
                    print(f"Not enough samples ({len(enroll_sigs)}). Need at least 3. Try again.")
                enroll_mode = False
                enroll_sigs = []

        # 6. Overlay
        fps = 1 / (time.time() - prev_time) if (time.time() - prev_time) > 0 else 0
        prev_time = time.time()
        
        cv2.putText(frame, f"Gesture: {final_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {final_conf:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw landmarks for visual aid
        if landmarks:
            for lm in landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (255, 0, 0), -1)

        cv2.imshow("VisTouch Gesture Control", frame)
        prev_landmarks = landmarks

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()

if __name__ == "__main__":
    main()

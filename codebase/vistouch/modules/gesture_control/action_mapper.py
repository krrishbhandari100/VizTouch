import pyautogui
import numpy as np
import os
import subprocess
import json

class ActionMapper:
    """
    Maps recognized gestures to OS actions using PyAutoGUI.
    """
    def __init__(self, config_path=None):
        import json
        if config_path is None:
            # Find config path relative to this file: ../../config/gesture_actions.json
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(base_dir, 'config', 'gesture_actions.json')
            
        with open(config_path, 'r') as f:
            self.mapping = json.load(f)
        
        # Mapping configuration: Define a margin to make it easier to reach screen edges
        # 0.1 means the inner 80% of the camera view maps to 100% of the screen
        self.MARGIN = 0.1 
        
        # PyAutoGUI safety
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0

    def _map_coordinate(self, value, screen_dim):
        """
        Maps a normalized value (0-1) with a margin to the full screen dimension.
        """
        # Clip the value to be within the margin boundaries
        clipped = max(self.MARGIN, min(1.0 - self.MARGIN, value))
        # Normalize the clipped value to 0-1 range
        normalized = (clipped - self.MARGIN) / (1.0 - 2 * self.MARGIN)
        return int(normalized * screen_dim)

    def execute(self, gesture, landmarks=None, prev_pos=None):
        """
        Executes the action associated with the gesture.
        """
        action = self.mapping.get(gesture)
        if not action:
            return

        if action == "left_click":
            pyautogui.click()
        elif action == "right_click":
            pyautogui.rightClick()
        elif action == "volume_up":
            pyautogui.press("volumeup")
        elif action == "volume_down":
            pyautogui.press("volumedown")
        elif action == "move_cursor":
            if landmarks:
                # Index fingertip is landmark 8
                index_tip = landmarks[8]
                width, height = pyautogui.size()
                
                # Use the map_coordinate helper to amplify movement
                screen_x = self._map_coordinate(index_tip.x, width)
                screen_y = self._map_coordinate(index_tip.y, height)
                pyautogui.moveTo(screen_x, screen_y)
        elif action == "scroll_mode":
            if landmarks and prev_pos:
                curr_y = landmarks[8].y
                prev_y = prev_pos[8].y
                diff = curr_y - prev_y
                if abs(diff) > 0.01:
                    scroll_amount = -10 if diff > 0 else 10
                    pyautogui.scroll(scroll_amount)

    def execute_action_from_config(self, action_config, landmarks=None, prev_pos=None):
        """
        Execute an action from a configuration dictionary.
        action_config: dict with 'type' and optional parameters
        """
        if not action_config or not isinstance(action_config, dict):
            return
        
        action_type = action_config.get('type')
        
        if action_type == "left_click":
            pyautogui.click()
        elif action_type == "right_click":
            pyautogui.rightClick()
        elif action_type == "volume_up":
            pyautogui.press("volumeup")
        elif action_type == "volume_down":
            pyautogui.press("volumedown")
        elif action_type == "move_cursor":
            if landmarks:
                # Index fingertip is landmark 8
                index_tip = landmarks[8]
                width, height = pyautogui.size()
                
                # Use the map_coordinate helper to amplify movement
                screen_x = self._map_coordinate(index_tip.x, width)
                screen_y = self._map_coordinate(index_tip.y, height)
                pyautogui.moveTo(screen_x, screen_y)
        elif action_type == "scroll_mode":
            if landmarks and prev_pos:
                curr_y = landmarks[8].y
                prev_y = prev_pos[8].y
                diff = curr_y - prev_y
                if abs(diff) > 0.01:
                    scroll_amount = -10 if diff > 0 else 10
                    pyautogui.scroll(scroll_amount)
        elif action_type == "desktop_switch":
            direction = action_config.get('direction', 'left')
            if direction == "left":
                pyautogui.hotkey('ctrl', 'win', 'left')  # Windows: previous desktop
                # For Mac: pyautogui.hotkey('ctrl', 'left')
                # For Linux: pyautogui.hotkey('ctrl', 'alt', 'left')
            else:  # right
                pyautogui.hotkey('ctrl', 'win', 'right')  # Windows: next desktop
                # For Mac: pyautogui.hotkey('ctrl', 'right')
                # For Linux: pyautogui.hotkey('ctrl', 'alt', 'right')
        elif action_type == "launch_app":
            app = action_config.get('app', '').strip()
            if app:
                try:
                    subprocess.Popen(app, shell=True)  # Windows
                    # For Mac: subprocess.Popen(['open', '-a', app])
                    # For Linux: subprocess.Popen(app, shell=True)
                except Exception as e:
                    print(f"Failed to launch {app}: {e}")
        elif action_type == "hotkey":
            keys = action_config.get('keys', [])
            if keys:
                pyautogui.hotkey(*keys)
        elif action_type == "scroll":
            amount = action_config.get('amount', 3)
            pyautogui.scroll(amount)
        # Add more action types as needed


import numpy as np
import json
import os


class CustomGestureManager:
    def __init__(self, data_path=None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_path = os.path.join(base_dir, 'data', 'custom_gestures.json')
        self.data_path = data_path
        self.gestures = self._load_gestures()
        self.actions = self._load_actions()

    def _load_gestures(self):
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            migrated = {}
            for name, samples in data.items():
                migrated[name] = []
                for sample in samples:
                    if isinstance(sample, list) and len(sample) == 63 and isinstance(sample[0], (int, float)):
                        migrated[name].append(self._flat_to_signature(sample))
                    elif isinstance(sample, list) and len(sample) == 210:
                        migrated[name].append(sample)
            return migrated
        return {}

    def _save_gestures(self):
        with open(self.data_path, 'w') as f:
            json.dump(self.gestures, f)

    def _load_actions(self):
        actions_path = self.data_path.replace('.json', '_actions.json')
        if os.path.exists(actions_path):
            with open(actions_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_actions(self):
        actions_path = self.data_path.replace('.json', '_actions.json')
        with open(actions_path, 'w') as f:
            json.dump(self.actions, f, indent=2)

    @staticmethod
    def _flat_to_signature(flat):
        pts = np.array(flat).reshape(21, 3)
        centered = pts - pts[0]
        hand_size = np.linalg.norm(centered[9])
        if hand_size < 1e-6:
            hand_size = 1.0
        scaled = centered / hand_size
        dists = []
        for i in range(21):
            for j in range(i + 1, 21):
                dists.append(np.linalg.norm(scaled[i] - scaled[j]))
        return np.array(dists)

    def _landmarks_to_signature(self, landmarks):
        if not landmarks or len(landmarks) != 21:
            return np.zeros(210)
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        return self._flat_to_signature(pts.flatten().tolist())

    def _to_signature(self, landmarks_or_flat):
        if isinstance(landmarks_or_flat, list) and len(landmarks_or_flat) == 63 and isinstance(landmarks_or_flat[0], (int, float)):
            return self._flat_to_signature(landmarks_or_flat)
        return self._landmarks_to_signature(landmarks_or_flat)

    def enroll_gesture(self, name, samples):
        if isinstance(samples[0] if samples else [], (int, float)):
            samples = [self._to_signature(s) for s in samples]
        self.gestures[name] = [s.tolist() for s in samples]
        self._save_gestures()

    def recognize(self, current_landmarks, k=5, threshold=0.55):
        if not self.gestures or not current_landmarks:
            return None, None, 0.0
        sig = self._to_signature(current_landmarks)
        results = []
        for name, samples in self.gestures.items():
            distances = sorted([np.linalg.norm(sig - np.array(s)) for s in samples])
            k_dists = distances[:k]
            votes = sum(1 for d in k_dists if d < threshold)
            avg_dist = sum(k_dists) / len(k_dists) if k_dists else float('inf')
            results.append((name, avg_dist, votes))
        results.sort(key=lambda x: (-x[2], x[1]))
        best_name, best_dist, best_votes = results[0]
        if best_votes == 0:
            return None, None, 0.0
        confidence = best_votes / k
        action_config = self.get_gesture_action(best_name)
        return best_name, action_config, confidence

    def set_gesture_action(self, name, action_config):
        self.actions[name] = action_config
        self._save_actions()

    def get_gesture_action(self, name):
        return self.actions.get(name)

    def list_gestures(self):
        return [(name, len(samples), self.get_gesture_action(name)) for name, samples in self.gestures.items()]

    def delete_gesture(self, name):
        deleted = False
        if name in self.gestures:
            del self.gestures[name]
            self._save_gestures()
            deleted = True
        if name in self.actions:
            del self.actions[name]
            self._save_actions()
        return deleted

    def pick_action(self):
        print("\nSelect action type:")
        print("  1. move_cursor")
        print("  2. left_click")
        print("  3. right_click")
        print("  4. scroll_mode")
        print("  5. volume_up")
        print("  6. volume_down")
        print("  7. scroll (param: amount)")
        print("  8. hotkey (param: keys e.g. ctrl+c)")
        print("  9. launch_app (param: app path)")
        print("  10. desktop_switch (param: direction)")
        print("  0. None (no action)")
        choice = input("Choice: ").strip()
        if choice == '1':
            return {'type': 'move_cursor'}
        elif choice == '2':
            return {'type': 'left_click'}
        elif choice == '3':
            return {'type': 'right_click'}
        elif choice == '4':
            return {'type': 'scroll_mode'}
        elif choice == '5':
            return {'type': 'volume_up'}
        elif choice == '6':
            return {'type': 'volume_down'}
        elif choice == '7':
            amount = input("Scroll amount (default 3): ").strip()
            return {'type': 'scroll', 'amount': int(amount) if amount.isdigit() else 3}
        elif choice == '8':
            keys = input("Keys e.g. ctrl+c: ").strip()
            return {'type': 'hotkey', 'keys': keys.split('+')}
        elif choice == '9':
            app = input("App path: ").strip()
            return {'type': 'launch_app', 'app': app}
        elif choice == '10':
            direction = input("Direction (left/right, default left): ").strip()
            return {'type': 'desktop_switch', 'direction': direction or 'left'}
        return None

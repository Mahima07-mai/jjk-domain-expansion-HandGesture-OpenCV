import cv2
import mediapipe as mp
import numpy as np
from typing import List, NamedTuple, Optional

import mediapipe as mp
# We access the solutions through the 'solutions' attribute that 
# exists inside the compiled mediapipe module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandData(NamedTuple):
    landmarks: np.ndarray        # (21, 2) pixel coords
    norm_landmarks: np.ndarray   # (21, 3) normalized coords
    handedness: str
    palm_facing_camera: bool
    stability_score: float

class HandTracker:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.history = {} # Stores last 10 frames of landmarks for stability

    def _is_palm_facing_camera(self, landmarks, handedness):
        # Calculate cross product of MCP 0, 5, 17 to find palm normal
        # pts are [x, y, z]
        p0 = landmarks[0]
        p5 = landmarks[5]
        p17 = landmarks[17]
        
        v1 = p5 - p0
        v2 = p17 - p0
        normal = np.cross(v1, v2)
        # In MediaPipe, Z is towards the camera (negative)
        # A rough heuristic: if normal.z is negative, palm faces camera for Right hand
        # This varies by hand orientation, but for Domain Expansion gestures (facing camera)
        # we check the z-component of the normal vector.
        return normal[2] < 0 if handedness == "Left" else normal[2] > 0

    def update(self, frame) -> List[HandData]:
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_detector.process(rgb_frame)
        
        hands_data = []
        if results.multi_hand_landmarks:
            for i, (lms, hand_info) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                label = hand_info.classification[0].label # "Left" / "Right"
                
                # Extract landmarks
                raw_pts = []
                norm_pts = []
                for lm in lms.landmark:
                    raw_pts.append([lm.x * w, lm.y * h])
                    norm_pts.append([lm.x, lm.y, lm.z])
                
                raw_pts = np.array(raw_pts)
                norm_pts = np.array(norm_pts)
                
                # Stability
                if i not in self.history:
                    self.history[i] = []
                self.history[i].append(raw_pts)
                if len(self.history[i]) > 10:
                    self.history[i].pop(0)
                
                stability = 1.0
                if len(self.history[i]) > 1:
                    movements = [np.linalg.norm(self.history[i][j] - self.history[i][j-1]) for j in range(1, len(self.history[i]))]
                    stability = 1.0 / (1.0 + np.mean(movements)) # Lower movement = higher stability
                
                hands_data.append(HandData(
                    landmarks=raw_pts,
                    norm_landmarks=norm_pts,
                    handedness=label,
                    palm_facing_camera=self._is_palm_facing_camera(norm_pts, label),
                    stability_score=stability
                ))
        return hands_data

    def draw_landmarks(self, frame, hands: List[HandData]):
        # Custom skeleton overlay with color based on handedness
        for hand in hands:
            color = (0, 255, 0) if hand.handedness == "Right" else (255, 0, 0)
            for i in range(21):
                x, y = int(hand.landmarks[i, 0]), int(hand.landmarks[i, 1])
                cv2.circle(frame, (x, y), 5, color, -1)
            
            # Simplified connection lines (optional - can use mp_draw for full)
            # self.mp_draw.draw_landmarks(...) # Standard way

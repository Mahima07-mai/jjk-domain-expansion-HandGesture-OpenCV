import numpy as np
import time
from typing import List, Optional, NamedTuple
from hand_tracking import HandData

class GestureResult(NamedTuple):
    gesture: str
    confidence: float
    stable_since: float
    triggered: bool

def finger_angle(landmarks, finger_base, finger_mid, finger_tip):
    v1 = landmarks[finger_mid] - landmarks[finger_base]
    v2 = landmarks[finger_tip] - landmarks[finger_mid]
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def fingertip_distance(landmarks_a, tip_a, landmarks_b, tip_b):
    return np.linalg.norm(landmarks_a[tip_a] - landmarks_b[tip_b])

def hand_symmetry_score(left_hand, right_hand):
    if not left_hand or not right_hand:
        return 0.0
    return 1.0 - np.clip(
        np.mean(np.abs(left_hand.norm_landmarks[:, 0] + right_hand.norm_landmarks[:, 0] - 1.0)),
        0, 1
    )

def is_finger_extended(landmarks, finger_indices):
    angle = finger_angle(landmarks, finger_indices[0], finger_indices[1], finger_indices[2])
    return angle < 25.0

class GestureRecognizer:
    def __init__(self):
        self.gesture_history   = {"gojo": [], "sukuna": [], "hollow_purple": []}
        self.stability_timers  = {"gojo": None, "sukuna": None, "hollow_purple": None}
        self.cooldown_end      = 0.0

        self.fingers = {
            "thumb":  [1,  2,  4],
            "index":  [5,  6,  8],
            "middle": [9,  10, 12],
            "ring":   [13, 14, 16],
            "pinky":  [17, 18, 20],
        }

    # ── GOJO: two fingers (index+middle) pinched together, ring+pinky curled ──
    def _check_gojo(self, left_hand, right_hand):
        target_hand = right_hand if right_hand else left_hand
        if not target_hand:
            return 0.0

        lms = target_hand.landmarks
        dist_tips  = np.linalg.norm(lms[8] - lms[12])
        dist_bases = np.linalg.norm(lms[5] - lms[9])

        is_index_ext  = is_finger_extended(lms, self.fingers["index"])
        is_middle_ext = is_finger_extended(lms, self.fingers["middle"])
        is_ring_curl  = not is_finger_extended(lms, self.fingers["ring"])
        is_pinky_curl = not is_finger_extended(lms, self.fingers["pinky"])

        if is_index_ext and is_middle_ext and dist_tips < dist_bases * 0.85:
            if is_ring_curl and is_pinky_curl:
                return 0.95
            return 0.70
        return 0.0

    # ── SUKUNA: both palms pressed together, all fingertips close ────────────
    def _check_sukuna(self, left_hand, right_hand):
        if not left_hand or not right_hand:
            return 0.0

        sym = hand_symmetry_score(left_hand, right_hand)
        if sym < 0.75:
            return 0.0

        d_thumb = fingertip_distance(left_hand.landmarks, 4,  right_hand.landmarks, 4)
        d_index = fingertip_distance(left_hand.landmarks, 8,  right_hand.landmarks, 8)
        d_pinky = fingertip_distance(left_hand.landmarks, 20, right_hand.landmarks, 20)

        palm_dist = np.linalg.norm(left_hand.landmarks[0] - right_hand.landmarks[0])
        if palm_dist == 0:
            return 0.0

        r_thumb = d_thumb / palm_dist
        r_index = d_index / palm_dist
        r_pinky = d_pinky / palm_dist

        if r_thumb < 0.25 and r_index < 0.25 and r_pinky < 0.25:
            return float(0.5 + 0.5 * sym)
        return 0.0

    # ── HOLLOW PURPLE: one hand, all 5 fingers extended & fanned/spread open ─
    # The sign (from the image) is a relaxed open hand, fingers spread apart,
    # palm facing camera. Key cues:
    #   1. All 4 fingers extended (low bend angle)
    #   2. Fingers spread wide (large gaps between adjacent fingertips)
    #   3. Thumb also extended outward
    #   4. Palm facing camera (palm_facing_camera == True)
    def _check_hollow_purple(self, left_hand, right_hand):
        # Single hand only — prefer right
        target_hand = right_hand if right_hand else left_hand
        if not target_hand:
            return 0.0

        # Reject if both hands present (avoid conflict with sukuna)
        if left_hand and right_hand:
            return 0.0

        lms = target_hand.landmarks

        # 1. Index and middle fingers must be EXTENDED
        is_index_ext  = is_finger_extended(lms, self.fingers["index"])
        is_middle_ext = is_finger_extended(lms, self.fingers["middle"])
        if not (is_index_ext and is_middle_ext):
            return 0.0

        # 2. Ring and pinky must be CURLED inward
        is_ring_curl  = not is_finger_extended(lms, self.fingers["ring"])
        is_pinky_curl = not is_finger_extended(lms, self.fingers["pinky"])
        if not (is_ring_curl and is_pinky_curl):
            return 0.0

        # 3. Index and middle must be SEPARATED (V shape) — NOT pinched like Gojo
        #    Gojo: tips close together. Hollow Purple: tips spread apart.
        dist_tips  = np.linalg.norm(lms[8] - lms[12])   # index tip to middle tip
        dist_bases = np.linalg.norm(lms[5] - lms[9])    # index MCP to middle MCP
        # V shape = tips further apart than bases (spread ratio > 1.1)
        spread_ratio = dist_tips / (dist_bases + 1e-6)
        if spread_ratio < 1.05:   # tips must be MORE spread than bases
            return 0.0

        # 4. Thumb pressing down on curled fingers
        #    Thumb tip (4) should be CLOSE to ring/pinky region (landmarks 14-16 area)
        #    Use distance from thumb tip to ring middle joint as proxy
        d_thumb_ring = np.linalg.norm(lms[4] - lms[14])
        palm_size    = np.linalg.norm(lms[0] - lms[9])   # wrist to middle MCP
        if palm_size < 1e-3:
            return 0.0
        thumb_press_ratio = d_thumb_ring / palm_size
        if thumb_press_ratio > 0.55:   # thumb too far from curled fingers
            return 0.0

        # 5. Confidence: scales with spread of V and closeness of thumb press
        v_score     = float(np.clip((spread_ratio - 1.05) / 0.5, 0.0, 1.0))
        thumb_score = float(np.clip(1.0 - (thumb_press_ratio / 0.55), 0.0, 1.0))
        conf = 0.70 + 0.15 * v_score + 0.15 * thumb_score
        return float(np.clip(conf, 0.70, 0.98))

    # ─────────────────────────────────────────────────────────────────────────
    def update(self, hands: List[HandData], frame_shape) -> Optional[GestureResult]:
        now = time.time()

        if now < self.cooldown_end:
            return None

        results = {"gojo": 0.0, "sukuna": 0.0, "hollow_purple": 0.0}

        left_hand  = next((h for h in hands if h.handedness == "Left"),  None)
        right_hand = next((h for h in hands if h.handedness == "Right"), None)

        if len(hands) >= 1:
            results["gojo"]         = self._check_gojo(left_hand, right_hand)
            results["hollow_purple"]= self._check_hollow_purple(left_hand, right_hand)
        if len(hands) == 2:
            results["sukuna"]       = self._check_sukuna(left_hand, right_hand)

        best_gesture = None
        best_conf    = 0.5

        for g, conf in results.items():
            self.gesture_history[g].append(conf)
            if len(self.gesture_history[g]) > 10:
                self.gesture_history[g].pop(0)
            avg_conf = float(np.mean(self.gesture_history[g]))

            if avg_conf >= 0.80:
                if self.stability_timers[g] is None:
                    self.stability_timers[g] = now
                if now - self.stability_timers[g] >= 1.0:
                    self.cooldown_end = now + 2.5
                    temp_timer = self.stability_timers[g]
                    self.stability_timers[g] = None
                    return GestureResult(g, avg_conf, temp_timer, True)
            else:
                if avg_conf < 0.75:
                    self.stability_timers[g] = None

            if avg_conf > best_conf:
                best_conf    = avg_conf
                best_gesture = g

        if best_gesture:
            stable = self.stability_timers[best_gesture] or 0.0
            return GestureResult(best_gesture, best_conf, stable, False)

        return None
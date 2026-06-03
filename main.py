import cv2
import numpy as np
import time
import traceback
from PIL import Image, ImageDraw, ImageFont
from hand_tracking import HandTracker
from gesture_recognition import GestureRecognizer
from effects.gojo import UnlimitedVoid
from effects.sukuna import MalevolentShrine
from effects.hollow_purple import HollowPurple, generate_space_bg

# ── JJK TITLE TEXT DATA PROFILES ──────────────────────────────────────────────
JJK_TEXT_PROFILES = {
    "sukuna": {
        "jp": "伏魔御厨子",
        "en": "Domain Expansion : Malevolent Shrine"
    },
    "gojo": {
        "jp": "無量空処",
        "en": "Domain Expansion : Unlimited Void"
    },
    "hollow_purple": {
        "jp": "虚式「茈」",
        "en": "Hollow Technique : Purple"
    }
}

def draw_jjk_ui_titles(frame, japanese_text, english_text):
    """ Renders centered Jujutsu Kaisen cinematic titles scaled to current viewport dimensions """
    h, w = frame.shape[:2]
    
    # LINE 1: JAPANESE TEXT (PIL RENDER PASS)
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    font_size_jp = int(h * 0.065)
    try:
        font_jp = ImageFont.truetype("msgothic.ttc", font_size_jp)
    except IOError:
        try:
            font_jp = ImageFont.truetype("msmincho.ttc", font_size_jp)
        except IOError:
            try:
                font_jp = ImageFont.truetype("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", font_size_jp)
            except IOError:
                font_jp = ImageFont.load_default()

    jp_bbox = draw.textbbox((0, 0), japanese_text, font=font_jp)
    jp_w = jp_bbox[2] - jp_bbox[0]
    jp_x = (w - jp_w) // 2
    jp_y = int(h * 0.08)
    
    draw.text((jp_x, jp_y), japanese_text, font=font_jp, fill=(165, 169, 172))
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # LINE 2: ENGLISH SUBTITLE (OPENCV COMPONENT)
    font_en = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = (h / 1080) * 1.05
    thickness = max(2, int(font_scale * 2))
    
    en_size = cv2.getTextSize(english_text, font_en, font_scale, thickness)[0]
    en_x = (w - en_size[0]) // 2
    en_y = jp_y + int(h * 0.09)
    
    cv2.putText(frame, english_text, (en_x + 2, en_y + 2), font_en, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, english_text, (en_x, en_y), font_en, font_scale, (220, 220, 0), thickness, cv2.LINE_AA)
    
    return frame

def draw_hud(frame, result, cooldown_end):
    h, w = frame.shape[:2]
    now = time.time()

    GESTURE_COLORS = {
        "gojo":         (255, 200,  50),
        "sukuna":       (0,    50, 255),
        "hollow_purple":(255,   0, 255),
    }

    if result:
        col  = GESTURE_COLORS.get(result.gesture, (0, 255, 0))
        label = result.gesture.replace("_", " ").upper()
        text = f"{label}: {int(result.confidence * 100)}%"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
        cv2.rectangle(frame, (20, 50), (int(20 + 200 * result.confidence), 65), col, -1)
        if result.stable_since and result.stable_since > 0:
            elapsed = now - result.stable_since
            angle   = int(360 * min(1.0, elapsed / 1.0))
            cv2.ellipse(frame, (250, 35), (15, 15), 0, 0, angle, col, 2)

    if now < cooldown_end:
        rem = cooldown_end - now
        cv2.putText(frame, f"COOLDOWN: {rem:.1f}s", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def draw_landmarks_scaled(img, hands, sx, sy):
    for h_data in hands:
        color = (0, 255, 0) if h_data.handedness == "Right" else (0, 128, 255)
        for i in range(21):
            px = int(h_data.landmarks[i, 0] * sx)
            py = int(h_data.landmarks[i, 1] * sy)
            cv2.circle(img, (px, py), 3, color, -1)

def main():
    window_name = "JJK Domain Expansion Studio"
    
    # 🌟 CRITICAL: Allow window to be completely resizable and check screen changes
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(0)
    ret, test_frame = cap.read()
    if not ret:
        print("CRITICAL: Failed to access visual webcam tracking stream.")
        return

    # Baseline Webcam Dimensions
    cam_h, cam_w = test_frame.shape[:2]

    tracker = HandTracker()
    recognizer = GestureRecognizer()

    print("[SYSTEM ENGINE] Pre-generating deep nebula galaxy layers...")
    space_bg = generate_space_bg(cam_w, cam_h)

    effects = {
        "gojo":          UnlimitedVoid((cam_h, cam_w)),
        "sukuna":        MalevolentShrine((cam_h, cam_w)),
        "hollow_purple": HollowPurple((cam_h, cam_w, 3), bg_image=space_bg),
    }

    needs_camera = {"gojo"}
    active_effect = None
    last_time = time.time()

    print("\n>>> JJK DOMAIN EXPANSION STUDIO ACTIVE <<<")
    print("Press 'q' to quit. Go full screen safely without cropping issues now!\n")

    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        frame = cv2.flip(frame, 1)
        now = time.time()
        dt = min(now - last_time, 0.1)
        last_time = now

        # Get actual target windows dimensions (dynamically checks if you maximized full screen)
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
            if win_w <= 0 or win_h <= 0: # Fallback to cam size if window properties aren't loaded yet
                win_w, win_h = cam_w, cam_h
        except:
            win_w, win_h = cam_w, cam_h

        # Compute dynamic ratios
        PIP_W, PIP_H = int(win_w * 0.23), int(win_h * 0.23)
        scale_x, scale_y = PIP_W / cam_w, PIP_H / cam_h

        hands = tracker.update(frame)
        result = recognizer.update(hands, (cam_h, cam_w))

        if result and result.triggered:
            g = result.gesture.lower()
            if g in effects and (active_effect is None or not active_effect.active):
                active_effect = effects[g]
                active_effect.activate()

        if active_effect and active_effect.active:
            effect_key = next((k for k, v in effects.items() if v is active_effect), None)
            effect_input = frame if effect_key in needs_camera else np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            
            try:
                out = active_effect.update(effect_input, dt)
                base_frame = out if (out is not None and out.shape[:2] == (cam_h, cam_w)) else frame.copy()
                
                # 🌟 Fix: Resize the base visual canvas to match your expanded window size cleanly BEFORE text placement
                display_frame = cv2.resize(base_frame, (win_w, win_h))
                
                if effect_key in JJK_TEXT_PROFILES:
                    display_frame = draw_jjk_ui_titles(
                        display_frame, 
                        JJK_TEXT_PROFILES[effect_key]["jp"], 
                        JJK_TEXT_PROFILES[effect_key]["en"]
                    )
                    
            except Exception as e:
                print(f"ERROR inside active graphics frame pass pipeline: {e}")
                traceback.print_exc()
                active_effect.active = False
                active_effect = None
                display_frame = cv2.resize(frame, (win_w, win_h))
        else:
            active_effect = None
            # Scale raw frame up to meet full-screen monitor sizing
            display_frame = cv2.resize(frame, (win_w, win_h))
            
            # Draw tracking assets relative to new window size coordinates
            scaled_hands = []
            for h_data in hands:
                scaled_lms = h_data.landmarks.copy()
                scaled_lms[:, 0] *= (win_w / cam_w)
                scaled_lms[:, 1] *= (win_h / cam_h)
                # Quick draw circle overlay
                color = (0, 255, 0) if h_data.handedness == "Right" else (255, 0, 0)
                for pt in scaled_lms:
                    cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 5, color, -1)

            draw_hud(display_frame, result, recognizer.cooldown_end)

        # ── RENDER DYNAMIC PIP CAM STREAM FEED ────────────────────────────────
        camera_pip = cv2.resize(frame, (PIP_W, PIP_H))
        draw_landmarks_scaled(camera_pip, hands, scale_x, scale_y)
        y_start = win_h - PIP_H
        
        # Safely blend PIP view into scaled main frame matrix
        display_frame[y_start:win_h, 0:PIP_W] = camera_pip
        cv2.rectangle(display_frame, (0, y_start), (PIP_W - 1, win_h - 1), (0, 255, 255), 2)

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("g"):
            active_effect = effects["gojo"]
            active_effect.activate()
        elif key == ord("s"):
            active_effect = effects["sukuna"]
            active_effect.activate()
        elif key == ord("h"):
            active_effect = effects["hollow_purple"]
            active_effect.activate()
        elif key == ord("r"):
            if active_effect:
                active_effect.active = False
                active_effect = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
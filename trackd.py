"""
Trackd - Gesture-Controlled Daft Punk Instrument
=================================================
Turn thumb-pinch gestures into an interactive remix of
"Harder, Better, Faster, Stronger" using MediaPipe hand tracking.

Controls:
  q - quit
  +/- - adjust pinch threshold
"""

import os
import sys
import time
import math

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    RunningMode,
)
import pygame


# ---------------------------------------------
# Configuration
# ---------------------------------------------
PINCH_THRESHOLD = 50
COOLDOWN_FRAMES = 5
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
HELMET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "helmet.png")
WINDOW_NAME = "Trackd"

THUMB_TIP = 4
FINGER_TIPS = {
    "index":  8,
    "middle": 12,
    "ring":   16,
    "pinky":  20,
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

GESTURE_MAP = {
    ("Left",  "index"):  ("work it",   "work_it.wav"),
    ("Left",  "middle"): ("harder",    "harder.wav"),
    ("Left",  "ring"):   ("make it",   "make_it.wav"),
    ("Left",  "pinky"):  ("better",    "better.wav"),
    ("Right", "index"):  ("do it",     "do_it.wav"),
    ("Right", "middle"): ("faster",    "faster.wav"),
    ("Right", "ring"):   ("makes us",  "makes_us.wav"),
    ("Right", "pinky"):  ("stronger",  "stronger.wav"),
}

LYRIC_SEQUENCE = [
    "work it", "harder", "make it", "better",
    "do it", "faster", "makes us", "stronger",
]

COLOR_CYAN         = (255, 255, 0)
COLOR_MAGENTA      = (255, 0, 255)
COLOR_GOLD         = (0, 215, 255)
COLOR_WHITE        = (255, 255, 255)
COLOR_DIM          = (100, 100, 100)
COLOR_GREEN        = (0, 255, 100)
COLOR_GLOW_CYAN    = (255, 230, 0)
COLOR_GLOW_MAGENTA = (255, 50, 255)


class GestureState:
    IDLE     = 0
    ACTIVE   = 1
    COOLDOWN = 2

    def __init__(self):
        self.state = self.IDLE
        self.cooldown_counter = 0

    def update(self, is_pinching):
        triggered = False
        if self.state == self.IDLE:
            if is_pinching:
                self.state = self.ACTIVE
                triggered = True
        elif self.state == self.ACTIVE:
            if not is_pinching:
                self.state = self.COOLDOWN
                self.cooldown_counter = COOLDOWN_FRAMES
        elif self.state == self.COOLDOWN:
            self.cooldown_counter -= 1
            if self.cooldown_counter <= 0:
                self.state = self.IDLE
        return triggered


class AudioEngine:
    def __init__(self):
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(16)
        self.sounds = {}
        self.channels = {}
        channel_idx = 0
        for key, (label, filename) in GESTURE_MAP.items():
            filepath = os.path.join(AUDIO_DIR, filename)
            if os.path.exists(filepath):
                self.sounds[key] = pygame.mixer.Sound(filepath)
                self.channels[key] = pygame.mixer.Channel(channel_idx)
                channel_idx += 1
            else:
                print(f"[WARNING] Audio file not found: {filepath}")

    def play(self, gesture_key):
        if gesture_key in self.sounds:
            self.channels[gesture_key].play(self.sounds[gesture_key])

    def cleanup(self):
        pygame.mixer.quit()


class HUD:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.last_triggered = None
        self.trigger_time = 0

    def draw_hand_skeleton(self, frame, landmarks, handedness):
        h, w, _ = frame.shape
        color = COLOR_CYAN if handedness == "Left" else COLOR_MAGENTA
        glow  = COLOR_GLOW_CYAN if handedness == "Left" else COLOR_GLOW_MAGENTA
        for conn in HAND_CONNECTIONS:
            p1 = landmarks[conn[0]]
            p2 = landmarks[conn[1]]
            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), glow, 4, cv2.LINE_AA)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, COLOR_WHITE, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 6, color, 1, cv2.LINE_AA)

    def draw_pinch_indicator(self, frame, pos, word, is_active):
        if not is_active:
            return
        x, y = pos
        now = time.time()
        pulse = int(20 * math.sin(now * 8) + 35)
        radius = 25 + pulse // 3
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), radius + 15, COLOR_GOLD, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        overlay2 = frame.copy()
        cv2.circle(overlay2, (x, y), radius, COLOR_GOLD, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay2, 0.3, frame, 0.7, 0, frame)
        cv2.circle(frame, (x, y), 8, COLOR_WHITE, -1, cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(word.upper(), font, 0.8, 2)[0]
        tx = x - text_size[0] // 2
        ty = y - radius - 15
        padding = 10
        cv2.rectangle(frame, (tx - padding, ty - text_size[1] - padding),
                      (tx + text_size[0] + padding, ty + padding), (0, 0, 0), -1)
        cv2.rectangle(frame, (tx - padding, ty - text_size[1] - padding),
                      (tx + text_size[0] + padding, ty + padding), COLOR_GOLD, 2)
        cv2.putText(frame, word.upper(), (tx, ty), font, 0.8, COLOR_GOLD, 2, cv2.LINE_AA)

    def draw_lyric_bar(self, frame):
        bar_height = 60
        y_start = self.h - bar_height
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_start), (self.w, self.h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, y_start), (self.w, y_start), COLOR_GOLD, 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        spacing = 30
        word_widths = []
        for word in LYRIC_SEQUENCE:
            sz = cv2.getTextSize(word.upper(), font, font_scale, 2)[0]
            word_widths.append(sz[0])
        total_width = sum(word_widths) + spacing * (len(LYRIC_SEQUENCE) - 1)
        start_x = (self.w - total_width) // 2
        text_y = y_start + 38
        now = time.time()
        glow_active = (now - self.trigger_time) < 1.0
        cursor_x = start_x
        for i, word in enumerate(LYRIC_SEQUENCE):
            is_current = (word == self.last_triggered and glow_active)
            if is_current:
                color = COLOR_GOLD
                thickness = 2
                overlay3 = frame.copy()
                pad = 8
                cv2.rectangle(overlay3, (cursor_x - pad, text_y - 20 - pad),
                              (cursor_x + word_widths[i] + pad, text_y + pad), COLOR_GOLD, -1)
                cv2.addWeighted(overlay3, 0.15, frame, 0.85, 0, frame)
            else:
                color = COLOR_DIM
                thickness = 1
            cv2.putText(frame, word.upper(), (cursor_x, text_y),
                        font, font_scale, color, thickness, cv2.LINE_AA)
            cursor_x += word_widths[i] + spacing

    def draw_header(self, frame, fps, threshold):
        header_height = 45
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, header_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.line(frame, (0, header_height), (self.w, header_height), COLOR_GOLD, 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "TRACKD", (15, 30), font, 0.7, COLOR_GOLD, 2, cv2.LINE_AA)
        cv2.putText(frame, "GESTURE INSTRUMENT", (115, 30), font, 0.4, COLOR_DIM, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {int(fps)}", (self.w - 120, 30), font, 0.5, COLOR_GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, f"THR: {threshold}px", (self.w - 250, 30), font, 0.5, COLOR_DIM, 1, cv2.LINE_AA)

    def draw_gesture_guide(self, frame, side, x_anchor):
        font = cv2.FONT_HERSHEY_SIMPLEX
        if side == "Left":
            gestures = [("IDX", "work it"), ("MID", "harder"), ("RNG", "make it"), ("PNK", "better")]
            color = COLOR_CYAN
        else:
            gestures = [("IDX", "do it"), ("MID", "faster"), ("RNG", "makes us"), ("PNK", "stronger")]
            color = COLOR_MAGENTA
        y_start = 70
        for i, (finger, word) in enumerate(gestures):
            cv2.putText(frame, f"{finger}: {word}", (x_anchor, y_start + i * 25),
                        font, 0.4, color, 1, cv2.LINE_AA)

    def mark_trigger(self, word):
        self.last_triggered = word
        self.trigger_time = time.time()

    def draw_corner_decorations(self, frame):
        corner_len = 30
        thickness = 2
        margin = 10
        color = COLOR_GOLD
        h, w = self.h, self.w
        cv2.line(frame, (margin, margin), (margin + corner_len, margin), color, thickness)
        cv2.line(frame, (margin, margin), (margin, margin + corner_len), color, thickness)
        cv2.line(frame, (w - margin, margin), (w - margin - corner_len, margin), color, thickness)
        cv2.line(frame, (w - margin, margin), (w - margin, margin + corner_len), color, thickness)
        cv2.line(frame, (margin, h - margin), (margin + corner_len, h - margin), color, thickness)
        cv2.line(frame, (margin, h - margin), (margin, h - margin - corner_len), color, thickness)
        cv2.line(frame, (w - margin, h - margin), (w - margin - corner_len, h - margin), color, thickness)
        cv2.line(frame, (w - margin, h - margin), (w - margin, h - margin - corner_len), color, thickness)


def load_helmet(path):
    """Load helmet PNG with alpha channel."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[WARNING] Could not load helmet image: {path}")
        return None
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def overlay_image(background, overlay_img, x, y, w, h):
    """Overlay a BGRA image onto a BGR frame with alpha blending."""
    resized = cv2.resize(overlay_img, (w, h), interpolation=cv2.INTER_AREA)
    b, g, r, a = cv2.split(resized)
    alpha = a.astype(float) / 255.0

    # Clamp to frame bounds
    y1 = max(0, y)
    y2 = min(background.shape[0], y + h)
    x1 = max(0, x)
    x2 = min(background.shape[1], x + w)

    # Corresponding region in overlay
    oy1 = y1 - y
    oy2 = oy1 + (y2 - y1)
    ox1 = x1 - x
    ox2 = ox1 + (x2 - x1)

    if y2 <= y1 or x2 <= x1:
        return

    roi = background[y1:y2, x1:x2]
    overlay_crop = resized[oy1:oy2, ox1:ox2]
    alpha_crop = alpha[oy1:oy2, ox1:ox2]

    for c in range(3):
        roi[:, :, c] = (alpha_crop * overlay_crop[:, :, c] +
                        (1.0 - alpha_crop) * roi[:, :, c]).astype(np.uint8)


def calculate_distance(lm1, lm2, w, h):
    x1, y1 = int(lm1.x * w), int(lm1.y * h)
    x2, y2 = int(lm2.x * w), int(lm2.y * h)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_midpoint(lm1, lm2, w, h):
    x = int((lm1.x + lm2.x) / 2 * w)
    y = int((lm1.y + lm2.y) / 2 * h)
    return (x, y)


def main():
    global PINCH_THRESHOLD

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print("  Download it with:")
        print('  python -c "import urllib.request; urllib.request.urlretrieve('
              "'https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/latest/hand_landmarker.task', "
              "'hand_landmarker.task')\"")
        sys.exit(1)

    # Store latest async result
    latest_result = [None]

    def on_result(result, output_image, timestamp_ms):
        latest_result[0] = result

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        result_callback=on_result,
    )
    landmarker = HandLandmarker.create_from_options(options)

    audio = AudioEngine()

    # Load helmet overlay and face detector
    helmet_img = load_helmet(HELMET_PATH)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    hud = HUD(frame_w, frame_h)

    gesture_states = {}
    for key in GESTURE_MAP:
        gesture_states[key] = GestureState()

    prev_time = time.time()
    fps = 0.0
    frame_timestamp_ms = 0

    print(f"\n{'='*50}")
    print("  TRACKD - Gesture-Controlled Daft Punk Instrument")
    print(f"{'='*50}")
    print(f"  Resolution: {frame_w}x{frame_h}")
    print(f"  Pinch threshold: {PINCH_THRESHOLD}px")
    print(f"  Controls: q=quit, +/-=threshold")
    print(f"{'='*50}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame capture failed.")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_timestamp_ms += 33
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        frame = cv2.addWeighted(frame, 0.7, np.zeros_like(frame), 0.3, 0)

        # Face detection + helmet overlay
        if helmet_img is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )
            for (fx, fy, fw, fh) in faces:
                # Scale helmet to fully cover head (much wider and taller than face box)
                hw = int(fw * 2.0)
                hh = int(fh * 2.2)
                hx = fx - int(fw * 0.5)
                hy = fy - int(fh * 0.7)
                overlay_image(frame, helmet_img, hx, hy, hw, hh)

        result = latest_result[0]

        if result and result.hand_landmarks and result.handedness:
            for hand_landmarks, handedness_list in zip(
                result.hand_landmarks, result.handedness
            ):
                hand_label = handedness_list[0].category_name

                hud.draw_hand_skeleton(frame, hand_landmarks, hand_label)

                thumb = hand_landmarks[THUMB_TIP]

                for finger_name, tip_idx in FINGER_TIPS.items():
                    gesture_key = (hand_label, finger_name)
                    if gesture_key not in GESTURE_MAP:
                        continue

                    finger_tip = hand_landmarks[tip_idx]
                    dist = calculate_distance(thumb, finger_tip, frame_w, frame_h)
                    is_pinching = dist < PINCH_THRESHOLD

                    triggered = gesture_states[gesture_key].update(is_pinching)

                    if triggered:
                        word, _ = GESTURE_MAP[gesture_key]
                        audio.play(gesture_key)
                        hud.mark_trigger(word)
                        print(f"  {word.upper()}")

                    if is_pinching:
                        midpoint = get_midpoint(thumb, finger_tip, frame_w, frame_h)
                        word, _ = GESTURE_MAP[gesture_key]
                        hud.draw_pinch_indicator(frame, midpoint, word, True)

        hud.draw_corner_decorations(frame)
        hud.draw_header(frame, fps, PINCH_THRESHOLD)
        hud.draw_lyric_bar(frame)
        hud.draw_gesture_guide(frame, "Left", 15)
        hud.draw_gesture_guide(frame, "Right", frame_w - 150)

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 0.001)
        prev_time = current_time

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            PINCH_THRESHOLD = min(PINCH_THRESHOLD + 5, 150)
            print(f"  Threshold: {PINCH_THRESHOLD}px")
        elif key == ord('-') or key == ord('_'):
            PINCH_THRESHOLD = max(PINCH_THRESHOLD - 5, 15)
            print(f"  Threshold: {PINCH_THRESHOLD}px")

    cap.release()
    cv2.destroyAllWindows()
    audio.cleanup()
    landmarker.close()
    print("\n  Trackd closed.\n")


if __name__ == "__main__":
    main()

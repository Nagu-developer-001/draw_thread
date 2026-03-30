"""
╔══════════════════════════════════════════════════╗
║   🎯  AR HAND GESTURE SHOOTER  (Lightweight)     ║
║   Python + OpenCV + MediaPipe                    ║
╚══════════════════════════════════════════════════╝

INSTALL:
    pip install opencv-python mediapipe numpy

CONTROLS:
    👆  Point index finger  →  AIM (moves cursor)
    🤏  Pinch (thumb + index close)  →  POP the ball!
    Q   →  Quit
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time

# ── MediaPipe ──────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# ── Window size ────────────────────────────────────────────────────
W, H = 900, 600

# ── Ball colors ────────────────────────────────────────────────────
BALL_COLORS = [
    (0,   255, 0),    # green
    (255, 100, 0),    # blue
    (200, 0,   200),  # purple
    (0,   220, 255),  # yellow
    (0,   120, 255),  # orange
    (255, 0,   100),  # pink-blue
]

# ── Ball ───────────────────────────────────────────────────────────
class Ball:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x     = float(random.randint(80, W - 80))
        self.y     = float(random.randint(80, H - 80))
        self.r     = random.randint(28, 50)
        self.color = random.choice(BALL_COLORS)
        self.vx    = random.uniform(-1.5, 1.5)
        self.vy    = random.uniform(-1.5, 1.5)
        self.alive = True
        self.pop_anim = 0.0   # 0 = alive, >0 = popping

    def update(self, dt):
        if self.pop_anim > 0:
            self.pop_anim -= dt * 3.5
            if self.pop_anim <= 0:
                self.alive = False
            return

        self.x += self.vx
        self.y += self.vy

        # Bounce walls
        if self.x - self.r < 0:
            self.x  = self.r
            self.vx = abs(self.vx)
        if self.x + self.r > W:
            self.x  = W - self.r
            self.vx = -abs(self.vx)
        if self.y - self.r < 0:
            self.y  = self.r
            self.vy = abs(self.vy)
        if self.y + self.r > H:
            self.y  = H - self.r
            self.vy = -abs(self.vy)

    def pop(self):
        self.pop_anim = 1.0

    def draw(self, frame):
        cx, cy = int(self.x), int(self.y)
        if self.pop_anim > 0:
            # Expanding burst ring
            t = 1.0 - self.pop_anim
            burst_r = int(self.r + t * 60)
            alpha   = self.pop_anim
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), burst_r, self.color, 3)
            cv2.addWeighted(overlay, alpha * 0.8, frame, 1 - alpha * 0.8, 0, frame)
            # Particles effect — radial lines
            for i in range(8):
                angle = math.radians(i * 45 + t * 180)
                x1 = int(cx + (self.r + t * 20) * math.cos(angle))
                y1 = int(cy + (self.r + t * 20) * math.sin(angle))
                x2 = int(cx + (self.r + t * 55) * math.cos(angle))
                y2 = int(cy + (self.r + t * 55) * math.sin(angle))
                col = tuple(int(c * alpha) for c in self.color)
                cv2.line(frame, (x1, y1), (x2, y2), col, 2)
            return

        # Glow
        for g in range(3, 0, -1):
            ov = frame.copy()
            cv2.circle(ov, (cx, cy), self.r + g * 3, self.color, 2)
            cv2.addWeighted(ov, 0.12 * g, frame, 1 - 0.12 * g, 0, frame)

        # Fill
        cv2.circle(frame, (cx, cy), self.r, self.color, -1)

        # Shine highlight
        hx = cx - self.r // 3
        hy = cy - self.r // 3
        hr = max(4, self.r // 4)
        cv2.circle(frame, (hx, hy), hr, (255, 255, 255), -1)


# ── Cursor / crosshair ─────────────────────────────────────────────
def draw_cursor(frame, cx, cy, pinching):
    color = (0, 255, 255) if not pinching else (0, 80, 255)
    r     = 18
    gap   = 6
    thick = 3 if pinching else 2

    cv2.line(frame, (cx - r, cy), (cx - gap, cy), color, thick)
    cv2.line(frame, (cx + gap, cy), (cx + r, cy), color, thick)
    cv2.line(frame, (cx, cy - r), (cx, cy - gap), color, thick)
    cv2.line(frame, (cx, cy + gap), (cx, cy + r), color, thick)
    cv2.circle(frame, (cx, cy), r + 4, color, 1)
    if pinching:
        cv2.circle(frame, (cx, cy), 6, color, -1)


# ── Hand detection ─────────────────────────────────────────────────
def get_hand(landmarks, w, h):
    """Returns (cursor_xy, is_pinching)"""
    lm = landmarks.landmark

    # Index fingertip = cursor
    ix = int(lm[8].x * w)
    iy = int(lm[8].y * h)
    ix = max(0, min(w - 1, ix))
    iy = max(0, min(h - 1, iy))

    # Pinch: thumb tip (4) <-> index tip (8)
    tx = int(lm[4].x * w)
    ty = int(lm[4].y * h)
    pinch_dist = math.hypot(tx - ix, ty - iy)
    pinching   = pinch_dist < 40

    return (ix, iy), pinching


# ── Score flash ────────────────────────────────────────────────────
class ScoreFlash:
    def __init__(self, x, y, text):
        self.x    = x
        self.y    = float(y)
        self.text = text
        self.life = 1.2

    def update(self, dt):
        self.life -= dt
        self.y    -= 1.2

    def draw(self, frame):
        alpha = min(1.0, self.life)
        col   = (0, int(255 * alpha), int(200 * alpha))
        cv2.putText(frame, self.text, (self.x - 20, int(self.y)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(frame, self.text, (self.x - 20, int(self.y)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, col, 2)


# ── Main ───────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    # Use smaller capture resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
        model_complexity=0,          # 0 = lightest & fastest
    )

    # Game state
    score       = 0
    high_score  = 0
    balls       = [Ball() for _ in range(4)]
    flashes     = []
    cursor      = (W // 2, H // 2)
    was_pinching= False
    pinch_lock  = False

    prev_time = time.time()
    fps_val   = 30.0

    print("\n🎯  AR HAND GESTURE SHOOTER  (Lightweight)")
    print("   Point = AIM  |  Pinch = SHOOT  |  Q = Quit\n")

    while True:
        ret, raw = cap.read()
        if not ret:
            break

        # ── Flip webcam ──
        raw = cv2.flip(raw, 1)

        # ── Resize webcam for mediapipe (fast) ──
        small = cv2.resize(raw, (320, 240))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result= hands.process(rgb)

        # ── Delta time ──
        now = time.time()
        dt  = min(now - prev_time, 0.1)
        prev_time = now
        fps_val = fps_val * 0.9 + (1.0 / max(dt, 0.001)) * 0.1

        # ── Black canvas (not webcam bg — matches video style) ──
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # ── Hand tracking ──
        pinching = False
        hand_lms = None
        if result.multi_hand_landmarks:
            hand_lms = result.multi_hand_landmarks[0]
            # Scale back to game resolution
            raw_cursor, pinching = get_hand(hand_lms, W, H)
            # Smooth cursor
            cx = int(cursor[0] * 0.35 + raw_cursor[0] * 0.65)
            cy = int(cursor[1] * 0.35 + raw_cursor[1] * 0.65)
            cursor = (cx, cy)

        # ── Pinch: shoot (only on new pinch) ──
        if pinching and not was_pinching and not pinch_lock:
            pinch_lock = True
            cx, cy = cursor
            for ball in balls:
                if ball.pop_anim == 0 and ball.alive:
                    d = math.hypot(cx - ball.x, cy - ball.y)
                    if d < ball.r + 20:
                        ball.pop()
                        pts = 10 + max(0, 50 - ball.r)   # smaller = more points
                        score += pts
                        high_score = max(high_score, score)
                        flashes.append(ScoreFlash(cx, cy, f"+{pts}"))
                        break
        if not pinching:
            pinch_lock = False
        was_pinching = pinching

        # ── Respawn dead balls ──
        for i, b in enumerate(balls):
            if not b.alive:
                nb = Ball()
                balls[i] = nb

        # ── Update balls ──
        for b in balls:
            b.update(dt)

        # ── Draw balls ──
        for b in balls:
            b.draw(frame)

        # ── Score flashes ──
        flashes = [f for f in flashes if f.life > 0]
        for f in flashes:
            f.update(dt)
            f.draw(frame)

        # ── Cursor ──
        draw_cursor(frame, cursor[0], cursor[1], pinching)

        # ── HUD (minimal, top-right like in video) ──
        hud_font  = cv2.FONT_HERSHEY_DUPLEX
        hud_font2 = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"SCORE  {score}", (W - 220, 38), hud_font, 0.85, (0,0,0),   4)
        cv2.putText(frame, f"SCORE  {score}", (W - 220, 38), hud_font, 0.85, (57,255,20),2)
        cv2.putText(frame, f"BEST   {high_score}", (W - 220, 62), hud_font2, 0.55, (0,200,150), 1)
        cv2.putText(frame, f"FPS {int(fps_val)}", (10, 22), hud_font2, 0.5, (80, 80, 80), 1)

        # ── Webcam thumbnail (bottom right corner, like in video) ──
        thumb_w, thumb_h = 160, 120
        thumb = cv2.resize(raw, (thumb_w, thumb_h))
        # Draw skeleton on thumbnail
        if hand_lms:
            small_thumb = cv2.resize(raw, (320, 240))
            mp_draw.draw_landmarks(
                small_thumb, hand_lms, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=2),
                mp_draw.DrawingSpec(color=(57,255,20), thickness=1, circle_radius=1),
            )
            thumb = cv2.resize(small_thumb, (thumb_w, thumb_h))

        tx0 = W - thumb_w - 10
        ty0 = H - thumb_h - 10
        frame[ty0:ty0+thumb_h, tx0:tx0+thumb_w] = thumb
        cv2.rectangle(frame, (tx0-1, ty0-1), (tx0+thumb_w, ty0+thumb_h), (0,255,255), 1)

        # ── Gesture hint ──
        hint = "PINCH = SHOOT" if not pinching else "🎯 FIRE!"
        col  = (100,100,100) if not pinching else (0, 80, 255)
        cv2.putText(frame, hint, (10, H - 14), hud_font2, 0.55, col, 1)

        cv2.imshow("🎯 AR Hand Gesture Shooter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"\n🏆  Final Score: {score}   Best: {high_score}")


if __name__ == "__main__":
    main()
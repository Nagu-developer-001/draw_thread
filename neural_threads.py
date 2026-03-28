import cv2
import mediapipe as mp
import numpy as np
import math, random, time

# ══════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════
WIN   = "Neural Threads"
N     = 64          # curve knots per thread

# Finger tip landmark IDs
TIPS  = [4, 8, 12, 16, 20]

# One distinct hue per finger (thumb→pinky)
# warm gold, electric teal, soft violet, rose, lime
HUES  = [0.11, 0.50, 0.72, 0.93, 0.30]

# ══════════════════════════════════════════════════
#  TINY HELPERS
# ══════════════════════════════════════════════════

def lerp(a, b, t):
    return a + (b - a) * t

def lp2(p, q, t):
    return (lerp(p[0], q[0], t), lerp(p[1], q[1], t))

def perp(dx, dy):
    L = math.hypot(dx, dy)
    return (-dy / L, dx / L) if L > 1e-6 else (0., 0.)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def tip_px(lm, idx, W, H):
    l = lm.landmark[idx]
    return (int(clamp(l.x * W, 0, W-1)),
            int(clamp(l.y * H, 0, H-1)))

def hsv(h, s, v):
    h = h % 1.
    i = int(h * 6); f = h * 6 - i
    p, q, t2 = v*(1-s), v*(1-s*f), v*(1-s*(1-f))
    i %= 6
    r, g, b = [(v,t2,p),(q,v,p),(p,v,t2),(p,q,v),(t2,p,v),(v,p,q)][i]
    return (int(b*255), int(g*255), int(r*255))

# ══════════════════════════════════════════════════
#  CURVE BUILDER
#  Generates a smooth sinusoidal wave between two points
#  offset_side: perpendicular offset as fraction of length
# ══════════════════════════════════════════════════

def wave_curve(p1, p2, phase, freq, amp_frac, offset_side=0., n=N):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    L      = math.hypot(dx, dy)
    px, py = perp(dx, dy)
    amp    = clamp(L * amp_frac, 4., 38.)
    side   = L * offset_side

    pts = []
    for i in range(n):
        t  = i / (n - 1)
        bx, by = lp2(p1, p2, t)
        env    = math.sin(math.pi * t)          # taper to zero at endpoints
        wave   = amp * env * math.sin(freq * math.pi * t * 2 + phase)
        ox     = px * (wave + side)
        oy     = py * (wave + side)
        pts.append((int(bx + ox), int(by + oy)))
    return pts

# ══════════════════════════════════════════════════
#  DRAW ONE THREAD  (3 layered lines)
# ══════════════════════════════════════════════════

def draw_thread(frame, glow, p1, p2, hue, phase, t):
    """
    3 lines per finger pair:
      1. Core        — bright, gradient, central
      2. Left flank  — offset left, complementary hue, slower wave
      3. Right flank — offset right, triadic hue, faster wave
    """
    # ── time-varying wave parameters ──────────────
    freq_c = 2.8 + 0.6 * math.sin(t * 0.7 + hue * 5)
    freq_l = 1.9 + 0.5 * math.sin(t * 0.5 + hue * 3)
    freq_r = 3.7 + 0.5 * math.sin(t * 0.9 + hue * 7)

    # ── 1. CORE BEAM ──────────────────────────────
    core = wave_curve(p1, p2, phase, freq_c, 0.06, 0.)
    for i in range(len(core) - 1):
        tt = i / (len(core) - 1)
        h  = (hue + tt * 0.10) % 1.
        # main frame line
        cv2.line(frame, core[i], core[i+1], hsv(h, 1., .95), 2, cv2.LINE_AA)
        # thick glow copy
        cv2.line(glow,  core[i], core[i+1], hsv(h, .7, 1.),  7, cv2.LINE_AA)

    # ── 2. LEFT AURORA ────────────────────────────
    hue_l = (hue + 0.28) % 1.
    left  = wave_curve(p1, p2, phase * 1.2 + 1.0, freq_l, 0.05, +0.05)
    for i in range(len(left) - 1):
        tt = i / (len(left) - 1)
        h  = (hue_l + tt * 0.08) % 1.
        a  = 0.55 + 0.30 * math.sin(t * 2.1 + tt * 5)
        cv2.line(frame, left[i], left[i+1], hsv(h, 1., a),   1, cv2.LINE_AA)
        cv2.line(glow,  left[i], left[i+1], hsv(h, .55, 1.), 3, cv2.LINE_AA)

    # ── 3. RIGHT AURORA ───────────────────────────
    hue_r = (hue + 0.55) % 1.
    right = wave_curve(p1, p2, phase * 0.8 - 0.7, freq_r, 0.05, -0.05)
    for i in range(len(right) - 1):
        tt = i / (len(right) - 1)
        h  = (hue_r + tt * 0.08) % 1.
        a  = 0.55 + 0.30 * math.sin(t * 1.8 + tt * 5 + 1.)
        cv2.line(frame, right[i], right[i+1], hsv(h, 1., a),   1, cv2.LINE_AA)
        cv2.line(glow,  right[i], right[i+1], hsv(h, .55, 1.), 3, cv2.LINE_AA)

    # ── Endpoint dots ─────────────────────────────
    for ep in [p1, p2]:
        r = int(4 + 2 * math.sin(t * 4 + hue * 18))
        cv2.circle(frame, ep, r,     hsv(hue, 1., 1.),  -1, cv2.LINE_AA)
        cv2.circle(glow,  ep, r + 6, hsv(hue, .5, 1.),  -1, cv2.LINE_AA)

# ══════════════════════════════════════════════════
#  BLOOM
# ══════════════════════════════════════════════════

def bloom(frame, glow, strength=0.80):
    b = cv2.GaussianBlur(glow, (0, 0), 18)
    cv2.addWeighted(frame, 1., b, strength, 0, dst=frame)

# ══════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════

def main():
    mp_h = mp.solutions.hands
    mp_d = mp.solutions.drawing_utils
    sol  = mp_h.Hands(
        static_image_mode        = False,
        max_num_hands            = 2,
        min_detection_confidence = 0.65,
        min_tracking_confidence  = 0.60,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No webcam found."); return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    t0 = time.time()
    print(f"[Neural Threads]  {W}×{H}   Q = quit")
    print("Show Your hands — 5 threads connect finger to finger.")

    while True:
        ret, raw = cap.read()
        if not ret:
            break

        frame = cv2.flip(raw, 1)

        try:
            t     = time.time() - t0
            phase = t * 2.2          # global wave phase

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = sol.process(rgb)

            # ── Dark background ────────────────────────
            frame = (frame.astype(np.float32) * 0.22).astype(np.uint8)
            glow  = np.zeros_like(frame)

            # ── Parse hands ────────────────────────────
            tips_L = None   # list of 5 (x,y) for left hand
            tips_R = None

            if result.multi_hand_landmarks and result.multi_handedness:
                for lm, hi in zip(result.multi_hand_landmarks,
                                  result.multi_handedness):
                    label = hi.classification[0].label  # "Left" / "Right"

                    # invisible skeleton — very dim
                    mp_d.draw_landmarks(
                        frame, lm, mp_h.HAND_CONNECTIONS,
                        mp_d.DrawingSpec(color=(20, 15, 40),  thickness=1, circle_radius=1),
                        mp_d.DrawingSpec(color=(30, 20, 55),  thickness=1),
                    )

                    tips = [tip_px(lm, tid, W, H) for tid in TIPS]

                    if label == "Left":
                        tips_L = tips
                    else:
                        tips_R = tips

            # ── Draw threads when both hands visible ────
            if tips_L and tips_R:
                for fi in range(5):
                    draw_thread(
                        frame, glow,
                        tips_L[fi], tips_R[fi],
                        HUES[fi],
                        phase + fi * 1.2,   # stagger phase per finger
                        t
                    )

            # ── Bloom pass ─────────────────────────────
            bloom(frame, glow)

            # ── Minimal HUD ────────────────────────────
            status = "both hands" if (tips_L and tips_R) else "show both hands..."
            cv2.putText(frame, "HAND GESTURES",
                        (14, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (140, 80, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, status,
                        (14, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (70, 50, 110), 1, cv2.LINE_AA)

        except Exception as e:
            print(f"[skip] {e}")

        cv2.imshow(WIN, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    sol.close()
    print("Done.")

if __name__ == "__main__":
    main()
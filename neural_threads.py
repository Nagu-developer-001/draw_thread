import cv2
import mediapipe as mp
import numpy as np
import math, time, subprocess, webbrowser, platform, os

# ══════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════
WIN   = "Neural Threads"
N     = 64          # curve knots per thread

# Finger tip landmark IDs
TIPS  = [4, 8, 12, 16, 20]

# One distinct hue per finger (thumb→pinky)
HUES  = [0.11, 0.50, 0.72, 0.93, 0.30]

# ══════════════════════════════════════════════════
#  GESTURE LAUNCHER CONFIG  ← edit these freely
#
#  Each entry: thumb (tip 4) pinched against finger tip
#    tip 8  = Index   finger
#    tip 12 = Middle  finger
#    tip 16 = Ring    finger
#    tip 20 = Pinky   finger
#
#  type  : "web"  → opens URL in default browser
#           "app"  → launches a desktop application
#
#  For "app" on Windows  use the exe name, e.g. "notepad", "calc", "mspaint"
#            on macOS    use the app name,  e.g. "Safari", "Calculator", "Notes"
#            on Linux    use the binary,    e.g. "firefox", "gedit", "gnome-calculator"
# ══════════════════════════════════════════════════
LAUNCH_MAP = {
    8:  {"type": "web", "target": "https://www.google.com",   "label": "Google",      "hue": 0.50},
    12: {"type": "web", "target": "https://www.youtube.com",  "label": "YouTube",     "hue": 0.93},
    16: {"type": "web", "target": "https://www.github.com",   "label": "GitHub",      "hue": 0.72},
    20: {"type": "app", "target": "calc",                     "label": "Calculator",  "hue": 0.30},
    #
    # More examples you can uncomment / swap in:
    # 20: {"type": "app",  "target": "notepad",              "label": "Notepad",     "hue": 0.30},
    # 20: {"type": "app",  "target": "mspaint",              "label": "Paint",       "hue": 0.30},
    # 16: {"type": "web",  "target": "https://chat.openai.com","label":"ChatGPT",    "hue": 0.72},
}

PINCH_THRESHOLD_FRAC = 0.06   # fraction of frame width = pinch distance trigger
COOLDOWN_SEC         = 2.5    # seconds before same gesture can fire again

# ── SHUSH-TO-QUIT gesture ──────────────────────────
# Hold index finger tip near your lips for this long → program exits
SHUSH_HOLD_SEC       = 1.5
SHUSH_RADIUS_FRAC    = 0.10   # fraction of frame width = "near lips" radius
# MediaPipe FaceMesh lip landmarks (upper + lower inner lip centres)
MOUTH_LM_IDS         = [13, 14, 78, 308]  # averaged to get mouth centre

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

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def hsv(h, s, v):
    h = h % 1.
    i = int(h * 6); f = h * 6 - i
    p, q, t2 = v*(1-s), v*(1-s*f), v*(1-s*(1-f))
    i %= 6
    r, g, b = [(v,t2,p),(q,v,p),(p,v,t2),(p,q,v),(t2,p,v),(v,p,q)][i]
    return (int(b*255), int(g*255), int(r*255))

# ══════════════════════════════════════════════════
#  LAUNCHER
# ══════════════════════════════════════════════════

def launch(entry):
    """Open an app or URL depending on entry type."""
    try:
        if entry["type"] == "web":
            webbrowser.open(entry["target"])
        else:
            OS = platform.system()
            app = entry["target"]
            if OS == "Windows":
                os.startfile(app) if os.path.isabs(app) else subprocess.Popen(
                    ["cmd", "/c", "start", "", app], shell=False)
            elif OS == "Darwin":   # macOS
                subprocess.Popen(["open", "-a", app])
            else:                  # Linux
                subprocess.Popen([app])
        print(f"[Launch] {entry['label']} → {entry['target']}")
    except Exception as e:
        print(f"[Launch ERROR] {e}")

# ══════════════════════════════════════════════════
#  CURVE BUILDER
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
        env    = math.sin(math.pi * t)
        wave   = amp * env * math.sin(freq * math.pi * t * 2 + phase)
        ox     = px * (wave + side)
        oy     = py * (wave + side)
        pts.append((int(bx + ox), int(by + oy)))
    return pts

# ══════════════════════════════════════════════════
#  DRAW ONE THREAD
# ══════════════════════════════════════════════════

def draw_thread(frame, glow, p1, p2, hue, phase, t):
    freq_c = 2.8 + 0.6 * math.sin(t * 0.7 + hue * 5)
    freq_l = 1.9 + 0.5 * math.sin(t * 0.5 + hue * 3)
    freq_r = 3.7 + 0.5 * math.sin(t * 0.9 + hue * 7)

    core = wave_curve(p1, p2, phase, freq_c, 0.06, 0.)
    for i in range(len(core) - 1):
        tt = i / (len(core) - 1)
        h  = (hue + tt * 0.10) % 1.
        cv2.line(frame, core[i], core[i+1], hsv(h, 1., .95), 2, cv2.LINE_AA)
        cv2.line(glow,  core[i], core[i+1], hsv(h, .7, 1.),  7, cv2.LINE_AA)

    hue_l = (hue + 0.28) % 1.
    left  = wave_curve(p1, p2, phase * 1.2 + 1.0, freq_l, 0.05, +0.05)
    for i in range(len(left) - 1):
        tt = i / (len(left) - 1)
        h  = (hue_l + tt * 0.08) % 1.
        a  = 0.55 + 0.30 * math.sin(t * 2.1 + tt * 5)
        cv2.line(frame, left[i], left[i+1], hsv(h, 1., a),   1, cv2.LINE_AA)
        cv2.line(glow,  left[i], left[i+1], hsv(h, .55, 1.), 3, cv2.LINE_AA)

    hue_r = (hue + 0.55) % 1.
    right = wave_curve(p1, p2, phase * 0.8 - 0.7, freq_r, 0.05, -0.05)
    for i in range(len(right) - 1):
        tt = i / (len(right) - 1)
        h  = (hue_r + tt * 0.08) % 1.
        a  = 0.55 + 0.30 * math.sin(t * 1.8 + tt * 5 + 1.)
        cv2.line(frame, right[i], right[i+1], hsv(h, 1., a),   1, cv2.LINE_AA)
        cv2.line(glow,  right[i], right[i+1], hsv(h, .55, 1.), 3, cv2.LINE_AA)

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
#  PINCH HUD  — shows the gesture → action legend
# ══════════════════════════════════════════════════

FINGER_NAMES = {8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}

def draw_hud(frame, H, W, last_fired, now):
    """
    Bottom-left legend: each row shows  [finger icon]  label
    Active / cooling-down entries pulse or dim.
    """
    x0, y0 = 14, H - 14
    pad     = 22

    # background pill
    rows = len(LAUNCH_MAP)
    bh   = rows * pad + 10
    cv2.rectangle(frame,
                  (x0 - 6, y0 - bh),
                  (x0 + 220, y0 + 6),
                  (15, 10, 30), -1)
    cv2.rectangle(frame,
                  (x0 - 6, y0 - bh),
                  (x0 + 220, y0 + 6),
                  (60, 40, 100), 1)

    for row, (tip_id, entry) in enumerate(LAUNCH_MAP.items()):
        y = y0 - row * pad - 10
        cd_key = tip_id
        elapsed = now - last_fired.get(cd_key, 0)
        cooling  = elapsed < COOLDOWN_SEC

        icon = FINGER_NAMES.get(tip_id, "?")
        label = entry["label"]
        target_short = entry["target"].replace("https://www.", "")[:22]
        hue  = entry["hue"]

        base_col = hsv(hue, 1., 0.4 if cooling else 0.85)
        dim_col  = hsv(hue, .4, 0.25)

        # finger name
        cv2.putText(frame, f"Thumb+{icon}", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    dim_col if cooling else base_col, 1, cv2.LINE_AA)
        # arrow + label
        cv2.putText(frame, f"-> {label}  [{target_short}]",
                    (x0 + 85, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                    dim_col if cooling else base_col, 1, cv2.LINE_AA)

        # cooldown bar
        if cooling:
            bar_w = int(200 * (1 - elapsed / COOLDOWN_SEC))
            cv2.rectangle(frame, (x0, y + 2), (x0 + bar_w, y + 4),
                          hsv(hue, .8, .6), -1)

# ══════════════════════════════════════════════════
#  PINCH DETECTION  (single-hand, thumb vs each finger)
# ══════════════════════════════════════════════════

def check_pinches(lm, W, H, last_fired, now):
    """
    Returns list of tip IDs that are currently pinched against thumb.
    Also fires launch() if cooldown has expired.
    """
    thumb = tip_px(lm, 4, W, H)
    threshold = W * PINCH_THRESHOLD_FRAC
    fired = []

    for tip_id, entry in LAUNCH_MAP.items():
        finger_tip = tip_px(lm, tip_id, W, H)
        d = dist(thumb, finger_tip)
        if d < threshold:
            elapsed = now - last_fired.get(tip_id, 0)
            if elapsed >= COOLDOWN_SEC:
                last_fired[tip_id] = now
                launch(entry)
                fired.append(tip_id)

    return fired

# ══════════════════════════════════════════════════
#  PINCH FLASH OVERLAY  (brief flash on trigger)
# ══════════════════════════════════════════════════

def draw_flash(frame, glow, lm, tip_id, W, H, hue):
    thumb = tip_px(lm, 4, W, H)
    fing  = tip_px(lm, tip_id, W, H)
    mid   = ((thumb[0]+fing[0])//2, (thumb[1]+fing[1])//2)
    cv2.circle(glow,  mid, 40, hsv(hue, .5, 1.), -1, cv2.LINE_AA)
    cv2.circle(frame, mid, 12, hsv(hue, 1., 1.), -1, cv2.LINE_AA)
    label = LAUNCH_MAP[tip_id]["label"]
    cv2.putText(frame, f"Opening {label}!",
                (mid[0] - 55, mid[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                hsv(hue, .3, 1.), 1, cv2.LINE_AA)

# ══════════════════════════════════════════════════
#  SHUSH-TO-QUIT  — index finger near lips → exit
# ══════════════════════════════════════════════════

def get_mouth_px(face_lm, W, H):
    """Average several lip landmarks → (x, y) in pixels."""
    xs = [face_lm.landmark[i].x for i in MOUTH_LM_IDS]
    ys = [face_lm.landmark[i].y for i in MOUTH_LM_IDS]
    mx = int(clamp(sum(xs) / len(xs) * W, 0, W-1))
    my = int(clamp(sum(ys) / len(ys) * H, 0, H-1))
    return (mx, my)

def check_shush(hand_lm, mouth_pt, W, H, now, shush_state):
    """
    shush_state = {"start": float or None}
    Returns (is_shushing, hold_fraction 0‥1, should_quit)
    """
    index_tip = tip_px(hand_lm, 8, W, H)   # landmark 8 = index fingertip
    threshold  = W * SHUSH_RADIUS_FRAC
    near       = dist(index_tip, mouth_pt) < threshold

    if near:
        if shush_state["start"] is None:
            shush_state["start"] = now
        held     = now - shush_state["start"]
        fraction = min(held / SHUSH_HOLD_SEC, 1.0)
        quit_now = held >= SHUSH_HOLD_SEC
        return True, fraction, quit_now
    else:
        shush_state["start"] = None
        return False, 0.0, False

def draw_shush_indicator(frame, glow, mouth_pt, fraction, now):
    """Animated ring around the mouth that fills up as you hold the pose."""
    cx, cy = mouth_pt
    radius = 48
    # pulsing outer glow
    pulse  = int(6 + 4 * math.sin(now * 10))
    cv2.circle(glow,  (cx, cy), radius + pulse, hsv(0.0, .6, 1.), 3, cv2.LINE_AA)

    # arc that fills clockwise as fraction → 1
    angle  = int(360 * fraction)
    color  = hsv(0.0 + fraction * 0.15, 1., 1.)   # white → warm as it fills
    cv2.ellipse(frame, (cx, cy), (radius, radius),
                -90, 0, angle, color, 3, cv2.LINE_AA)

    # centre dot
    cv2.circle(frame, (cx, cy), 5, hsv(0., 1., 1.), -1, cv2.LINE_AA)

    # label
    secs_left = SHUSH_HOLD_SEC * (1 - fraction)
    txt = f"🤫 Quitting in {secs_left:.1f}s" if fraction > 0.05 else "🤫 Hold to quit"
    cv2.putText(frame, txt,
                (cx - 70, cy - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                hsv(0., .3, 1.), 1, cv2.LINE_AA)

# ══════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════

def main():
    mp_h  = mp.solutions.hands
    mp_fm = mp.solutions.face_mesh
    mp_d  = mp.solutions.drawing_utils
    sol   = mp_h.Hands(
        static_image_mode        = False,
        max_num_hands            = 2,
        min_detection_confidence = 0.65,
        min_tracking_confidence  = 0.60,
    )
    # FaceMesh — only need the mouth region, refine=True gives lip landmarks
    face_sol = mp_fm.FaceMesh(
        static_image_mode        = False,
        max_num_faces            = 1,
        refine_landmarks         = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No webcam found."); return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    t0          = time.time()
    last_fired  = {}          # tip_id → last launch timestamp
    flash_q     = []          # [(lm, tip_id, hue, expire_time)]
    shush_state = {"start": None}   # tracks shush hold start time

    print(f"[Neural Threads]  {W}×{H}   Q = quit")
    print("Show both hands → threads appear.")
    print("Pinch thumb+finger on EITHER hand → launch app/site!")
    print("Index finger to lips → hold 1.5 s → quit")

    while True:
        ret, raw = cap.read()
        if not ret:
            break

        frame = cv2.flip(raw, 1)

        try:
            t     = time.time() - t0
            now   = time.time()
            phase = t * 2.2

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = sol.process(rgb)
            face_r = face_sol.process(rgb)

            # ── Dark background ────────────────────────
            frame = (frame.astype(np.float32) * 0.22).astype(np.uint8)
            glow  = np.zeros_like(frame)

            # ── Parse hands ────────────────────────────
            tips_L    = None
            tips_R    = None
            lm_all    = []

            if result.multi_hand_landmarks and result.multi_handedness:
                for lm, hi in zip(result.multi_hand_landmarks,
                                  result.multi_handedness):
                    label = hi.classification[0].label

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

                    lm_all.append(lm)

                    # ── Pinch detection (any hand) ──────
                    fired = check_pinches(lm, W, H, last_fired, now)
                    for tip_id in fired:
                        h = LAUNCH_MAP[tip_id]["hue"]
                        flash_q.append((lm, tip_id, h, now + 0.8))

            # ── Shush-to-quit detection ─────────────────
            should_quit = False
            if face_r.multi_face_landmarks and lm_all:
                face_lm   = face_r.multi_face_landmarks[0]
                mouth_pt  = get_mouth_px(face_lm, W, H)
                # check every visible hand; any one can trigger shush
                for hand_lm in lm_all:
                    is_sh, frac, sq = check_shush(
                        hand_lm, mouth_pt, W, H, now, shush_state)
                    if is_sh:
                        draw_shush_indicator(frame, glow, mouth_pt, frac, now)
                        if sq:
                            should_quit = True
                        break
                else:
                    # no hand near lips → reset state
                    shush_state["start"] = None

            # ── Draw threads when both hands visible ────
            if tips_L and tips_R:
                for fi in range(5):
                    draw_thread(
                        frame, glow,
                        tips_L[fi], tips_R[fi],
                        HUES[fi],
                        phase + fi * 1.2,
                        t
                    )

            # ── Draw active flash overlays ──────────────
            flash_q[:] = [(lm, tid, h, exp) for lm, tid, h, exp in flash_q if now < exp]
            for lm, tid, h, _ in flash_q:
                draw_flash(frame, glow, lm, tid, W, H, h)

            # ── Bloom pass ─────────────────────────────
            bloom(frame, glow)

            # ── HUD ────────────────────────────────────
            draw_hud(frame, H, W, last_fired, now)

            status = "both hands ✓" if (tips_L and tips_R) else "show both hands..."
            cv2.putText(frame, "NEURAL THREADS",
                        (14, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (140, 80, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, status,
                        (14, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (70, 50, 110), 1, cv2.LINE_AA)

        except Exception as e:
            print(f"[skip] {e}")

        cv2.imshow(WIN, frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or should_quit:
            if should_quit:
                print("[Shush] Quit gesture held — bye!")
            break

    cap.release()
    cv2.destroyAllWindows()
    sol.close()
    face_sol.close()
    print("Done.")

if __name__ == "__main__":
    main()
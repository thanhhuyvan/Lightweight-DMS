"""
demo_app.py  —  Stage E DMS Live Demo (Streamlit)
Run:  streamlit run demo_app.py
"""

import time
import math
import random
import numpy as np
import streamlit as st
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Stage E DMS — Live Demo",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Remove default padding */
    .block-container { padding-top: 0.5rem; padding-bottom: 0rem; }
    /* Card-style boxes */
    .metric-card {
        background: #1E1E2E;
        border-radius: 10px;
        padding: 10px 16px;
        text-align: center;
        border: 1px solid #313244;
    }
    .metric-val  { font-size: 1.5rem; font-weight: 700; color: #CDD6F4; }
    .metric-lbl  { font-size: 0.72rem; color: #6C7086; text-transform: uppercase; letter-spacing: 1px; }
    /* Alert banners */
    .alert-green {
        background: #1e3a2f;
        border: 2px solid #2ECC71;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        color: #2ECC71;
        font-size: 2rem;
        font-weight: 900;
        letter-spacing: 4px;
    }
    .alert-red {
        background: #3a1e1e;
        border: 2px solid #E74C3C;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        color: #E74C3C;
        font-size: 2rem;
        font-weight: 900;
        letter-spacing: 4px;
        animation: pulse 1s infinite;
    }
    .failsafe-normal   { background:#1a2a1a; border:1px solid #2ECC71; border-radius:8px; padding:10px; color:#2ECC71; font-size:0.82rem; }
    .failsafe-critical { background:#2a1a1a; border:1px solid #E74C3C; border-radius:8px; padding:10px; color:#E74C3C; font-size:0.82rem; }
    .failsafe-formula  { background:#1a1a2a; border:1px solid #3498DB; border-radius:8px; padding:10px; color:#3498DB; font-size:0.82rem; }
    .section-header { color:#CDD6F4; font-size:0.8rem; font-weight:700;
                      text-transform:uppercase; letter-spacing:2px; margin-bottom:4px; }
    /* Dark background for whole app */
    .stApp { background-color: #11111B; }
    /* Hide hamburger */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA SIMULATION ENGINE
# ─────────────────────────────────────────────
BUFFER_LEN = 40          # 40-frame window
CYCLE_SEC  = 12          # seconds per alert→drowsy→alert cycle
FPS_SIM    = 4           # simulated fps

def _smooth(val, target, alpha=0.15):
    return val + alpha * (target - val)

class DrivingSimulator:
    """
    Simulates a driver cycling through ALERT → DROWSY → ALERT states.
    All signals are mathematically correlated.
    """
    def __init__(self):
        self.t       = 0.0
        self.ear     = 0.28
        self.mar     = 0.03
        self.perclos = 0.05
        self.phase   = "ALERT"   # or DROWSY
        self.phase_t = 0.0
        self.occlusion = False

    def step(self):
        self.t      += 1 / FPS_SIM
        self.phase_t += 1 / FPS_SIM

        # Cycle: 8s alert, 4s drowsy
        if self.phase == "ALERT" and self.phase_t > 8:
            self.phase   = "DROWSY"
            self.phase_t = 0.0
        elif self.phase == "DROWSY" and self.phase_t > 4:
            self.phase   = "ALERT"
            self.phase_t = 0.0

        # Target values per phase
        if self.phase == "ALERT":
            ear_target     = 0.27 + 0.04 * math.sin(self.t * 0.8)   # natural blink rhythm
            mar_target     = 0.03 + 0.01 * random.random()
            perclos_target = 0.05 + 0.03 * random.random()
        else:  # DROWSY
            # EAR drops, MAR spikes (yawn), PERCLOS rises
            ear_target     = 0.15 + 0.03 * math.sin(self.t * 2.0)
            yawn           = max(0, math.sin(self.t * 1.5)) ** 2
            mar_target     = 0.06 + 0.20 * yawn + 0.02 * random.random()
            perclos_target = 0.35 + 0.25 * (self.phase_t / 4.0) + 0.05 * random.random()
            perclos_target = min(perclos_target, 0.85)

        # Smooth transitions
        self.ear     = _smooth(self.ear,     ear_target,     alpha=0.18)
        self.mar     = _smooth(self.mar,     mar_target,     alpha=0.15)
        self.perclos = _smooth(self.perclos, perclos_target, alpha=0.12)

        # Random occlusion events (5% chance per step)
        self.occlusion = random.random() < 0.05

        return {
            "ear":       round(self.ear,     4),
            "mar":       round(self.mar,     4),
            "perclos":   round(self.perclos, 4),
            "phase":     self.phase,
            "phase_t":   self.phase_t,
            "occlusion": self.occlusion,
        }


# ─────────────────────────────────────────────
# PATCH / FRAME IMAGE GENERATORS
# ─────────────────────────────────────────────

def _make_patch(label: str, ear_or_mar: float, is_drowsy: bool, size=80) -> Image.Image:
    """Generate a synthetic 80x80 patch image with visual cues."""
    bg_color = (30, 20, 20) if is_drowsy else (20, 30, 20)
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2

    if label in ("Left Eye", "Right Eye"):
        # Draw an eye shape — opening proportional to EAR
        eye_h = max(2, int(ear_or_mar * size * 1.2))
        eye_w = int(size * 0.7)
        draw.ellipse(
            [cx - eye_w//2, cy - eye_h//2, cx + eye_w//2, cy + eye_h//2],
            outline=(100, 220, 100) if not is_drowsy else (220, 80, 80),
            width=2
        )
        # Pupil
        draw.ellipse([cx-4, cy-4, cx+4, cy+4],
                     fill=(180, 180, 220))
        # Iris ring
        draw.ellipse([cx-8, cy-8, cx+8, cy+8],
                     outline=(100, 150, 255), width=1)
    else:
        # Mouth — opening proportional to MAR
        mouth_h = max(2, int(ear_or_mar * size * 0.8))
        mouth_w = int(size * 0.6)
        draw.ellipse(
            [cx - mouth_w//2, cy - mouth_h//2, cx + mouth_w//2, cy + mouth_h//2],
            outline=(220, 180, 80) if not is_drowsy else (220, 100, 60),
            width=2
        )
        if mouth_h > 8:
            # Show teeth when yawning
            draw.rectangle([cx - mouth_w//3, cy - mouth_h//4,
                             cx + mouth_w//3, cy + mouth_h//4],
                            fill=(230, 230, 230))

    # Mesh dots overlay (simulate MediaPipe landmarks)
    for _ in range(8):
        px = random.randint(10, size - 10)
        py = random.randint(10, size - 10)
        draw.ellipse([px-1, py-1, px+1, py+1],
                     fill=(0, 200, 255))
    return img


def _make_face_frame(phase: str, t: float, occlusion: bool, size=(320, 240)) -> Image.Image:
    """Generate a synthetic driver face frame with mesh overlay."""
    w, h = size
    bg   = (20, 20, 30) if phase == "ALERT" else (30, 15, 15)
    img  = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)

    cx, cy = w // 2, h // 2

    # Head oval
    draw.ellipse([cx-70, cy-90, cx+70, cy+90], outline=(80, 120, 80), width=2)

    # Eyes — size depends on EAR proxy
    eye_open = 0.28 if phase == "ALERT" else 0.15
    ew = int(25 + eye_open * 40)
    eh = max(3, int(eye_open * 50))
    # Left eye
    draw.ellipse([cx-55-ew//2, cy-20-eh//2, cx-55+ew//2, cy-20+eh//2],
                 outline=(100, 200, 100) if phase == "ALERT" else (200, 80, 80), width=2)
    draw.ellipse([cx-55-4, cy-24, cx-55+4, cy-16], fill=(180, 180, 220))
    # Right eye
    draw.ellipse([cx+55-ew//2, cy-20-eh//2, cx+55+ew//2, cy-20+eh//2],
                 outline=(100, 200, 100) if phase == "ALERT" else (200, 80, 80), width=2)
    draw.ellipse([cx+55-4, cy-24, cx+55+4, cy-16], fill=(180, 180, 220))

    # Nose
    draw.polygon([(cx, cy-5), (cx-8, cy+15), (cx+8, cy+15)],
                 outline=(100, 100, 120))

    # Mouth
    if phase == "DROWSY" and math.sin(t * 1.5) > 0.3:
        # Yawning
        draw.ellipse([cx-20, cy+30, cx+20, cy+55], fill=(60, 20, 20), outline=(200, 150, 80), width=2)
    else:
        draw.arc([cx-20, cy+30, cx+20, cy+45], start=0, end=180,
                 fill=(150, 120, 100), width=2)

    # Mesh landmark dots
    landmarks = [
        (cx-55, cy-20), (cx+55, cy-20),  # eye centers
        (cx, cy-10), (cx, cy+5),          # nose bridge
        (cx-15, cy+38), (cx+15, cy+38),   # mouth corners
        (cx-70, cy), (cx+70, cy),          # face edges
        (cx, cy-90), (cx, cy+90),          # top/bottom
        (cx-40, cy-60), (cx+40, cy-60),
    ]
    for lx, ly in landmarks:
        draw.ellipse([lx-2, ly-2, lx+2, ly+2], fill=(0, 200, 255))
        # Connection lines (simplified mesh)
        draw.line([(lx, ly), (cx, cy)], fill=(0, 100, 150), width=1)

    # Occlusion overlay
    if occlusion:
        overlay = Image.new("RGBA", size, (200, 50, 50, 60))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "! OCCLUSION", fill=(255, 80, 80))

    # Phase label
    color = (60, 220, 100) if phase == "ALERT" else (220, 60, 60)
    draw.rectangle([0, h-24, w, h], fill=(0, 0, 0, 180))
    draw.text((w//2 - 25, h-20), phase, fill=color)

    return img


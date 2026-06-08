"""
export_engine_with_pv_sync.py
─────────────────────────────
Combines the Stirling engine animation (left) with the P-V diagram
animation (right) into a single synchronized MP4.

Strategy:
  • Engine frames  – extracted from the existing engine GIF
                     (engine_video/stirling_engine_animation.gif, 60 frames)
  • P-V frames     – rendered fresh from physics_v10 simulation data
  • Compositing    – PIL side-by-side per frame, piped to ffmpeg

Phase alignment (both panels share the same mapping):
  frame i  →  θ = 360° · i / N_FRAMES
  frame 0  → θ =   0°
  frame 15 → θ =  90°
  frame 30 → θ = 180°
  frame 45 → θ = 270°

Outputs:
  engine_with_pv_sync.mp4
  engine_with_pv_sync_60sec.mp4  (60-second loop)
"""

import io
import math
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from physics_v10 import PROTOTYPE, GASES, to_si, build_geometry, simulate

# ── Constants ────────────────────────────────────────────────────────────────
N_FRAMES   = 60
FPS        = 20
GIF_PATH   = "engine_video/stirling_engine_animation.gif"
OUT_MP4    = "engine_with_pv_sync.mp4"
OUT_60SEC  = "engine_with_pv_sync_60sec.mp4"

PV_WIDTH_PX  = 700   # P-V panel width  (pixels at 100 dpi)
PV_HEIGHT_PX = 560   # P-V panel height

# ── Simulation ────────────────────────────────────────────────────────────────
params = dict(PROTOTYPE)
params['eps_reg'] = 0.85

losses_flags = dict(
    flow=True, regen_imp=True, mechanical=True,
    wall_cond=True, leakage=True, shuttle=True,
)

print("Running simulation …")
sim = simulate(params, model='schmidt', losses_flags=losses_flags)
if sim is None:
    sys.exit("Simulation failed.")

R    = sim['result']
geom = sim['geom']

theta = R['theta']   # radians, shape (N_SIM,)
P     = R['P']       # Pa
V_e   = R['V_e']     # m³
V_c   = R['V_c']     # m³

N_SIM = len(theta)

# Volume — same definition as the app P-V plot
dead_vol_cm3 = (geom.get('V_k', 0) + geom.get('V_r', 0) + geom.get('V_h', 0)) * 1e6
V_tot_cm3    = (V_e + V_c) * 1e6 + dead_vol_cm3
P_bar        = P / 1e5

# Map each animation frame → nearest simulation index
frame_to_sim = [round(i * N_SIM / N_FRAMES) % N_SIM for i in range(N_FRAMES)]

# Stage points at 0°, 90°, 180°, 270°
stage_angles_deg = [0, 90, 180, 270]
stage_labels     = ["1", "2", "3", "4"]
stage_colors     = ['#1565C0', '#2E7D32', '#C62828', '#E65100']
stage_sim_idx    = [round(a / 360 * N_SIM) % N_SIM for a in stage_angles_deg]
stage_V          = [V_tot_cm3[i] for i in stage_sim_idx]
stage_P          = [P_bar[i]     for i in stage_sim_idx]

# ── Load engine GIF frames ────────────────────────────────────────────────────
print(f"Loading engine GIF: {GIF_PATH}")
gif = Image.open(GIF_PATH)
if gif.n_frames != N_FRAMES:
    sys.exit(f"GIF has {gif.n_frames} frames, expected {N_FRAMES}.")

engine_frames = []
for i in range(N_FRAMES):
    gif.seek(i)
    engine_frames.append(gif.convert("RGB"))

ENG_W, ENG_H = engine_frames[0].size
print(f"Engine frame size: {ENG_W}×{ENG_H}")

# ── Build static P-V figure (axes, loop, stage markers) ──────────────────────
DPI = 100
fig_pv, ax_pv = plt.subplots(figsize=(PV_WIDTH_PX / DPI, PV_HEIGHT_PX / DPI), dpi=DPI)
fig_pv.patch.set_facecolor('white')
ax_pv.set_facecolor('white')

ax_pv.plot(V_tot_cm3, P_bar, color='#1565C0', lw=2.0, zorder=2)
ax_pv.fill(V_tot_cm3, P_bar, alpha=0.08, color='#1565C0', zorder=1)

for V_s, P_s, lbl, col in zip(stage_V, stage_P, stage_labels, stage_colors):
    ax_pv.scatter(V_s, P_s, s=90, color=col, zorder=5,
                  edgecolors='white', linewidths=1.0)
    ax_pv.annotate(
        lbl,
        xy=(V_s, P_s),
        xytext=(6, 6),
        textcoords='offset points',
        fontsize=10,
        fontweight='bold',
        color=col,
        zorder=6,
    )

ax_pv.set_xlabel("Volume",   fontsize=12)
ax_pv.set_ylabel("Pressure", fontsize=12)
ax_pv.tick_params(labelbottom=False, labelleft=False)
ax_pv.grid(alpha=0.25, linestyle='--')
for spine in ax_pv.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color('#AAAAAA')

# Moving dot
dot, = ax_pv.plot([], [], 'o', color='#FF6F00', markersize=11, zorder=7,
                  markeredgecolor='white', markeredgewidth=1.5)

plt.tight_layout(pad=1.4)

# ── Helper: render P-V frame i as PIL Image ───────────────────────────────────
def render_pv_frame(i: int) -> Image.Image:
    idx = frame_to_sim[i]
    dot.set_data([V_tot_cm3[idx]], [P_bar[idx]])
    buf = io.BytesIO()
    fig_pv.savefig(buf, format='png', dpi=DPI, bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf).convert("RGB")

# Render a test frame to get exact P-V output size
_test = render_pv_frame(0)
PV_W, PV_H = _test.size
print(f"P-V frame size:    {PV_W}×{PV_H}")

# ── Compositing: choose canvas height ────────────────────────────────────────
# Match heights; scale engine to PV height if different.
CANVAS_H = PV_H if PV_H % 2 == 0 else PV_H + 1  # must be even for libx264
scale     = CANVAS_H / ENG_H
ENG_W_sc  = round(ENG_W * scale)
ENG_W_sc  = ENG_W_sc if ENG_W_sc % 2 == 0 else ENG_W_sc + 1
PV_W      = PV_W if PV_W % 2 == 0 else PV_W + 1
CANVAS_W  = ENG_W_sc + PV_W

print(f"Canvas size:       {CANVAS_W}×{CANVAS_H}")

# ── ffmpeg pipe ───────────────────────────────────────────────────────────────
def open_ffmpeg_pipe(output_path: str):
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{CANVAS_W}x{CANVAS_H}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "slow",
        "-crf", "18",
        "-movflags", "+faststart",
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

# ── Render & write frames ─────────────────────────────────────────────────────
print(f"\nRendering {N_FRAMES} frames …")
proc = open_ffmpeg_pipe(OUT_MP4)

frames_raw = []   # keep for the 60-sec loop

for i in range(N_FRAMES):
    # Engine frame (scale to canvas height)
    eng_img = engine_frames[i].resize((ENG_W_sc, CANVAS_H), Image.LANCZOS)

    # P-V frame
    pv_img  = render_pv_frame(i)
    if pv_img.size != (PV_W, CANVAS_H):
        pv_img = pv_img.resize((PV_W, CANVAS_H), Image.LANCZOS)

    # Composite side by side on white background
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (255, 255, 255))
    canvas.paste(eng_img, (0, 0))
    canvas.paste(pv_img,  (ENG_W_sc, 0))

    raw = canvas.tobytes()
    frames_raw.append(raw)
    proc.stdin.write(raw)

    if (i + 1) % 10 == 0:
        print(f"  frame {i+1}/{N_FRAMES}")

proc.stdin.close()
rc = proc.wait()
if rc != 0:
    sys.exit(f"ffmpeg failed (exit {rc}) writing {OUT_MP4}")
print(f"Saved: {OUT_MP4}")

# ── 60-second looped version ──────────────────────────────────────────────────
TOTAL_FRAMES_60 = FPS * 60   # 1200
print(f"\nWriting 60-second loop ({TOTAL_FRAMES_60} frames) …")
proc2 = open_ffmpeg_pipe(OUT_60SEC)

for j in range(TOTAL_FRAMES_60):
    proc2.stdin.write(frames_raw[j % N_FRAMES])

proc2.stdin.close()
rc2 = proc2.wait()
if rc2 != 0:
    sys.exit(f"ffmpeg failed (exit {rc2}) writing {OUT_60SEC}")
print(f"Saved: {OUT_60SEC}")

plt.close(fig_pv)

# ── Sync verification ─────────────────────────────────────────────────────────
print("\nPhase sync check:")
for chk in [0, N_FRAMES//4, N_FRAMES//2, 3*N_FRAMES//4]:
    deg = math.degrees(theta[frame_to_sim[chk]]) % 360
    print(f"  frame {chk:2d}  →  θ = {deg:.1f}°")

print("\nDone.")

"""
export_pv_animation.py — P-V diagram animation synchronized with engine cycle.

Frame i → θ = 2π·i/N_FRAMES
  frame 0       → θ = 0°
  frame N/4     → θ = 90°
  frame N/2     → θ = 180°
  frame 3N/4    → θ = 270°

Output: pv_diagram_animation.mp4 (H.264, yuv420p)
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from physics_v10 import PROTOTYPE, GASES, to_si, build_geometry, simulate

# ── Simulation ────────────────────────────────────────────────────────────────
params = dict(PROTOTYPE)
params['eps_reg'] = 0.85

losses_flags = dict(
    flow=True, regen_imp=True, mechanical=True,
    wall_cond=True, leakage=True, shuttle=True
)

sim = simulate(params, model='schmidt', losses_flags=losses_flags)
if sim is None:
    raise RuntimeError("Simulation failed with default prototype parameters.")

R    = sim['result']
geom = sim['geom']

theta = R['theta']          # radians, length N_SIM
P     = R['P']              # Pa
V_e   = R['V_e']            # m³
V_c   = R['V_c']            # m³

# Dead volume — matches app _dead_vol()
dead_vol_cm3 = (geom.get('V_k', 0) + geom.get('V_r', 0) + geom.get('V_h', 0)) * 1e6
V_tot_cm3    = (V_e + V_c) * 1e6 + dead_vol_cm3   # cm³  (matches app P-V plot)
P_bar        = P / 1e5                              # bar  (for display — unlabelled)

N_SIM = len(theta)   # typically 360

# ── Animation parameters — match animation_v10.py ────────────────────────────
N_FRAMES = 60
FPS      = 20

# Map each animation frame to the nearest simulation index
frame_to_sim = [round(i * N_SIM / N_FRAMES) % N_SIM for i in range(N_FRAMES)]

# ── Stage points: θ = 0°, 90°, 180°, 270° ────────────────────────────────────
stage_angles_deg = [0, 90, 180, 270]
stage_labels     = ["1", "2", "3", "4"]
stage_colors     = ['#1565C0', '#2E7D32', '#C62828', '#E65100']

stage_sim_idx = [round(a / 360 * N_SIM) % N_SIM for a in stage_angles_deg]
stage_V       = [V_tot_cm3[i] for i in stage_sim_idx]
stage_P       = [P_bar[i]     for i in stage_sim_idx]

# ── Figure setup ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Full P-V loop
ax.plot(V_tot_cm3, P_bar, color='#1565C0', lw=2.0, zorder=2)
ax.fill(V_tot_cm3, P_bar, alpha=0.08, color='#1565C0', zorder=1)

# Stage markers
for V_s, P_s, lbl, col in zip(stage_V, stage_P, stage_labels, stage_colors):
    ax.scatter(V_s, P_s, s=90, color=col, zorder=5, edgecolors='white', linewidths=1.0)
    ax.annotate(
        lbl,
        xy=(V_s, P_s),
        xytext=(6, 6),
        textcoords='offset points',
        fontsize=10,
        fontweight='bold',
        color=col,
        zorder=6,
    )

ax.set_xlabel("Volume", fontsize=12)
ax.set_ylabel("Pressure", fontsize=12)
ax.tick_params(labelbottom=False, labelleft=False)
ax.grid(alpha=0.25, linestyle='--')

for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color('#AAAAAA')

# Moving dot
dot, = ax.plot([], [], 'o', color='#FF6F00', markersize=11, zorder=7,
               markeredgecolor='white', markeredgewidth=1.5)

# Theta label (top-right corner)
theta_text = ax.text(
    0.97, 0.96, '',
    transform=ax.transAxes,
    ha='right', va='top',
    fontsize=10, color='#444444',
    fontfamily='monospace',
)

plt.tight_layout(pad=1.4)

# ── Animation functions ───────────────────────────────────────────────────────
def init():
    dot.set_data([], [])
    theta_text.set_text('')
    return dot, theta_text


def update(frame):
    idx = frame_to_sim[frame]
    dot.set_data([V_tot_cm3[idx]], [P_bar[idx]])
    deg = math.degrees(theta[idx]) % 360
    theta_text.set_text(f"θ = {deg:.0f}°")
    return dot, theta_text


ani = animation.FuncAnimation(
    fig, update, frames=N_FRAMES,
    init_func=init, blit=True, interval=1000 / FPS
)

# ── Export MP4 ────────────────────────────────────────────────────────────────
output_path = 'pv_diagram_animation.mp4'

writer = animation.FFMpegWriter(
    fps=FPS,
    codec='libx264',
    extra_args=['-pix_fmt', 'yuv420p', '-preset', 'slow', '-crf', '18'],
)

ani.save(output_path, writer=writer, dpi=150)
plt.close(fig)

print(f"Saved: {output_path}")
print(f"  Frames : {N_FRAMES}  FPS : {FPS}  Duration : {N_FRAMES/FPS:.1f}s")
print(f"  θ sync : frame 0 → {math.degrees(theta[frame_to_sim[0]]):.0f}°, "
      f"frame {N_FRAMES//4} → {math.degrees(theta[frame_to_sim[N_FRAMES//4]]):.0f}°, "
      f"frame {N_FRAMES//2} → {math.degrees(theta[frame_to_sim[N_FRAMES//2]]):.0f}°, "
      f"frame {3*N_FRAMES//4} → {math.degrees(theta[frame_to_sim[3*N_FRAMES//4]]):.0f}°")

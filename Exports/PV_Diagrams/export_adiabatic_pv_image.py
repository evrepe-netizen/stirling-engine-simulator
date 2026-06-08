import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try v10 imports first, fallback to v9_4
try:
    from physics_v10 import PROTOTYPE, simulate
    print("Using physics_v10.py")
except ImportError:
    from physics_v9_4 import PROTOTYPE, simulate
    print("Using physics_v9_4.py")

params = dict(PROTOTYPE)

losses_flags = dict(
    flow=True,
    regen_imp=True,
    mechanical=True,
    wall_cond=True,
    leakage=True,
    shuttle=False
)

print("Running adiabatic simulation...")
sim = simulate(params, model="adiabatic", losses_flags=losses_flags)

if sim is None:
    raise RuntimeError("Adiabatic simulation failed.")

R = sim["result"]
L = sim["losses"]
geom = sim["geom"]

# Use full gas volume: variable volumes + constant dead volumes
V_total = (
    R["V_e"] + R["V_c"] +
    geom.get("V_k", 0) +
    geom.get("V_r", 0) +
    geom.get("V_h", 0)
) * 1e6  # cm^3

P_bar = R["P"] / 1e5

out_dir = "pv_outputs"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "adiabatic_pv_diagram.png")

fig, ax = plt.subplots(figsize=(10, 7), dpi=220)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.plot(V_total, P_bar, linewidth=3)
ax.fill(V_total, P_bar, alpha=0.10)

# Mark four phase points
n = len(V_total)
phase_indices = [0, n//4, n//2, 3*n//4]
phase_labels = ["1", "2", "3", "4"]

for idx, label in zip(phase_indices, phase_labels):
    ax.scatter(V_total[idx], P_bar[idx], s=120, zorder=5)
    ax.text(
        V_total[idx], P_bar[idx],
        f"  {label}",
        fontsize=16,
        fontweight="bold",
        va="center"
    )

ax.set_title("Adiabatic P-V Diagram", fontsize=20, fontweight="bold")
ax.set_xlabel("Volume", fontsize=16)
ax.set_ylabel("Pressure", fontsize=16)
ax.grid(True, alpha=0.25)

# Small info box
info = (
    f"W_cycle = {L['W_cycle']:.3f} J\n"
    f"P_mean = {L['P_mean']/1e5:.3f} bar\n"
    f"η_brake = {L['eta_brake']*100:.2f}%"
)
ax.text(
    0.98, 0.04, info,
    transform=ax.transAxes,
    ha="right", va="bottom",
    fontsize=11,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

plt.tight_layout()
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)

print("Saved:", os.path.abspath(out_path))

"""
animation_v10.py — Adiabatic Gamma Stirling Cycle — qualitative visualization
v10: gauge fix — three gauges stacked vertically with no overlap.
     Figure bottom extended to fit T_c gauge.
"""

import math, base64, tempfile, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

try:
    import streamlit as st
    _CACHE = st.cache_data
except Exception:
    def _CACHE(fn):
        return fn

_CMAP = LinearSegmentedColormap.from_list('temp',
    ['#4FC3F7', '#00BCD4', '#66BB6A', '#FFEE58', '#FFA726', '#EF5350'])

def _tc(T, Tk, Th):
    return _CMAP(float(np.clip((T - Tk) / max(Th - Tk, 1.0), 0.0, 1.0)))


@_CACHE
def build_engine_animation(geom_frozen, params_frozen):
    """
    Returns base64-encoded animated GIF.
    Accepts frozen (tuple-of-pairs) representations of geom and params dicts
    so that @st.cache_data can hash the inputs.
    """
    geom   = dict(geom_frozen)
    params = dict(params_frozen)

    N_FRAMES = 60
    FPS      = 20

    D_d = geom.get('D_displacer', params.get('D_displacer', 75.0)    * 1e-3)
    S_d = geom.get('S_displacer', params.get('S_displacer', 101.5)   * 1e-3)
    L_d = float(params.get('L_displacer', 0.235))
    D_p = geom.get('D_power',     params.get('D_power',    65.6)     * 1e-3)
    S_p = geom.get('S_power',     params.get('S_power',    61.6)     * 1e-3)
    L_r = geom.get('L_r',         params.get('L_r',        236.0)    * 1e-3)
    D_r = geom.get('D_r',         params.get('D_r',         40.0)    * 1e-3)
    phi = geom.get('phi',         math.radians(params.get('phi_deg', 90.0)))
    T_h = float(params.get('T_h', 873.0))
    T_k = float(params.get('T_k', 300.0))
    P_m = float(params.get('P_mean_bar', 1.0))

    def n(x): return x / D_d

    R_D = 0.5
    R_P = n(D_p) / 2
    R_R = n(D_r) / 2
    sD  = n(S_d)
    sP  = n(S_p)
    lR  = n(L_r)
    lD  = n(L_d)
    W   = 0.07

    cylD_len = lD + sD + 0.30

    yD  = 0.0
    gap = 0.60
    yR  = yD + R_D + W + gap + R_R + W
    yP  = yD - R_D - W - gap - R_P - W

    xD0 = 0.0
    xD1 = xD0 + cylD_len

    xR0 = xD0 + (cylD_len - lR) / 2.0
    xR1 = xR0 + lR

    xCold = xD1 + W

    xP0 = xD1 - sP * 0.3
    xP1 = xP0 + sP + 0.40

    xFW = max(xD1, xP1) + 1.10
    yFW = (yD + yP) / 2.0
    Rfw = max(sD, sP) * 0.28 + 0.18

    xD_cross = xFW - Rfw - 0.20
    xP_cross = xFW - Rfw - 0.20

    xG  = xFW + Rfw + 0.55
    yG  = yR - 0.05   # kept for legacy ref; gauges use g_y below

    # ── Gauge layout constants (computed once for figure bounds) ──────────────
    _gauge_g_y        = yR - 0.10       # pressure gauge centre y
    _gauge_gr         = 0.30            # pressure dial radius
    _gauge_bh         = 0.70            # thermometer bar height
    _gauge_gap        = 0.25            # vertical gap between gauges
    _g2_bot = _gauge_g_y - _gauge_gr - 0.17 - _gauge_gap - _gauge_bh
    _g3_bot = _g2_bot    - _gauge_gap  - _gauge_bh

    fig_x0 = xD0 - 0.80
    fig_x1 = xG  + 0.75
    fig_y0 = min(yP - R_P - W - 0.40, _g3_bot - 0.30)   # extended for T_c gauge
    fig_y1 = yR  + R_R + W + 0.60

    fig, ax = plt.subplots(figsize=(15, 7), dpi=85)
    ax.set_xlim(fig_x0, fig_x1)
    ax.set_ylim(fig_y0, fig_y1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    CW   = '#888888'
    CW2  = '#AAAAAA'
    CROD = '#999999'
    CFW  = '#555555'

    artists = []

    def rect(x, y, w, h, fc, ec=CW, lw=1.2, alpha=1.0, z=3):
        p = mpatches.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z)
        ax.add_patch(p); artists.append(p)

    def circ(cx, cy, r, fc, ec='none', lw=0.8, alpha=1.0, z=4):
        p = plt.Circle((cx, cy), r, fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z)
        ax.add_patch(p); artists.append(p)

    def line(x0, y0, x1, y1, c=CW, lw=1.5, z=4):
        ln, = ax.plot([x0, x1], [y0, y1], color=c, lw=lw, zorder=z,
                      solid_capstyle='round')
        artists.append(ln)

    def txt(x, y, s, color='#333333', fs=6.5, ha='center', va='center',
            bold=False, z=10):
        t = ax.text(x, y, s, color=color, fontsize=fs, ha=ha, va=va,
                    fontweight='bold' if bold else 'normal', zorder=z)
        artists.append(t)

    def _bezier(t_arr, p0, pc, p1):
        x = (1-t_arr)**2*p0[0] + 2*(1-t_arr)*t_arr*pc[0] + t_arr**2*p1[0]
        y = (1-t_arr)**2*p0[1] + 2*(1-t_arr)*t_arr*pc[1] + t_arr**2*p1[1]
        return x, y

    _t80 = np.linspace(0.0, 1.0, 80)

    _ps1 = (xD0 - W, yD)
    _pe1 = (xR0 - W, yR)
    _pc1 = (min(_ps1[0], _pe1[0]) - 0.55, (_ps1[1] + _pe1[1]) / 2.0)
    _bx1, _by1 = _bezier(_t80, _ps1, _pc1, _pe1)

    _ps2 = (xD1 + W, yD)
    _pe2 = (xR1 + W, yR)
    _pc2 = (max(_ps2[0], _pe2[0]) + 0.55, (_ps2[1] + _pe2[1]) / 2.0)
    _bx2, _by2 = _bezier(_t80, _ps2, _pc2, _pe2)

    def _bl(p0, pc, p1, n=200):
        t = np.linspace(0, 1, n)
        x, y = _bezier(t, p0, pc, p1)
        return float(np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)))

    L_hot   = sD * 0.45
    L_pipe1 = _bl(_ps1, _pc1, _pe1)
    L_regen = lR
    L_pipe2 = _bl(_ps2, _pc2, _pe2)
    L_cold  = sD * 0.45
    L_total = L_hot + L_pipe1 + L_regen + L_pipe2 + L_cold

    f1 = L_hot / L_total
    f2 = (L_hot + L_pipe1) / L_total
    f3 = (L_hot + L_pipe1 + L_regen) / L_total
    f4 = (L_hot + L_pipe1 + L_regen + L_pipe2) / L_total

    def _path_xy(s, dx_now):
        s = float(np.clip(s, 0.0, 1.0))
        if s < f1:
            f = s / f1
            x_span = max(0.06, dx_now - xD0 - 0.10)
            x = xD0 + 0.05 + f * x_span; y = yD; T_base = T_h
        elif s < f2:
            f = (s - f1) / (f2 - f1); t = np.array([f])
            x = float(_bezier(t, _ps1, _pc1, _pe1)[0][0])
            y = float(_bezier(t, _ps1, _pc1, _pe1)[1][0]); T_base = T_h
        elif s < f3:
            f = (s - f2) / (f3 - f2)
            x = xR0 + f * lR; y = yR
            T_base = T_h - (T_h - T_k) * f
        elif s < f4:
            f = (s - f3) / (f4 - f3); t = np.array([1.0 - f])
            x = float(_bezier(t, _ps2, _pc2, _pe2)[0][0])
            y = float(_bezier(t, _ps2, _pc2, _pe2)[1][0]); T_base = T_k
        else:
            f = (s - f4) / max(1e-9, 1.0 - f4)
            cold_right_face = dx_now + lD + 0.08
            x_span = max(0.06, xD1 - 0.10 - cold_right_face)
            x = xD1 - 0.05 - f * x_span; y = yD; T_base = T_k
        return x, y, T_base

    # Pressure cycle pre-computation
    V_swe = math.pi * (D_d/2)**2 * S_d
    V_swc = math.pi * (D_p/2)**2 * S_p
    Vrl   = geom.get('V_r_lumped', 4.0e-4)
    Tr    = (T_h - T_k) / math.log(max(T_h/T_k, 1.001))
    Ve0   = geom.get('V_cle', 1.77e-5) + V_swe/2
    Vc0   = geom.get('V_clc', 7.2e-5)  + V_swc/2
    Sig0  = Vc0/T_k + Vrl/Tr + Ve0/T_h

    _th_pre = np.linspace(0, 2*math.pi, 120, endpoint=False)
    _P_pre  = []
    for _th in _th_pre:
        _Ve = geom.get('V_cle', 1.77e-5) + (V_swe/2)*(1 + math.cos(_th))
        _Vc = (geom.get('V_clc', 7.2e-5)
               + (V_swc/2)*(1 + math.cos(_th - phi))
               + (V_swe/2)*(1 - math.cos(_th)))
        _Sig = _Vc/T_k + Vrl/Tr + _Ve/T_h
        _P_pre.append(P_m * Sig0 / _Sig)
    P_min_c = min(_P_pre)
    P_max_c = max(_P_pre)

    N_MAIN  = 34
    N_POWER = 12

    rng = np.random.default_rng(42)
    s0_main       = np.linspace(0.0, 1.0, N_MAIN, endpoint=False)
    jit_phase     = rng.uniform(0, 2*math.pi, N_MAIN)
    jit_amp_frac  = rng.uniform(0.3, 0.85, N_MAIN)

    s0_power      = np.linspace(0.0, 1.0, N_POWER, endpoint=False)
    jit_phase_p   = rng.uniform(0, 2*math.pi, N_POWER)

    A_flow = (L_regen / L_total) * 0.85 + 0.05

    def _dx(th):
        return xD0 + 0.10 + (sD/2) * (1.0 - math.cos(th))

    def _px(th):
        return xP0 + 0.08 + (sP/2) * (1.0 - math.cos(th - phi))

    def draw_static():
        # Regenerator
        ax.add_patch(mpatches.Rectangle((xR0, yR+R_R), lR, W, fc=CW, ec='none', zorder=2))
        ax.add_patch(mpatches.Rectangle((xR0, yR-R_R-W), lR, W, fc=CW, ec='none', zorder=2))
        ax.add_patch(mpatches.Rectangle((xR0-W, yR-R_R-W), W, 2*R_R+2*W, fc=CW, ec='none', zorder=2))
        ax.add_patch(mpatches.Rectangle((xR1, yR-R_R-W), W, 2*R_R+2*W, fc=CW, ec='none', zorder=2))
        ax.add_patch(mpatches.Rectangle((xR0, yR-R_R), lR, 2*R_R, fc='#E8E8E8', ec=CW, lw=1.0, zorder=2))
        for k in range(1, 12):
            xm = xR0 + lR * k / 12.0
            ax.plot([xm, xm], [yR-R_R, yR+R_R], color='#BBBBBB', lw=0.7, alpha=0.7, zorder=3)
        ax.text((xR0+xR1)/2, yR+R_R+W+0.14, 'REGENERATOR',
                color='#555', fontsize=7, ha='center', fontweight='bold', zorder=3)
        ax.text(xR0+0.06, yR+R_R+W+0.14, f'HOT {T_h:.0f}K',
                color='#C62828', fontsize=6, ha='left', fontweight='bold', zorder=3)
        ax.text(xR1-0.06, yR+R_R+W+0.14, f'COLD {T_k:.0f}K',
                color='#1565C0', fontsize=6, ha='right', fontweight='bold', zorder=3)
        # Displacer cylinder walls
        ax.add_patch(mpatches.Rectangle((xD0, yD+R_D), cylD_len, W, fc=CW, ec='none', zorder=2))
        ax.add_patch(mpatches.Rectangle((xD0, yD-R_D-W), cylD_len, W, fc=CW, ec='none', zorder=2))
        ax.add_patch(mpatches.Rectangle((xD0-W, yD-R_D-W), W, 2*R_D+2*W, fc=CW, ec='none', zorder=2))
        ax.add_patch(mpatches.Rectangle((xD1, yD-R_D-W), W, 2*R_D+2*W, fc=CW, ec='none', zorder=2))
        ax.add_patch(mpatches.Rectangle((xD0, yD-R_D), cylD_len*0.5, 2*R_D,
            fc='#FFE8E8', ec='none', alpha=0.55, zorder=1))
        ax.add_patch(mpatches.Rectangle((xD0+cylD_len*0.5, yD-R_D), cylD_len*0.5, 2*R_D,
            fc='#E8EEFF', ec='none', alpha=0.55, zorder=1))
        ax.text(xD0+cylD_len/2, yD-R_D-W-0.17, 'DISPLACER CYLINDER',
                color='#444', fontsize=7, ha='center', fontweight='bold', zorder=3)
        ax.text(xD0+0.10, yD+R_D+W+0.13, f'HOT  {T_h:.0f} K',
                color='#C62828', fontsize=6.5, ha='left', fontweight='bold', zorder=3)
        ax.text(xD1-0.10, yD+R_D+W+0.13, f'COLD  {T_k:.0f} K',
                color='#1565C0', fontsize=6.5, ha='right', fontweight='bold', zorder=3)
        ax.plot(_bx1, _by1, color=CW,  lw=9, solid_capstyle='round', zorder=2)
        ax.plot(_bx1, _by1, color=CW2, lw=5, solid_capstyle='round', zorder=2)
        ax.plot(_bx2, _by2, color=CW,  lw=9, solid_capstyle='round', zorder=2)
        ax.plot(_bx2, _by2, color=CW2, lw=5, solid_capstyle='round', zorder=2)
        pW3 = 0.09
        conn_x = xCold - pW3 * 0.5
        ax.add_patch(mpatches.FancyBboxPatch(
            (conn_x - pW3, yP+R_P+W), 2*pW3, yD-R_D-W - (yP+R_P+W),
            boxstyle='round,pad=0.01', fc=CW2, ec=CW, lw=1.5, zorder=2))
        ax.add_patch(plt.Circle((conn_x, yD-R_D-W*0.5), pW3*1.1, fc=CW, ec='none', zorder=4))
        ax.text(conn_x + 0.18, (yD-R_D + yP+R_P)*0.5,
                'cold-space\npassage', color='#666', fontsize=5.5,
                ha='left', va='center', zorder=3)
        ax.add_patch(mpatches.Rectangle((xP0, yP+R_P), xP1-xP0, W, fc=CW, ec='none', zorder=2))
        ax.add_patch(mpatches.Rectangle((xP0, yP-R_P-W), xP1-xP0, W, fc=CW, ec='none', zorder=2))
        ax.add_patch(mpatches.Rectangle((xP0-W, yP-R_P-W), W, 2*R_P+2*W, fc=CW, ec='none', zorder=2))
        ax.add_patch(mpatches.Rectangle((xP1, yP-R_P-W), W, 2*R_P+2*W, fc=CW, ec='none', zorder=2))
        ax.text((xP0+xP1)/2, yP-R_P-W-0.17, 'POWER CYLINDER',
                color='#444', fontsize=7, ha='center', fontweight='bold', zorder=3)
        ax.text((fig_x0+fig_x1)/2, fig_y1-0.10,
                'Adiabatic Gamma Stirling Cycle — qualitative visualization',
                color='#333', fontsize=8.5, ha='center', fontstyle='italic', zorder=3)
        for sign, yoff, col in [(+1, R_R*0.4, '#C62828'), (-1, -R_R*0.4, '#1565C0')]:
            xa = xR0 + lR * 0.5
            ax.annotate('', xy=(xa + sign*lR*0.18, yR+yoff),
                        xytext=(xa - sign*lR*0.18, yR+yoff),
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.2), zorder=3)

    draw_static()

    def frame(i):
        for a in artists: a.remove()
        artists.clear()

        theta = 2.0 * math.pi * i / N_FRAMES
        dx    = _dx(theta)
        pw_x  = _px(theta)

        Ve    = geom.get('V_cle', 1.77e-5) + (V_swe/2)*(1 + math.cos(theta))
        Vc    = (geom.get('V_clc', 7.2e-5)
                 + (V_swc/2)*(1 + math.cos(theta - phi))
                 + (V_swe/2)*(1 - math.cos(theta)))
        Sig   = Vc/T_k + Vrl/Tr + Ve/T_h
        P_now = P_m * Sig0 / Sig
        P_rel = float(np.clip((P_now - P_min_c) / max(P_max_c - P_min_c, 0.01), 0, 1))

        dT_adi = (P_rel - 0.5) * (T_h - T_k) * 0.28

        # Dynamic gas zone tints
        hw = max(0.01, dx - xD0 - 0.12)
        rect(xD0+0.02, yD-R_D+0.02, hw, 2*R_D-0.04, '#FFE4E4', ec='none', alpha=0.70, z=1)
        cw = max(0.01, xD1 - 0.12 - (dx+lD+0.08))
        if cw > 0.01:
            rect(dx+lD+0.08, yD-R_D+0.02, cw, 2*R_D-0.04, '#E4E8FF', ec='none', alpha=0.70, z=1)
        pgw = max(0.01, pw_x - xP0 - 0.08)
        rect(xP0+0.02, yP-R_P+0.02, pgw, 2*R_P-0.04, '#FFFDE7', ec='none', alpha=0.60, z=1)

        # Displacer body + rod
        rect(dx, yD - R_D*0.95, lD, 2*R_D*0.95, '#888888', ec='#666', lw=1.2, z=6)
        line(dx+lD, yD, xD1+W+0.05, yD, c=CROD, lw=3.5, z=5)
        rect(xD1+W*0.1, yD-0.055, 0.065, 0.11, '#AAAAAA', ec=CW, lw=1.0, z=7)
        line(xD1+W+0.07, yD, xD_cross, yD, c=CROD, lw=3.0, z=5)

        # Power piston + rod
        rect(pw_x, yP-R_P*0.93, 0.09, 2*R_P*0.93, '#888888', ec='#666', lw=1.2, z=6)
        line(pw_x+0.09, yP, xP1+W+0.05, yP, c=CROD, lw=3.5, z=5)
        rect(xP1+W*0.1, yP-0.045, 0.065, 0.09, '#AAAAAA', ec=CW, lw=1.0, z=7)
        line(xP1+W+0.07, yP, xP_cross, yP, c=CROD, lw=3.0, z=5)

        # Flywheel
        fw = plt.Circle((xFW, yFW), Rfw, fc='#F0F0F0', ec='#333', lw=2.0, zorder=5)
        ax.add_patch(fw); artists.append(fw)
        for k in range(6):
            ang = theta + k * math.pi / 3.0
            line(xFW + Rfw*0.12*math.cos(ang), yFW + Rfw*0.12*math.sin(ang),
                 xFW + Rfw*0.90*math.cos(ang), yFW + Rfw*0.90*math.sin(ang),
                 c=CFW, lw=1.3, z=6)
        circ(xFW, yFW, Rfw*0.12, CFW, ec='#222', lw=1.5, z=7)

        pA_ang = theta + math.pi / 2.0
        pA_x = xFW + Rfw*0.80*math.cos(pA_ang)
        pA_y = yFW + Rfw*0.80*math.sin(pA_ang)
        circ(pA_x, pA_y, 0.042, '#EF5350', ec='#B71C1C', lw=1.2, z=8)
        line(xD_cross, yD, pA_x, pA_y, c=CROD, lw=2.8, z=5)

        pB_ang = theta + math.pi / 2.0 - phi
        pB_x = xFW + Rfw*0.80*math.cos(pB_ang)
        pB_y = yFW + Rfw*0.80*math.sin(pB_ang)
        circ(pB_x, pB_y, 0.038, '#42A5F5', ec='#1565C0', lw=1.2, z=8)
        line(xP_cross, yP, pB_x, pB_y, c=CROD, lw=2.8, z=5)

        # Gas particles
        flow_disp = A_flow * math.cos(theta)
        for idx in range(N_MAIN):
            s = float(np.clip(s0_main[idx] + flow_disp, 0.0, 1.0))
            x, y, T_base = _path_xy(s, dx)
            if s < f1:
                r_reg = R_D * 0.70
            elif s < f2 or (f3 <= s < f4):
                r_reg = R_R * 0.35
            elif s < f3:
                r_reg = R_R * 0.65
            else:
                r_reg = R_D * 0.70
            jy = jit_amp_frac[idx] * r_reg * math.sin(jit_phase[idx] + theta * 0.6)
            T_vis = float(np.clip(T_base + dT_adi, T_k * 0.85, T_h * 1.1))
            circ(x, y + jy, 0.050, _tc(T_vis, T_k, T_h), ec='#888', lw=0.4, alpha=0.90, z=7)

        vol_back  = xP0 + 0.06
        vol_front = max(vol_back + 0.04, pw_x - 0.02)
        for idx in range(N_POWER):
            x = vol_back + s0_power[idx] * max(0.04, vol_front - vol_back)
            jy = 0.55 * R_P * math.sin(jit_phase_p[idx] + theta * 0.5)
            T_vis = float(np.clip(T_k + dT_adi * 1.3, T_k * 0.85, T_h * 0.9))
            circ(x, yP + jy, 0.045, _tc(T_vis, T_k, T_h), ec='#888', lw=0.4, alpha=0.88, z=7)

        # ── Temperature values for gauges ────────────────────────────────────
        T_e_now = T_h + (T_h - T_k) * 0.07 * math.sin(theta)
        T_c_now = T_k + (T_h - T_k) * 0.05 * math.sin(theta - phi + math.pi)

        # ── Gauge panel — three gauges stacked vertically, no overlap ─────────

        # Gauge 1: Pressure dial — at top of panel, near regenerator height
        gr  = 0.30
        g_x = xG
        g_y = yR - 0.10

        circ(g_x, g_y, gr, '#F9F9F9', ec='#666', lw=1.5, z=9)
        wedge = mpatches.Wedge((g_x, g_y), gr*0.88, 20, 160,
                                fc='#E8F5E9', ec='none', zorder=9)
        ax.add_patch(wedge); artists.append(wedge)
        for td in range(20, 161, 28):
            a = math.radians(td)
            line(g_x + gr*0.78*math.cos(a), g_y + gr*0.78*math.sin(a),
                 g_x + gr*0.93*math.cos(a), g_y + gr*0.93*math.sin(a),
                 c='#666', lw=1.0, z=10)
        nfrac = np.clip(P_now / (P_m * 1.5), 0, 1)
        nang  = math.radians(20 + nfrac * 140)
        line(g_x, g_y,
             g_x + gr*0.80*math.cos(nang),
             g_y + gr*0.80*math.sin(nang),
             c='#C62828', lw=2.2, z=11)
        circ(g_x, g_y, gr*0.07, '#C62828', ec='none', z=12)
        txt(g_x, g_y + gr + 0.14, 'PRESSURE', color='#333', fs=6.5, bold=True)
        txt(g_x, g_y - gr - 0.17, f'P = {P_now:.2f} bar', color='#333', fs=6.5)

        # Gauge 2: T_e thermometer — below pressure dial, clear spacing
        bw    = 0.14
        bh    = 0.70
        gap_between_gauges = 0.25

        g2_bot = g_y - gr - 0.17 - gap_between_gauges - bh
        bx_e   = g_x - 0.10
        fe     = np.clip((T_e_now - T_k) / (T_h * 1.1 - T_k), 0, 1)

        rect(bx_e - bw/2, g2_bot, bw, bh, '#EEEEEE', ec='#888', lw=1, z=9)
        if fe > 0:
            rect(bx_e - bw/2, g2_bot, bw, bh * fe,
                 _tc(T_e_now, T_k, T_h), ec='none', alpha=0.9, z=10)
        txt(bx_e, g2_bot + bh + 0.12, 'Tₑ', color='#B71C1C', fs=7, bold=True)
        txt(bx_e, g2_bot - 0.16, f'Tₑ = {T_e_now:.0f} K', color='#B71C1C', fs=6.2)

        # Gauge 3: T_c thermometer — below T_e bar, clear spacing
        g3_bot = g2_bot - gap_between_gauges - bh
        bx_c   = g_x + 0.10
        fc2    = np.clip((T_c_now - T_k) / (T_h * 1.1 - T_k), 0, 1)

        rect(bx_c - bw/2, g3_bot, bw, bh, '#EEEEEE', ec='#888', lw=1, z=9)
        if fc2 > 0:
            rect(bx_c - bw/2, g3_bot, bw, bh * fc2,
                 _tc(T_c_now, T_k, T_h), ec='none', alpha=0.9, z=10)
        txt(bx_c, g3_bot + bh + 0.12, 'Tc', color='#1565C0', fs=7, bold=True)
        txt(bx_c, g3_bot - 0.16, f'Tc = {T_c_now:.0f} K', color='#1565C0', fs=6.2)

        # Overlap assertions (debug verification)
        assert g2_bot + bh < g_y - gr, "T_e bar overlaps pressure dial"
        assert g3_bot + bh < g2_bot,   "T_c bar overlaps T_e bar"

        deg = int(math.degrees(theta)) % 360
        txt(fig_x0+0.15, fig_y1-0.20, f'θ = {deg}°',
            color='#444', fs=8, ha='left', bold=True, z=11)

        return artists

    ani = animation.FuncAnimation(fig, frame, frames=N_FRAMES,
                                   interval=1000 // FPS, blit=False)
    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        ani.save(tmp_path, writer='pillow', fps=FPS, dpi=85)
        plt.close(fig)
        with open(tmp_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

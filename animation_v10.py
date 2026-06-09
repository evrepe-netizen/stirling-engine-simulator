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
def build_engine_animation(geom_frozen, params_frozen, mode='Auto', render_version='png_states_v2'):
    """
    Returns base64-encoded animated GIF.
    Accepts frozen (tuple-of-pairs) representations of geom and params dicts
    so that @st.cache_data can hash the inputs.
    """
    geom   = dict(geom_frozen)
    params = dict(params_frozen)

    # Auto mode: about 14 seconds per full cycle.
    # Static stage modes render one synchronized frozen frame.
    N_FRAMES = 140
    FPS      = 10

    stage_theta_map = {
        # Static key states, synchronized with the corrected phase relation.
        # State 1: start of hot expansion, minimum volume, high pressure
        # State 2: end of hot expansion, maximum volume, higher pressure
        # State 3: end of cooling, maximum volume, low pressure
        # State 4: end of cold compression, minimum volume, lower than state 1
        'State 1 — start of hot expansion': 0.00 * 2 * math.pi,
        'State 2 — end of hot expansion':   0.25 * 2 * math.pi,
        'State 3 — end of cooling':         0.50 * 2 * math.pi,
        'State 4 — end of compression':     0.75 * 2 * math.pi,
    }
    FREEZE_THETA = stage_theta_map.get(mode)

    # Robust fallback: every non-Auto mode is a static state.
    # This prevents Streamlit/app label mismatches from accidentally rendering a GIF.
    if FREEZE_THETA is None and mode != 'Auto':
        if 'State 1' in mode or '1→2' in mode:
            FREEZE_THETA = 0.00 * 2 * math.pi
        elif 'State 2' in mode or '2→3' in mode:
            FREEZE_THETA = 0.25 * 2 * math.pi
        elif 'State 3' in mode or '3→4' in mode:
            FREEZE_THETA = 0.50 * 2 * math.pi
        elif 'State 4' in mode or '4→1' in mode:
            FREEZE_THETA = 0.75 * 2 * math.pi

    if FREEZE_THETA is not None:
        N_FRAMES = 1
        FPS = 1

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

    # Wider figure with separated panels:
    # left = engine animation, right = synchronized P-V and T-S diagrams.
    fig = plt.figure(figsize=(17, 7), dpi=80)
    fig.patch.set_facecolor('white')

    ax = fig.add_axes([0.03, 0.08, 0.58, 0.84])
    ax.set_xlim(fig_x0, fig_x1)
    ax.set_ylim(fig_y0, fig_y1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')

    # Synchronized educational diagrams.
    # These axes use the exact same theta/phase as the engine animation.
    ax_pv = fig.add_axes([0.68, 0.58, 0.28, 0.32])
    ax_ts = fig.add_axes([0.68, 0.14, 0.28, 0.32])

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

    # Educational T-S curve synchronized to the same phase.
    def _ts_point(th):
        """
        Ideal Stirling T-S diagram:
          1 (top-left)  →  2 (top-right)   : isothermal expansion  at T_h
          2 (top-right) →  3 (bot-right)   : isochoric cooling via regenerator (bows outward)
          3 (bot-right) →  4 (bot-left)    : isothermal compression at T_c
          4 (bot-left)  →  1 (top-left)    : isochoric heating via regenerator (bows outward)
        """
        phase = (th % (2*math.pi)) / (2*math.pi)
        Th, Tc = 1.0, 0.45
        # Corner entropy values
        S1, S2 = 0.13, 0.57   # top-left, top-right
        S3, S4 = 0.54, 0.16   # bot-right, bot-left
        if phase < 0.25:
            # 1→2: horizontal top, S increases
            u = phase / 0.25
            return S1 + (S2 - S1) * u, Th
        elif phase < 0.50:
            # 2→3: right side going down, bows slightly outward (rightward)
            u = (phase - 0.25) / 0.25
            S = S2 + (S3 - S2) * u + 0.06 * math.sin(math.pi * u)
            T = Th + (Tc - Th) * u
            return S, T
        elif phase < 0.75:
            # 3→4: horizontal bottom, S decreases
            u = (phase - 0.50) / 0.25
            return S3 + (S4 - S3) * u, Tc
        else:
            # 4→1: left side going up, bows slightly outward (leftward)
            u = (phase - 0.75) / 0.25
            S = S4 + (S1 - S4) * u - 0.05 * math.sin(math.pi * u)
            T = Tc + (Th - Tc) * u
            return S, T

    _S_pre = []
    _T_pre = []
    for _th in _th_pre:
        _s, _t = _ts_point(_th)
        _S_pre.append(_s)
        _T_pre.append(_t)

    def _stage_color_and_label(th):
        phase = (th % (2*math.pi)) / (2*math.pi)
        if phase < 0.25:
            return '#C62828', '1→2'
        elif phase < 0.50:
            return '#EF6C00', '2→3'
        elif phase < 0.75:
            return '#1565C0', '3→4'
        return '#6A1B9A', '4→1'

    # Ideal educational P-V diagram points.
    # The shape follows the classic Stirling-cycle form:
    # 1→2: curved hot isothermal expansion
    # 2→3: nearly constant-volume cooling
    # 3→4: curved cold isothermal compression
    # 4→1: nearly constant-volume heating
    #
    # It is intentionally educational, not the exact Schmidt loop.
    _PV1 = (950.0, 1.30)    # min volume, maximum pressure
    _PV2 = (1150.0, 0.95)   # max volume, pressure after expansion
    _PV3 = (1150.0, 0.78)   # max volume, low pressure after cooling
    _PV4 = (950.0, 1.05)    # min volume, pressure after compression

    def _pv_ideal_point(th):
        phase = (th % (2*math.pi)) / (2*math.pi)

        V1, P1 = _PV1
        V2, P2 = _PV2
        V3, P3 = _PV3
        V4, P4 = _PV4

        if phase < 0.25:
            # 1→2: hot isothermal-like expansion.
            # Curved drop from high pressure / low volume to high volume.
            u = phase / 0.25
            V = V1 + (V2 - V1) * u
            P = P2 + (P1 - P2) * (1 - u)**1.55
            return V, P

        elif phase < 0.50:
            # 2→3: nearly isochoric cooling.
            # Same volume, pressure drops.
            u = (phase - 0.25) / 0.25
            V = V2
            P = P2 + (P3 - P2) * u
            return V, P

        elif phase < 0.75:
            # 3→4: cold isothermal-like compression.
            # Volume decreases, pressure rises along a curved lower path.
            u = (phase - 0.50) / 0.25
            V = V3 + (V4 - V3) * u
            P = P3 + (P4 - P3) * (u**1.45)
            return V, P

        else:
            # 4→1: nearly isochoric heating.
            # Same volume, pressure rises to state 1.
            u = (phase - 0.75) / 0.25
            V = V4
            P = P4 + (P1 - P4) * u
            return V, P

    # Build curved ideal P-V curve from the four states.
    _Vtot_pre = []
    _P_pre = []
    for _th in _th_pre:
        _v, _p = _pv_ideal_point(_th)
        _Vtot_pre.append(_v)
        _P_pre.append(_p)

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
        return xP0 + 0.08 + (sP/2) * (1.0 - math.cos(th + phi))

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

        theta = FREEZE_THETA if FREEZE_THETA is not None else 2.0 * math.pi * i / N_FRAMES
        dx    = _dx(theta)
        pw_x  = _px(theta)

        Ve    = geom.get('V_cle', 1.77e-5) + (V_swe/2)*(1 + math.cos(theta))
        Vc    = (geom.get('V_clc', 7.2e-5)
                 + (V_swc/2)*(1 + math.cos(theta + phi))
                 + (V_swe/2)*(1 - math.cos(theta)))
        Sig   = Vc/T_k + Vrl/Tr + Ve/T_h
        P_now = P_m * Sig0 / Sig
        P_rel = float(np.clip((P_now - P_min_c) / max(P_max_c - P_min_c, 0.01), 0, 1))

        # Current ideal P-V point synchronized with the same theta as the engine.
        Vtot_now, P_pv_now = _pv_ideal_point(theta)
        S_now, T_now = _ts_point(theta)
        stage_col, stage_lab = _stage_color_and_label(theta)

        # Draw synchronized P-V diagram
        ax_pv.clear()
        ax_pv.plot(_Vtot_pre, _P_pre, color='#1565C0', lw=2.0, label='P-V loop')
        ax_pv.fill(_Vtot_pre, _P_pre, color='#1565C0', alpha=0.10)

        for lab, idx_lab in [('1', 0), ('2', len(_th_pre)//4), ('3', len(_th_pre)//2), ('4', 3*len(_th_pre)//4)]:
            ax_pv.scatter([_Vtot_pre[idx_lab]], [_P_pre[idx_lab]], s=34,
                          color='white', edgecolor='#263238', linewidth=1.0, zorder=5)
            ax_pv.text(_Vtot_pre[idx_lab], _P_pre[idx_lab], lab,
                       ha='center', va='center', fontsize=7,
                       fontweight='bold', color='#263238', zorder=6)

        ax_pv.scatter([Vtot_now], [P_pv_now], s=70, color=stage_col,
                      edgecolor='white', linewidth=1.0, zorder=7)
        ax_pv.set_title('Ideal Stirling P-V Diagram', fontsize=9, fontweight='bold')
        ax_pv.set_xlabel('Total gas volume [cm³]', fontsize=7)
        ax_pv.set_ylabel('P [bar]', fontsize=7)
        ax_pv.grid(alpha=0.25)
        ax_pv.set_xlim(925, 1175)
        ax_pv.set_ylim(0.72, 1.35)
        ax_pv.tick_params(labelsize=6)
        ax_pv.legend(fontsize=6, loc='upper right')

        # Draw synchronized T-S diagram
        ax_ts.clear()
        ax_ts.plot(_S_pre, _T_pre, color='#263238', lw=1.8)
        for lab, idx_lab in [('1', 0), ('2', len(_th_pre)//4), ('3', len(_th_pre)//2), ('4', 3*len(_th_pre)//4)]:
            ax_ts.scatter([_S_pre[idx_lab]], [_T_pre[idx_lab]], s=34,
                          color='white', edgecolor='#263238', linewidth=1.0, zorder=5)
            ax_ts.text(_S_pre[idx_lab], _T_pre[idx_lab], lab,
                       ha='center', va='center', fontsize=7,
                       fontweight='bold', color='#263238', zorder=6)

        ax_ts.scatter([S_now], [T_now], s=70, color=stage_col,
                      edgecolor='white', linewidth=1.0, zorder=7)
        ax_ts.set_title('T-S Diagram', fontsize=9, fontweight='bold')
        ax_ts.set_xlabel('Entropy S', fontsize=7)
        ax_ts.set_ylabel('Temperature T', fontsize=7)
        ax_ts.grid(alpha=0.25)
        ax_ts.tick_params(labelsize=6)
        ax_ts.set_xlim(-0.02, 0.78)
        ax_ts.set_ylim(0.32, 1.15)

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

        pB_ang = theta + math.pi / 2.0 + phi
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

        # Gauge panel removed.
        # P-V and T-S diagrams now provide the synchronized thermodynamic visualization.

        deg = int(math.degrees(theta)) % 360
        txt(fig_x0+0.15, fig_y1-0.20, f'θ = {deg}°',
            color='#444', fs=8, ha='left', bold=True, z=11)

        return artists

    # Static state modes: render a real PNG still image, not a one-frame GIF.
    if FREEZE_THETA is not None:
        frame(0)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            fig.savefig(tmp_path, format='png', dpi=120, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            with open(tmp_path, 'rb') as f:
                return base64.b64encode(f.read()).decode()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

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


# ── Standalone 60-second export ────────────────────────────────────────────────

def export_animation_60s(output_path=None):
    """
    Render a ~60-second standalone animation of the Gamma Stirling cycle and
    save it to *output_path*.

    * Tries MP4 first (ffmpeg); falls back to GIF (pillow).
    * Uses the same physics/drawing as build_engine_animation().
    * Does NOT depend on any Streamlit state.
    * Returns the path of the saved file.
    """
    import shutil
    from pathlib import Path

    # ── Default engine parameters (match the defaults used inside build_engine_animation) ──
    _geom_frozen = (
        ('D_displacer', 75.0e-3),
        ('S_displacer', 101.5e-3),
        ('D_power',     65.6e-3),
        ('S_power',     61.6e-3),
        ('L_r',         236.0e-3),
        ('D_r',          40.0e-3),
        ('phi',         math.radians(90.0)),
        ('V_cle',       1.77e-5),
        ('V_clc',       7.2e-5),
        ('V_r_lumped',  4.0e-4),
    )
    _params_frozen = (
        ('L_displacer',  0.235),
        ('T_h',        873.0),
        ('T_k',        300.0),
        ('P_mean_bar',   1.0),
    )

    # ── Timing: 10 fps × 60 s = 600 frames ≈ 4.3 full cycles ──────────────────
    FPS_EXP         = 10
    TOTAL_SECONDS   = 60
    N_FRAMES_EXP    = FPS_EXP * TOTAL_SECONDS      # 600
    FRAMES_PER_CYCLE = 140                          # matches build_engine_animation

    # ── Decide output path and format ──────────────────────────────────────────
    have_ffmpeg = shutil.which('ffmpeg') is not None
    if output_path is None:
        ext = '.mp4' if have_ffmpeg else '.gif'
        output_path = str(Path(__file__).parent / f'stirling_animation_60s{ext}')

    use_mp4 = output_path.endswith('.mp4')

    # ── Reproduce the build_engine_animation setup with 'Auto' mode ─────────
    # We call build_engine_animation once to get a tiny probe GIF just to
    # confirm the function works, then we build the long animation separately.
    # (Replicating the inner logic avoids touching the cached function itself.)

    geom   = dict(_geom_frozen)
    params = dict(_params_frozen)

    D_d = geom['D_displacer']
    S_d = geom['S_displacer']
    L_d = float(params['L_displacer'])
    D_p = geom['D_power']
    S_p = geom['S_power']
    L_r = geom['L_r']
    D_r = geom['D_r']
    phi = geom['phi']
    T_h = float(params['T_h'])
    T_k = float(params['T_k'])
    P_m = float(params['P_mean_bar'])

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

    _gauge_g_y  = yR - 0.10
    _gauge_gr   = 0.30
    _gauge_bh   = 0.70
    _gauge_gap  = 0.25
    _g2_bot = _gauge_g_y - _gauge_gr - 0.17 - _gauge_gap - _gauge_bh
    _g3_bot = _g2_bot    - _gauge_gap  - _gauge_bh

    fig_x0 = xD0 - 0.80
    fig_x1 = xG  + 0.75
    fig_y0 = min(yP - R_P - W - 0.40, _g3_bot - 0.30)
    fig_y1 = yR  + R_R + W + 0.60

    # ── Vertical layout: engine on top, P-V and T-S side-by-side below ────────
    fig = plt.figure(figsize=(13, 16), dpi=80)
    fig.patch.set_facecolor('white')

    # Engine panel: full width, top 55 %
    ax = fig.add_axes([0.03, 0.44, 0.94, 0.53])
    ax.set_xlim(fig_x0, fig_x1)
    ax.set_ylim(fig_y0, fig_y1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')

    # Diagrams: side by side in the bottom 38 %
    ax_pv = fig.add_axes([0.06, 0.05, 0.40, 0.33])
    ax_ts = fig.add_axes([0.54, 0.05, 0.40, 0.33])

    CW = '#888888';  CW2 = '#AAAAAA';  CROD = '#999999';  CFW = '#555555'

    artists_exp = []

    def _rect(x, y, w, h, fc, ec=CW, lw=1.2, alpha=1.0, z=3):
        p = mpatches.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z)
        ax.add_patch(p); artists_exp.append(p)

    def _circ(cx, cy, r, fc, ec='none', lw=0.8, alpha=1.0, z=4):
        p = plt.Circle((cx, cy), r, fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z)
        ax.add_patch(p); artists_exp.append(p)

    def _line(x0, y0, x1, y1, c=CW, lw=1.5, z=4):
        ln, = ax.plot([x0, x1], [y0, y1], color=c, lw=lw, zorder=z,
                      solid_capstyle='round')
        artists_exp.append(ln)

    def _txt(x, y, s, color='#333333', fs=6.5, ha='center', va='center',
             bold=False, z=10):
        t = ax.text(x, y, s, color=color, fontsize=fs, ha=ha, va=va,
                    fontweight='bold' if bold else 'normal', zorder=z)
        artists_exp.append(t)

    def _bezier(t_arr, p0, pc, p1):
        x = (1-t_arr)**2*p0[0] + 2*(1-t_arr)*t_arr*pc[0] + t_arr**2*p1[0]
        y = (1-t_arr)**2*p0[1] + 2*(1-t_arr)*t_arr*pc[1] + t_arr**2*p1[1]
        return x, y

    _t80 = np.linspace(0.0, 1.0, 80)

    _ps1 = (xD0 - W, yD);  _pe1 = (xR0 - W, yR)
    _pc1 = (min(_ps1[0], _pe1[0]) - 0.55, (_ps1[1] + _pe1[1]) / 2.0)
    _bx1, _by1 = _bezier(_t80, _ps1, _pc1, _pe1)

    _ps2 = (xD1 + W, yD);  _pe2 = (xR1 + W, yR)
    _pc2 = (max(_ps2[0], _pe2[0]) + 0.55, (_ps2[1] + _pe2[1]) / 2.0)
    _bx2, _by2 = _bezier(_t80, _ps2, _pc2, _pe2)

    def _bl(p0, pc, p1, n=200):
        t = np.linspace(0, 1, n)
        x, y = _bezier(t, p0, pc, p1)
        return float(np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)))

    L_hot   = sD * 0.45;   L_pipe1 = _bl(_ps1, _pc1, _pe1)
    L_regen = lR;           L_pipe2 = _bl(_ps2, _pc2, _pe2)
    L_cold  = sD * 0.45;   L_total = L_hot + L_pipe1 + L_regen + L_pipe2 + L_cold

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
            x = xR0 + f * lR; y = yR; T_base = T_h - (T_h - T_k) * f
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

    V_swe = math.pi * (D_d/2)**2 * S_d
    V_swc = math.pi * (D_p/2)**2 * S_p
    Vrl   = geom['V_r_lumped']
    Tr    = (T_h - T_k) / math.log(max(T_h/T_k, 1.001))
    Ve0   = geom['V_cle'] + V_swe/2
    Vc0   = geom['V_clc'] + V_swc/2
    Sig0  = Vc0/T_k + Vrl/Tr + Ve0/T_h

    _th_pre = np.linspace(0, 2*math.pi, 120, endpoint=False)
    _P_pre_build = []
    for _th in _th_pre:
        _Ve = geom['V_cle'] + (V_swe/2)*(1 + math.cos(_th))
        _Vc = (geom['V_clc']
               + (V_swc/2)*(1 + math.cos(_th - phi))
               + (V_swe/2)*(1 - math.cos(_th)))
        _Sig = _Vc/T_k + Vrl/Tr + _Ve/T_h
        _P_pre_build.append(P_m * Sig0 / _Sig)
    P_min_c = min(_P_pre_build)
    P_max_c = max(_P_pre_build)

    def _ts_point(th):
        phase = (th % (2*math.pi)) / (2*math.pi)
        Th, Tc = 1.0, 0.45
        S1, S2 = 0.13, 0.57
        S3, S4 = 0.54, 0.16
        if phase < 0.25:
            u = phase / 0.25
            return S1 + (S2 - S1) * u, Th
        elif phase < 0.50:
            u = (phase - 0.25) / 0.25
            return S2 + (S3 - S2) * u + 0.06 * math.sin(math.pi * u), Th + (Tc - Th) * u
        elif phase < 0.75:
            u = (phase - 0.50) / 0.25
            return S3 + (S4 - S3) * u, Tc
        else:
            u = (phase - 0.75) / 0.25
            return S4 + (S1 - S4) * u - 0.05 * math.sin(math.pi * u), Tc + (Th - Tc) * u

    _S_pre = [_ts_point(_th)[0] for _th in _th_pre]
    _T_pre = [_ts_point(_th)[1] for _th in _th_pre]

    def _stage_color_and_label(th):
        phase = (th % (2*math.pi)) / (2*math.pi)
        if phase < 0.25:   return '#C62828', '1→2  Isothermal Expansion'
        elif phase < 0.50: return '#EF6C00', '2→3  Constant-Volume Cooling'
        elif phase < 0.75: return '#1565C0', '3→4  Isothermal Compression'
        return '#6A1B9A', '4→1  Constant-Volume Heating'

    _PV1=(950.0,1.30); _PV2=(1150.0,0.95); _PV3=(1150.0,0.78); _PV4=(950.0,1.05)

    def _pv_ideal_point(th):
        phase = (th % (2*math.pi)) / (2*math.pi)
        V1,P1=_PV1; V2,P2=_PV2; V3,P3=_PV3; V4,P4=_PV4
        if phase < 0.25:
            u=phase/0.25; return V1+(V2-V1)*u, P2+(P1-P2)*(1-u)**1.55
        elif phase < 0.50:
            u=(phase-0.25)/0.25; return V2, P2+(P3-P2)*u
        elif phase < 0.75:
            u=(phase-0.50)/0.25; return V3+(V4-V3)*u, P3+(P4-P3)*(u**1.45)
        else:
            u=(phase-0.75)/0.25; return V4, P4+(P1-P4)*u

    _Vtot_pre = [_pv_ideal_point(_th)[0] for _th in _th_pre]
    _P_pv_pre = [_pv_ideal_point(_th)[1] for _th in _th_pre]

    N_MAIN  = 34;  N_POWER = 12
    rng = np.random.default_rng(42)
    s0_main      = np.linspace(0.0, 1.0, N_MAIN, endpoint=False)
    jit_phase    = rng.uniform(0, 2*math.pi, N_MAIN)
    jit_amp_frac = rng.uniform(0.3, 0.85, N_MAIN)
    s0_power     = np.linspace(0.0, 1.0, N_POWER, endpoint=False)
    jit_phase_p  = rng.uniform(0, 2*math.pi, N_POWER)

    A_flow = (L_regen / L_total) * 0.85 + 0.05

    def _dx(th): return xD0 + 0.10 + (sD/2) * (1.0 - math.cos(th))
    def _px(th): return xP0 + 0.08 + (sP/2) * (1.0 - math.cos(th + phi))

    # ── Draw static background (once) ──────────────────────────────────────
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
    pW3 = 0.09; conn_x = xCold - pW3 * 0.5
    ax.add_patch(mpatches.FancyBboxPatch(
        (conn_x - pW3, yP+R_P+W), 2*pW3, yD-R_D-W - (yP+R_P+W),
        boxstyle='round,pad=0.01', fc=CW2, ec=CW, lw=1.5, zorder=2))
    ax.add_patch(plt.Circle((conn_x, yD-R_D-W*0.5), pW3*1.1, fc=CW, ec='none', zorder=4))
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

    # ── Per-frame drawing ────────────────────────────────────────────────────
    def _frame_exp(i):
        for a in artists_exp: a.remove()
        artists_exp.clear()

        # theta advances continuously over all 600 frames, covering ~4.3 cycles
        theta = 2.0 * math.pi * i / FRAMES_PER_CYCLE
        dx    = _dx(theta)
        pw_x  = _px(theta)

        Ve    = geom['V_cle'] + (V_swe/2)*(1 + math.cos(theta))
        Vc    = (geom['V_clc']
                 + (V_swc/2)*(1 + math.cos(theta + phi))
                 + (V_swe/2)*(1 - math.cos(theta)))
        Sig   = Vc/T_k + Vrl/Tr + Ve/T_h
        P_now = P_m * Sig0 / Sig
        P_rel = float(np.clip((P_now - P_min_c) / max(P_max_c - P_min_c, 0.01), 0, 1))

        Vtot_now, P_pv_now = _pv_ideal_point(theta)
        S_now, T_now_ts    = _ts_point(theta)
        stage_col, stage_lab = _stage_color_and_label(theta)

        # Stage label (top-left of engine panel)
        _txt(fig_x0 + 0.15, fig_y1 - 0.20,
             f'θ = {int(math.degrees(theta)) % 360}°  |  {stage_lab}',
             color=stage_col, fs=7.8, ha='left', bold=True, z=11)

        # P-V diagram
        ax_pv.clear()
        ax_pv.plot(_Vtot_pre, _P_pv_pre, color='#1565C0', lw=2.0)
        ax_pv.fill(_Vtot_pre, _P_pv_pre, color='#1565C0', alpha=0.10)
        for lab, idx_l in [('1',0),('2',len(_th_pre)//4),('3',len(_th_pre)//2),('4',3*len(_th_pre)//4)]:
            ax_pv.scatter([_Vtot_pre[idx_l]], [_P_pv_pre[idx_l]], s=34,
                          color='white', edgecolor='#263238', linewidth=1.0, zorder=5)
            ax_pv.text(_Vtot_pre[idx_l], _P_pv_pre[idx_l], lab,
                       ha='center', va='center', fontsize=7, fontweight='bold',
                       color='#263238', zorder=6)
        ax_pv.scatter([Vtot_now], [P_pv_now], s=70, color=stage_col,
                      edgecolor='white', linewidth=1.0, zorder=7)
        ax_pv.set_title('Ideal Stirling P-V Diagram', fontsize=9, fontweight='bold')
        ax_pv.set_xlabel('Total gas volume [cm³]', fontsize=7)
        ax_pv.set_ylabel('P [bar]', fontsize=7)
        ax_pv.grid(alpha=0.25); ax_pv.set_xlim(925,1175); ax_pv.set_ylim(0.72,1.35)
        ax_pv.tick_params(labelsize=6)

        # T-S diagram
        ax_ts.clear()
        ax_ts.plot(_S_pre, _T_pre, color='#263238', lw=1.8)
        for lab, idx_l in [('1',0),('2',len(_th_pre)//4),('3',len(_th_pre)//2),('4',3*len(_th_pre)//4)]:
            ax_ts.scatter([_S_pre[idx_l]], [_T_pre[idx_l]], s=34,
                          color='white', edgecolor='#263238', linewidth=1.0, zorder=5)
            ax_ts.text(_S_pre[idx_l], _T_pre[idx_l], lab,
                       ha='center', va='center', fontsize=7, fontweight='bold',
                       color='#263238', zorder=6)
        ax_ts.scatter([S_now], [T_now_ts], s=70, color=stage_col,
                      edgecolor='white', linewidth=1.0, zorder=7)
        ax_ts.set_title('T-S Diagram', fontsize=9, fontweight='bold')
        ax_ts.set_xlabel('Entropy S', fontsize=7); ax_ts.set_ylabel('Temperature T', fontsize=7)
        ax_ts.grid(alpha=0.25); ax_ts.set_xlim(0.05,0.70); ax_ts.set_ylim(0.35,1.10)
        ax_ts.tick_params(labelsize=6)

        dT_adi = (P_rel - 0.5) * (T_h - T_k) * 0.28

        # Gas zones
        hw = max(0.01, dx - xD0 - 0.12)
        _rect(xD0+0.02, yD-R_D+0.02, hw, 2*R_D-0.04, '#FFE4E4', ec='none', alpha=0.70, z=1)
        cw = max(0.01, xD1 - 0.12 - (dx+lD+0.08))
        if cw > 0.01:
            _rect(dx+lD+0.08, yD-R_D+0.02, cw, 2*R_D-0.04, '#E4E8FF', ec='none', alpha=0.70, z=1)
        pgw = max(0.01, pw_x - xP0 - 0.08)
        _rect(xP0+0.02, yP-R_P+0.02, pgw, 2*R_P-0.04, '#FFFDE7', ec='none', alpha=0.60, z=1)

        # Displacer body + rod
        _rect(dx, yD - R_D*0.95, lD, 2*R_D*0.95, '#888888', ec='#666', lw=1.2, z=6)
        _line(dx+lD, yD, xD1+W+0.05, yD, c=CROD, lw=3.5, z=5)
        _rect(xD1+W*0.1, yD-0.055, 0.065, 0.11, '#AAAAAA', ec=CW, lw=1.0, z=7)
        _line(xD1+W+0.07, yD, xD_cross, yD, c=CROD, lw=3.0, z=5)

        # Power piston + rod
        _rect(pw_x, yP-R_P*0.93, 0.09, 2*R_P*0.93, '#888888', ec='#666', lw=1.2, z=6)
        _line(pw_x+0.09, yP, xP1+W+0.05, yP, c=CROD, lw=3.5, z=5)
        _rect(xP1+W*0.1, yP-0.045, 0.065, 0.09, '#AAAAAA', ec=CW, lw=1.0, z=7)
        _line(xP1+W+0.07, yP, xP_cross, yP, c=CROD, lw=3.0, z=5)

        # Flywheel
        fw = plt.Circle((xFW, yFW), Rfw, fc='#F0F0F0', ec='#333', lw=2.0, zorder=5)
        ax.add_patch(fw); artists_exp.append(fw)
        for k in range(6):
            ang = theta + k * math.pi / 3.0
            _line(xFW + Rfw*0.12*math.cos(ang), yFW + Rfw*0.12*math.sin(ang),
                  xFW + Rfw*0.90*math.cos(ang), yFW + Rfw*0.90*math.sin(ang),
                  c=CFW, lw=1.3, z=6)
        _circ(xFW, yFW, Rfw*0.12, CFW, ec='#222', lw=1.5, z=7)

        pA_ang = theta + math.pi / 2.0
        pA_x   = xFW + Rfw*0.80*math.cos(pA_ang)
        pA_y   = yFW + Rfw*0.80*math.sin(pA_ang)
        _circ(pA_x, pA_y, 0.042, '#EF5350', ec='#B71C1C', lw=1.2, z=8)
        _line(xD_cross, yD, pA_x, pA_y, c=CROD, lw=2.8, z=5)

        pB_ang = theta + math.pi / 2.0 + phi
        pB_x   = xFW + Rfw*0.80*math.cos(pB_ang)
        pB_y   = yFW + Rfw*0.80*math.sin(pB_ang)
        _circ(pB_x, pB_y, 0.038, '#42A5F5', ec='#1565C0', lw=1.2, z=8)
        _line(xP_cross, yP, pB_x, pB_y, c=CROD, lw=2.8, z=5)

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
            _circ(x, y + jy, 0.050, _tc(T_vis, T_k, T_h), ec='#888', lw=0.4, alpha=0.90, z=7)

        vol_back  = xP0 + 0.06
        vol_front = max(vol_back + 0.04, pw_x - 0.02)
        for idx in range(N_POWER):
            x  = vol_back + s0_power[idx] * max(0.04, vol_front - vol_back)
            jy = 0.55 * R_P * math.sin(jit_phase_p[idx] + theta * 0.5)
            T_vis = float(np.clip(T_k + dT_adi * 1.3, T_k * 0.85, T_h * 0.9))
            _circ(x, yP + jy, 0.045, _tc(T_vis, T_k, T_h), ec='#888', lw=0.4, alpha=0.88, z=7)

        return artists_exp

    ani = animation.FuncAnimation(fig, _frame_exp, frames=N_FRAMES_EXP,
                                   interval=1000 // FPS_EXP, blit=False)

    # ── Save ────────────────────────────────────────────────────────────────
    if use_mp4 and have_ffmpeg:
        writer = animation.FFMpegWriter(fps=FPS_EXP, bitrate=1800,
                                        extra_args=['-vcodec', 'libx264',
                                                    '-pix_fmt', 'yuv420p'])
        ani.save(output_path, writer=writer, dpi=85)
    else:
        # Fallback to GIF; keep dpi low to limit file size
        if use_mp4:
            output_path = output_path.replace('.mp4', '.gif')
        ani.save(output_path, writer='pillow', fps=FPS_EXP, dpi=72)

    plt.close(fig)
    return output_path


if __name__ == '__main__':
    import sys
    out = export_animation_60s(sys.argv[1] if len(sys.argv) > 1 else None)
    size_mb = os.path.getsize(out) / 1024 / 1024
    print(f'Saved: {out}  ({size_mb:.1f} MB)')


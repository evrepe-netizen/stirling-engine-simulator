"""
physics.py — Stirling Engine core physics
==========================================
Pure physics module: no Streamlit, no plotting.
Imported by app.py and optimization.py.

Option A (v4+): both models use the same gas mass M.
  - Schmidt computes M from P_mean target.
  - Adiabatic receives that same M; its P_mean is an output.
"""

import math
import numpy as np
from scipy.integrate import solve_ivp

# ── Prototype defaults ───────────────────────────────────────────────────────
PROTOTYPE = {
    'D_displacer': 75.0,    # mm
    'S_displacer': 101.5,   # mm
    'D_power':     65.6,    # mm
    'S_power':     61.6,    # mm
    'phi_deg':     90.0,    # degrees
    'D_r':         40.0,    # mm
    'L_r':         236.0,   # mm
    'd_wire':      1.0,     # mm
    'porosity':    0.9,
    'gas':         'Air',
    'T_h':         873,     # K
    'T_k':         300,     # K
    'P_mean_bar':  1.0,     # bar
    'f':           10,      # Hz
    'eps_reg':     0.85,
    'eta_mech':    0.85,
    'C_leak':      0.02,
    'k_metal':     26.0,    # W/m·K
    't_wall':      2.0,     # mm
    # Fixed geometry (measured on prototype, not user-editable)
    'V_loop_cold': 6.7278e-5,  # m³
    'V_loop_hot':  7.1127e-5,  # m³
    'V_cle':       1.7671e-5,  # m³  (hot-side clearance)
    'V_clc':       7.2041e-5,  # m³  (cold-side clearance)
    'L_displacer': 0.235,      # m
    'gap':         0.00025,    # m
    'P_ref':       1.0e5,      # Pa  (reference pressure for leakage)
}

# ── Gas properties ───────────────────────────────────────────────────────────
GASES = {
    'Air':      {'R': 287,  'Cv': 718,   'Cp': 1005,  'gamma': 1.4,   'mu': 2.7e-5, 'k_gas': 0.04},
    'Helium':   {'R': 2077, 'Cv': 3116,  'Cp': 5193,  'gamma': 1.667, 'mu': 3.4e-5, 'k_gas': 0.18},
    'Hydrogen': {'R': 4124, 'Cv': 10160, 'Cp': 14284, 'gamma': 1.406, 'mu': 1.4e-5, 'k_gas': 0.22},
}

# ── Unit conversion ──────────────────────────────────────────────────────────
def to_si(params):
    """Convert mm → m, bar → Pa."""
    p = dict(params)
    for key in ('D_displacer', 'S_displacer', 'D_power', 'S_power',
                'D_r', 'L_r', 'd_wire', 't_wall'):
        p[key] = params[key] * 1e-3
    p['P_mean'] = params['P_mean_bar'] * 1e5
    return p


# ── Geometry ─────────────────────────────────────────────────────────────────
def build_geometry(ps):
    """Compute derived volumes from SI parameters."""
    g = dict(ps)
    A_d = math.pi * (ps['D_displacer'] / 2) ** 2
    A_p = math.pi * (ps['D_power']     / 2) ** 2
    g['V_swe']      = A_d * ps['S_displacer']
    g['V_swc']      = A_p * ps['S_power']
    g['phi']        = math.radians(ps['phi_deg'])
    g['V_r_only']   = math.pi * (ps['D_r'] / 2) ** 2 * ps['L_r'] * ps['porosity']
    g['V_k']        = ps.get('V_loop_cold', PROTOTYPE['V_loop_cold'])
    g['V_h']        = ps.get('V_loop_hot',  PROTOTYPE['V_loop_hot'])
    g['V_r']        = g['V_r_only']
    g['V_r_lumped'] = g['V_k'] + g['V_r_only'] + g['V_h']
    g['V_cle']      = ps.get('V_cle', PROTOTYPE['V_cle'])
    g['V_clc']      = ps.get('V_clc', PROTOTYPE['V_clc'])
    return g


def _vol_arrays(geom, theta):
    """Instantaneous expansion (V_e) and compression (V_c) volumes [m³]."""
    V_e = geom['V_cle'] + (geom['V_swe'] / 2) * (1 + np.cos(theta))
    V_c = (geom['V_clc']
           + (geom['V_swc'] / 2) * (1 + np.cos(theta - geom['phi']))
           + (geom['V_swe'] / 2) * (1 - np.cos(theta)))
    return V_e, V_c


# ── Schmidt (isothermal) cycle ────────────────────────────────────────────────
def schmidt_cycle(geom, gas, P_target):
    """
    Closed-form isothermal (Schmidt) analysis.
    Returns result dict including M (gas mass used as reference for adiabatic).
    """
    theta = np.deg2rad(np.arange(361))
    V_e, V_c = _vol_arrays(geom, theta)
    T_h, T_k = gas['T_h'], gas['T_k']
    T_r = (T_h - T_k) / math.log(T_h / T_k)   # log-mean temperature
    Sigma = V_c / T_k + geom['V_r_lumped'] / T_r + V_e / T_h
    M     = P_target / (gas['R'] * (1.0 / Sigma).mean())
    P     = M * gas['R'] / Sigma
    return dict(
        theta=theta, V_e=V_e, V_c=V_c, P=P,
        T_c=np.full_like(theta, T_k),
        T_e=np.full_like(theta, T_h),
        M=M, T_r=T_r, model='Schmidt (Isothermal)'
    )


# ── Adiabatic cycle (Urieli-Berchowitz) ──────────────────────────────────────
def _adiabatic_rhs(theta, y, geom, gas):
    """ODE right-hand side for the ideal adiabatic model."""
    P, T_c, T_e, m_c, m_e = y
    if P <= 0 or T_c <= 0 or T_e <= 0 or m_c <= 0 or m_e <= 0:
        return [0, 0, 0, 0, 0]

    V_e = geom['V_cle'] + (geom['V_swe'] / 2) * (1 + np.cos(theta))
    V_c = (geom['V_clc']
           + (geom['V_swc'] / 2) * (1 + np.cos(theta - geom['phi']))
           + (geom['V_swe'] / 2) * (1 - np.cos(theta)))
    dV_e = -(geom['V_swe'] / 2) * np.sin(theta)
    dV_c = (-(geom['V_swc'] / 2) * np.sin(theta - geom['phi'])
             + (geom['V_swe'] / 2) * np.sin(theta))

    g   = gas['gamma']
    T_h = gas['T_h']; T_k = gas['T_k']; T_r = gas['T_r']; R = gas['R']

    # Interface temperatures (Urieli sign convention)
    T_ck = T_c if dV_c < 0 else T_k   # compression → cold end
    T_he = T_h if dV_e > 0 else T_e   # expansion  → hot end

    num  = -g * P * (dV_c / T_ck + dV_e / T_he)
    den  = (V_c / T_ck
            + g * (geom['V_k'] / T_k + geom['V_r'] / T_r + geom['V_h'] / T_h)
            + V_e / T_he)
    dP   = num / den
    dmc  = (P * dV_c + V_c * dP / g) / (R * T_ck)
    dme  = (P * dV_e + V_e * dP / g) / (R * T_he)
    dT_c = T_c * (dP / P + dV_c / V_c - dmc / m_c)
    dT_e = T_e * (dP / P + dV_e / V_e - dme / m_e)
    return [dP, dT_c, dT_e, dmc, dme]


def adiabatic_cycle(geom, gas, M_fixed, max_cycles=20, tol=0.5):
    """
    Option A — Fair comparison.
    Receives M_fixed (= Schmidt mass for the same P_mean target).
    P_mean is an OUTPUT — it may differ from the Schmidt target pressure.

    Returns result dict or None on failure.
    """
    T_h, T_k = gas['T_h'], gas['T_k']
    T_r = (T_h - T_k) / math.log(T_h / T_k)
    gas['T_r'] = T_r

    theta_eval = np.linspace(0, 2 * math.pi, 361)
    V_e_arr, V_c_arr = _vol_arrays(geom, theta_eval)

    # Initial conditions consistent with M_fixed
    # All gas starts at wall temperatures; pressure from mass constraint
    Sigma0 = (V_c_arr[0] / T_k
              + geom['V_k'] / T_k + geom['V_r'] / T_r + geom['V_h'] / T_h
              + V_e_arr[0] / T_h)
    P0  = M_fixed * gas['R'] / Sigma0
    mc0 = P0 * V_c_arr[0] / (gas['R'] * T_k)
    me0 = P0 * V_e_arr[0] / (gas['R'] * T_h)
    y0  = [P0, T_k, T_h, mc0, me0]

    for cycle in range(max_cycles):
        try:
            sol = solve_ivp(
                _adiabatic_rhs, (0, 2 * math.pi), y0,
                t_eval=theta_eval, args=(geom, gas),
                method='RK45', rtol=1e-8, atol=1e-11,
                max_step=math.radians(0.5)
            )
            if not sol.success:
                return None
        except Exception:
            return None

        P_arr, Tc_arr, Te_arr, mc_arr, me_arr = sol.y
        dTc = abs(Tc_arr[-1] - y0[1])
        dTe = abs(Te_arr[-1] - y0[2])
        if dTc < tol and dTe < tol and cycle > 0:
            break
        y0 = [P_arr[-1], Tc_arr[-1], Te_arr[-1], mc_arr[-1], me_arr[-1]]

    return dict(
        theta=theta_eval, V_e=V_e_arr, V_c=V_c_arr,
        P=P_arr, T_c=Tc_arr, T_e=Te_arr,
        M=M_fixed, T_r=T_r, model='Adiabatic (RK45)',
        cycles_to_converge=cycle + 1
    )


# ── Loss model ────────────────────────────────────────────────────────────────
def compute_losses(result, geom, gas, params_si, losses_flags):
    """
    Compute all loss components and shaft power.
    Returns a dict with all work and heat quantities [J/cycle] and [W].
    """
    V_e, V_c, P = result['V_e'], result['V_c'], result['P']
    M = result['M']
    dV_e  = np.diff(V_e); dV_c = np.diff(V_c)
    P_mid = 0.5 * (P[:-1] + P[1:])
    W_e     = float(np.sum(P_mid * dV_e))
    W_c     = float(np.sum(P_mid * dV_c))
    W_cycle = W_e + W_c
    P_mean  = float(P.mean())

    out = {
        'W_cycle': W_cycle, 'W_e_cycle': W_e, 'W_c_cycle': W_c,
        'P_mean': P_mean, 'P_max': float(P.max()), 'P_min': float(P.min()),
        'T_e_max': float(result['T_e'].max()), 'T_e_min': float(result['T_e'].min()),
        'T_c_max': float(result['T_c'].max()), 'T_c_min': float(result['T_c'].min()),
        'M': M,
    }

    # Flow resistance (Ergun equation for packed-bed regenerator)
    if losses_flags.get('flow', True):
        eps  = geom['porosity']; d_w = geom['d_wire']
        A_r  = math.pi * (geom['D_r'] / 2) ** 2
        R_v  = 150 * (1 - eps) ** 2 / (eps ** 3 * d_w ** 2)
        R_i  = 1.75 * (1 - eps)    / (eps ** 3 * d_w)
        Vdot = dV_e * 360 * params_si['f']
        u    = Vdot / (A_r * eps)
        rho  = P_mid / (gas['R'] * result['T_r'])
        dPdr = geom['L_r'] * (R_v * gas['mu'] * u + R_i * rho * u * np.abs(u))
        out['W_pump'] = float(np.sum(np.abs(dPdr * dV_e)))
    else:
        out['W_pump'] = 0.0

    W_after_flow = W_cycle - out['W_pump']

    # Seal leakage
    if losses_flags.get('leakage', True):
        out['W_leak'] = W_after_flow * params_si['C_leak'] * (P_mean / params_si['P_ref'])
    else:
        out['W_leak'] = 0.0

    W_after_leak = W_after_flow - out['W_leak']

    # Mechanical friction
    if losses_flags.get('mechanical', True):
        out['W_mech_loss'] = W_after_leak * (1 - params_si['eta_mech'])
    else:
        out['W_mech_loss'] = 0.0

    out['W_shaft'] = W_after_leak - out['W_mech_loss']
    out['P_brake'] = out['W_shaft'] * params_si['f']

    # Heat input
    out['Q_e']    = W_e
    out['Q_miss'] = (M * gas['Cv'] * (gas['T_h'] - gas['T_k']) * (1 - params_si['eps_reg'])
                     if losses_flags.get('regen_imp', True) else 0.0)
    if losses_flags.get('wall_cond', True):
        D_out        = geom['D_r'] + 2 * params_si['t_wall']
        A_ring       = math.pi * ((D_out / 2) ** 2 - (geom['D_r'] / 2) ** 2)
        out['Q_cond_W'] = params_si['k_metal'] * A_ring * (gas['T_h'] - gas['T_k']) / geom['L_r']
        out['Q_cond']   = out['Q_cond_W'] / params_si['f']
    else:
        out['Q_cond'] = 0.0; out['Q_cond_W'] = 0.0

    out['Q_shuttle'] = 0.0
    out['Q_in']      = out['Q_e'] + out['Q_miss'] + out['Q_cond'] + out['Q_shuttle']
    out['Q_in_W']    = out['Q_in'] * params_si['f']
    out['eta_brake'] = out['W_shaft'] / out['Q_in'] if out['Q_in'] > 0 else 0.0
    out['eta_carnot']  = 1 - gas['T_k'] / gas['T_h']
    out['frac_carnot'] = (out['eta_brake'] / out['eta_carnot']
                          if out['eta_carnot'] > 0 else 0.0)
    return out


# ── Top-level simulate() ──────────────────────────────────────────────────────
def simulate(params, model='schmidt', losses_flags=None):
    """
    Run Schmidt and/or Adiabatic cycle, apply loss model.
    Both models always use the same gas mass M (Option A).

    Returns dict with keys: result, losses, geom, gas, params, params_si,
                            losses_flags, model, M_schmidt.
    Returns None on solver failure.
    """
    if losses_flags is None:
        losses_flags = dict(flow=True, regen_imp=True, mechanical=True,
                            wall_cond=True, leakage=True, shuttle=False)

    params_si  = to_si(params)
    geom       = build_geometry(params_si)
    gas        = dict(GASES[params['gas']])
    gas['T_h'] = params['T_h']
    gas['T_k'] = params['T_k']
    P_target   = params_si['P_mean']

    # Schmidt always computed first — its M is the shared reference mass
    result_s  = schmidt_cycle(geom, gas, P_target)
    M_schmidt = result_s['M']

    if model == 'schmidt':
        result = result_s
    else:
        # Adiabatic receives the same M — P_mean is an output
        result = adiabatic_cycle(geom, gas, M_schmidt)
        if result is None:
            return None

    losses_out = compute_losses(result, geom, gas, params_si, losses_flags)
    return dict(
        result=result, losses=losses_out,
        geom=geom, gas=gas,
        params=params, params_si=params_si,
        losses_flags=losses_flags, model=model,
        M_schmidt=M_schmidt,
    )


# ── Validation helpers ────────────────────────────────────────────────────────
def validate_mass_conservation(result, geom, gas):
    """Δm/m_avg < 2% across cycle."""
    P   = result['P']
    T_h = gas['T_h']; T_k = gas['T_k']
    T_r = result.get('T_r', (T_h - T_k) / math.log(T_h / T_k))
    m   = (P * result['V_c'] / (gas['R'] * result['T_c'])
           + P * geom['V_k']  / (gas['R'] * T_k)
           + P * geom['V_r']  / (gas['R'] * T_r)
           + P * geom['V_h']  / (gas['R'] * T_h)
           + P * result['V_e'] / (gas['R'] * result['T_e']))
    delta = (m.max() - m.min()) / m.mean() * 100
    return delta, delta < 2.0


def validate_first_law(losses):
    """W_cycle = W_e + W_c (numerical identity, error < 0.1%)."""
    W  = losses['W_cycle']
    Wc = losses['W_e_cycle'] + losses['W_c_cycle']
    err = abs(W - Wc) / (abs(W) + 1e-12) * 100
    return err, err < 0.1 and W > 0


def validate_carnot(losses):
    """η_brake ≤ η_Carnot."""
    return losses['eta_brake'], losses['eta_carnot'], losses['eta_brake'] <= losses['eta_carnot']


def validate_pressure_scaling(params, losses_flags):
    """W ∝ P_mean: doubling pressure should double work (< 0.5% error)."""
    lf = dict(flow=False, regen_imp=False, mechanical=False,
              wall_cond=False, leakage=False, shuttle=False)
    p2 = dict(params); p2['P_mean_bar'] = params['P_mean_bar'] * 2.0
    s1 = simulate(params, model='schmidt', losses_flags=lf)
    s2 = simulate(p2,     model='schmidt', losses_flags=lf)
    if s1 is None or s2 is None:
        return None, None, None, False
    W1 = s1['losses']['W_cycle']; W2 = s2['losses']['W_cycle']
    ratio = W2 / W1 if abs(W1) > 1e-12 else 0
    err   = abs(ratio - 2.0) / 2.0 * 100
    return W1, W2, ratio, err < 0.5

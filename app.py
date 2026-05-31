"""
Stirling Engine Simulator — Streamlit App v2
=============================================
- Iterative adiabatic (P_mean matches target)
- Reset to prototype button
- Optimization tab with lockable parameters

To run:
    streamlit run stirling_app_v2.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import streamlit as st
from itertools import product

# ============================================================
# PROTOTYPE & CONSTANTS
# ============================================================

PROTOTYPE = {
    'D_displacer':  75.0,    # mm
    'S_displacer':  101.5,   # mm
    'D_power':      65.6,    # mm
    'S_power':      61.6,    # mm
    'phi_deg':      90.0,    # degrees
    'D_r':          40.0,    # mm
    'L_r':          236.0,   # mm
    'd_wire':       1.0,     # mm
    'porosity':     0.9,
    'gas':          'Air',
    'T_h':          873,     # K
    'T_k':          300,     # K
    'P_mean_bar':   1.0,     # bar
    'f':            10,      # Hz
    'eps_reg':      0.85,
    'eta_mech':     0.85,
    'C_leak':       0.02,
    'k_metal':      26.0,
    't_wall':       2.0,     # mm
    # Fixed (not user-editable)
    'V_loop_cold':  6.7278e-5,
    'V_loop_hot':   7.1127e-5,
    'V_cle':        1.7671e-5,
    'V_clc':        7.2041e-5,
    'L_displacer':  0.235,
    'gap':          0.00025,
    'P_ref':        1.0e5,
}

GASES = {
    'Air':      {'R': 287,  'Cv': 718,   'Cp': 1005,  'gamma': 1.4,   'mu': 2.7e-5, 'k_gas': 0.04},
    'Helium':   {'R': 2077, 'Cv': 3116,  'Cp': 5193,  'gamma': 1.667, 'mu': 3.4e-5, 'k_gas': 0.18},
    'Hydrogen': {'R': 4124, 'Cv': 10160, 'Cp': 14284, 'gamma': 1.406, 'mu': 1.4e-5, 'k_gas': 0.22},
}

# Parameters that can be locked/optimized (display name, key, min, max, step, units)
OPTIMIZABLE_PARAMS = [
    ('Displacer diameter', 'D_displacer', 30, 150, 5, 'mm'),
    ('Displacer stroke',   'S_displacer', 30, 200, 10, 'mm'),
    ('Power piston diameter', 'D_power', 30, 150, 5, 'mm'),
    ('Power piston stroke',   'S_power', 20, 150, 10, 'mm'),
    ('Phase angle',        'phi_deg', 60, 120, 15, '°'),
    ('Regen diameter',     'D_r', 20, 100, 10, 'mm'),
    ('Regen length',       'L_r', 50, 400, 50, 'mm'),
    ('Wire diameter',      'd_wire', 0.5, 3.0, 0.5, 'mm'),
    ('Hot temperature',    'T_h', 573, 1273, 100, 'K'),
    ('Mean pressure',      'P_mean_bar', 1.0, 30.0, 5.0, 'bar'),
    ('Frequency',          'f', 5, 50, 5, 'Hz'),
    ('Regen effectiveness','eps_reg', 0.5, 0.99, 0.05, ''),
    ('Mech efficiency',    'eta_mech', 0.6, 0.95, 0.05, ''),
]


# ============================================================
# CORE PHYSICS
# ============================================================

def to_si(params):
    """Convert user-friendly units to SI for the physics calculations."""
    p = dict(params)
    p['D_displacer'] = params['D_displacer'] * 1e-3
    p['S_displacer'] = params['S_displacer'] * 1e-3
    p['D_power']     = params['D_power'] * 1e-3
    p['S_power']     = params['S_power'] * 1e-3
    p['D_r']         = params['D_r'] * 1e-3
    p['L_r']         = params['L_r'] * 1e-3
    p['d_wire']      = params['d_wire'] * 1e-3
    p['t_wall']      = params['t_wall'] * 1e-3
    p['P_mean']      = params['P_mean_bar'] * 1e5
    return p


def build_geometry(params_si):
    g = dict(params_si)
    A_disp = math.pi * (g['D_displacer']/2)**2
    A_pow  = math.pi * (g['D_power']/2)**2
    g['V_swe'] = A_disp * g['S_displacer']
    g['V_swc'] = A_pow  * g['S_power']
    g['phi']   = math.radians(g['phi_deg'])
    g['V_r_only'] = math.pi * (g['D_r']/2)**2 * g['L_r'] * g['porosity']
    g['V_k'] = g.get('V_loop_cold', PROTOTYPE['V_loop_cold'])
    g['V_h'] = g.get('V_loop_hot', PROTOTYPE['V_loop_hot'])
    g['V_r'] = g['V_r_only']
    g['V_r_lumped'] = g['V_k'] + g['V_r_only'] + g['V_h']
    g['V_cle'] = g.get('V_cle', PROTOTYPE['V_cle'])
    g['V_clc'] = g.get('V_clc', PROTOTYPE['V_clc'])
    return g


def schmidt_cycle(geom, gas, P_target):
    theta = np.deg2rad(np.arange(361))
    V_e = geom['V_cle'] + (geom['V_swe']/2)*(1 + np.cos(theta))
    V_c = (geom['V_clc']
           + (geom['V_swc']/2)*(1 + np.cos(theta - geom['phi']))
           + (geom['V_swe']/2)*(1 - np.cos(theta)))
    T_h, T_k = gas['T_h'], gas['T_k']
    T_r = (T_h - T_k) / np.log(T_h / T_k)
    Sigma = V_c/T_k + geom['V_r_lumped']/T_r + V_e/T_h
    M = P_target / (gas['R'] * (1.0/Sigma).mean())
    P = M * gas['R'] / Sigma
    return dict(theta=theta, V_e=V_e, V_c=V_c, P=P,
                T_c=np.full_like(theta, T_k), T_e=np.full_like(theta, T_h),
                M=M, T_r=T_r, model='Schmidt (Isothermal)')


def adiabatic_rhs(theta, y, geom, gas, M):
    P, T_c, T_e, m_c, m_e = y
    if P <= 0 or T_c <= 0 or T_e <= 0 or m_c <= 0 or m_e <= 0:
        return [0, 0, 0, 0, 0]
    V_e = geom['V_cle'] + (geom['V_swe']/2)*(1 + np.cos(theta))
    V_c = (geom['V_clc']
           + (geom['V_swc']/2)*(1 + np.cos(theta - geom['phi']))
           + (geom['V_swe']/2)*(1 - np.cos(theta)))
    dV_e_dth = -(geom['V_swe']/2)*np.sin(theta)
    dV_c_dth = (-(geom['V_swc']/2)*np.sin(theta - geom['phi'])
                + (geom['V_swe']/2)*np.sin(theta))
    g = gas['gamma']
    T_h, T_k, T_r = gas['T_h'], gas['T_k'], gas['T_r']
    R_g = gas['R']
    T_ck = T_c if dV_c_dth < 0 else T_k
    T_he = T_h if dV_e_dth > 0 else T_e
    num = -g * P * (dV_c_dth/T_ck + dV_e_dth/T_he)
    den = (V_c/T_ck + g * (geom['V_k']/T_k + geom['V_r']/T_r + geom['V_h']/T_h) + V_e/T_he)
    dP_dth = num / den
    dmc_dth = (P*dV_c_dth + V_c*dP_dth/g) / (R_g * T_ck)
    dme_dth = (P*dV_e_dth + V_e*dP_dth/g) / (R_g * T_he)
    dTc_dth = T_c * (dP_dth/P + dV_c_dth/V_c - dmc_dth/m_c)
    dTe_dth = T_e * (dP_dth/P + dV_e_dth/V_e - dme_dth/m_e)
    return [dP_dth, dTc_dth, dTe_dth, dmc_dth, dme_dth]


def _run_adiabatic_once(geom, gas, M_init, max_cycles=15, tol=0.5):
    """Single adiabatic run with given M, no outer iteration."""
    T_h, T_k = gas['T_h'], gas['T_k']
    theta = np.deg2rad(np.arange(361))
    V_e_arr = geom['V_cle'] + (geom['V_swe']/2)*(1 + np.cos(theta))
    V_c_arr = (geom['V_clc']
               + (geom['V_swc']/2)*(1 + np.cos(theta - geom['phi']))
               + (geom['V_swe']/2)*(1 - np.cos(theta)))
    # First guess at pressure: P = M*R*T_avg/V_avg
    Sigma_avg = V_c_arr[0]/T_k + geom['V_r']/gas['T_r'] + V_e_arr[0]/T_h
    P0 = M_init * gas['R'] / Sigma_avg
    Tc0 = T_k; Te0 = T_h
    mc0 = P0 * V_c_arr[0] / (gas['R'] * Tc0)
    me0 = P0 * V_e_arr[0] / (gas['R'] * Te0)
    y0 = [P0, Tc0, Te0, mc0, me0]
    theta_eval = np.linspace(0, 2*np.pi, 361)
    for cycle in range(max_cycles):
        try:
            sol = solve_ivp(adiabatic_rhs, (0, 2*np.pi), y0, t_eval=theta_eval,
                            args=(geom, gas, M_init), method='RK45',
                            rtol=1e-8, atol=1e-11, max_step=np.deg2rad(0.5))
            if not sol.success: return None
        except Exception:
            return None
        P_arr, Tc_arr, Te_arr, mc_arr, me_arr = sol.y
        dTc = Tc_arr[-1] - y0[1]; dTe = Te_arr[-1] - y0[2]
        if abs(dTc) < tol and abs(dTe) < tol and cycle > 0:
            break
        y0 = [P_arr[-1], Tc_arr[-1], Te_arr[-1], mc_arr[-1], me_arr[-1]]
    return dict(theta=theta_eval, V_e=V_e_arr, V_c=V_c_arr, P=P_arr,
                T_c=Tc_arr, T_e=Te_arr, M=M_init, T_r=gas['T_r'],
                model='Adiabatic (RK45)', cycles_to_converge=cycle+1)


def adiabatic_cycle(geom, gas, P_target, max_pressure_iters=4, p_tol_pct=1.0):
    """
    Iterative adiabatic: adjust M until P_mean ≈ P_target within p_tol_pct%.
    """
    T_h, T_k = gas['T_h'], gas['T_k']
    T_r = (T_h - T_k) / np.log(T_h / T_k)
    gas['T_r'] = T_r
    # Initial M guess using Schmidt-style closed form
    theta = np.deg2rad(np.arange(361))
    V_e_arr = geom['V_cle'] + (geom['V_swe']/2)*(1 + np.cos(theta))
    V_c_arr = (geom['V_clc']
               + (geom['V_swc']/2)*(1 + np.cos(theta - geom['phi']))
               + (geom['V_swe']/2)*(1 - np.cos(theta)))
    Sigma = V_c_arr/T_k + geom['V_r']/T_r + V_e_arr/T_h
    M = P_target / (gas['R'] * (1.0/Sigma).mean())

    result = None
    for outer_iter in range(max_pressure_iters):
        result = _run_adiabatic_once(geom, gas, M)
        if result is None:
            return None
        P_mean_actual = float(result['P'].mean())
        # If within tolerance, done
        if abs(P_mean_actual - P_target) / P_target * 100 < p_tol_pct:
            break
        # Otherwise scale M proportionally
        M = M * (P_target / P_mean_actual)

    # Re-run one more time at the final M to ensure consistency
    result = _run_adiabatic_once(geom, gas, M)
    if result is not None:
        result['pressure_iterations'] = outer_iter + 1
    return result


def compute_losses(result, geom, gas, params_si, losses_flags):
    theta = result['theta']
    V_e, V_c, P = result['V_e'], result['V_c'], result['P']
    M = result['M']
    dV_e = np.diff(V_e); dV_c = np.diff(V_c)
    P_mid = 0.5*(P[:-1] + P[1:])
    W_e = float(np.sum(P_mid * dV_e))
    W_c = float(np.sum(P_mid * dV_c))
    W_cycle = W_e + W_c
    P_mean = float(P.mean())
    out = {'W_cycle': W_cycle, 'W_e_cycle': W_e, 'W_c_cycle': W_c,
           'P_mean': P_mean, 'P_max': float(P.max()), 'P_min': float(P.min()),
           'T_e_max': float(result['T_e'].max()), 'T_e_min': float(result['T_e'].min()),
           'T_c_max': float(result['T_c'].max()), 'T_c_min': float(result['T_c'].min()),
           'M': M}
    if losses_flags['flow']:
        eps_por = geom['porosity']; d_wire = geom['d_wire']
        A_reg = math.pi*(geom['D_r']/2)**2
        R_v = 150*(1-eps_por)**2/(eps_por**3 * d_wire**2)
        R_i = 1.75*(1-eps_por)/(eps_por**3 * d_wire)
        V_dot = dV_e * 360 * params_si['f']
        u = V_dot / (A_reg * eps_por)
        rho = P_mid / (gas['R'] * result['T_r'])
        dP_drop = geom['L_r']*(R_v*gas['mu']*u + R_i*rho*u*np.abs(u))
        out['W_pump'] = float(np.sum(np.abs(dP_drop * dV_e)))
    else:
        out['W_pump'] = 0.0
    W_after_flow = W_cycle - out['W_pump']
    if losses_flags['leakage']:
        out['W_leak'] = W_after_flow * params_si['C_leak'] * (P_mean / params_si['P_ref'])
    else:
        out['W_leak'] = 0.0
    W_after_leak = W_after_flow - out['W_leak']
    if losses_flags['mechanical']:
        out['W_mech_loss'] = W_after_leak * (1 - params_si['eta_mech'])
    else:
        out['W_mech_loss'] = 0.0
    out['W_shaft'] = W_after_leak - out['W_mech_loss']
    out['P_brake'] = out['W_shaft'] * params_si['f']
    out['Q_e'] = W_e
    out['Q_miss'] = M*gas['Cv']*(gas['T_h']-gas['T_k'])*(1-params_si['eps_reg']) if losses_flags['regen_imp'] else 0.0
    if losses_flags['wall_cond']:
        D_outer = geom['D_r'] + 2*params_si['t_wall']
        A_ring = math.pi*((D_outer/2)**2 - (geom['D_r']/2)**2)
        out['Q_cond_W'] = params_si['k_metal']*A_ring*(gas['T_h']-gas['T_k'])/geom['L_r']
        out['Q_cond'] = out['Q_cond_W']/params_si['f']
    else:
        out['Q_cond'] = 0.0; out['Q_cond_W'] = 0.0
    out['Q_shuttle'] = 0.0
    out['Q_in'] = out['Q_e'] + out['Q_miss'] + out['Q_cond'] + out['Q_shuttle']
    out['Q_in_W'] = out['Q_in'] * params_si['f']
    out['eta_brake'] = out['W_shaft']/out['Q_in'] if out['Q_in'] > 0 else 0
    out['eta_carnot'] = 1 - gas['T_k']/gas['T_h']
    out['frac_carnot'] = out['eta_brake']/out['eta_carnot'] if out['eta_carnot'] > 0 else 0
    return out


def simulate(params, model='schmidt', losses_flags=None):
    if losses_flags is None:
        losses_flags = dict(flow=True, regen_imp=True, mechanical=True,
                            wall_cond=True, leakage=True, shuttle=False)
    params_si = to_si(params)
    geom = build_geometry(params_si)
    gas = dict(GASES[params['gas']])
    gas['T_h'] = params['T_h']; gas['T_k'] = params['T_k']
    P_target = params_si['P_mean']
    if model == 'schmidt':
        result = schmidt_cycle(geom, gas, P_target)
    else:
        result = adiabatic_cycle(geom, gas, P_target)
        if result is None:
            return None
    losses_out = compute_losses(result, geom, gas, params_si, losses_flags)
    return dict(result=result, losses=losses_out, geom=geom, gas=gas,
                params=params, params_si=params_si,
                losses_flags=losses_flags, model=model)


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(page_title="Stirling Engine Simulator", layout="wide",
                   initial_sidebar_state="expanded")

# Initialize session state with prototype values on first run
if 'initialized' not in st.session_state:
    for key, value in PROTOTYPE.items():
        st.session_state[key] = value
    st.session_state.initialized = True

# Reset callback
def reset_to_prototype():
    for key, value in PROTOTYPE.items():
        st.session_state[key] = value

# ---- Header ----
col_title, col_reset = st.columns([4, 1])
with col_title:
    st.title("🔥 Stirling Engine Simulator")
    st.caption("Gamma-type engine — interactive analysis and optimization")
with col_reset:
    st.write("")
    st.write("")
    if st.button("🔄 Reset to Prototype", type="secondary"):
        reset_to_prototype()
        st.rerun()

# ---- Tabs ----
tab_analysis, tab_optimize = st.tabs(["📊 Analysis", "🎯 Optimization"])

# ============================================================
# TAB 1: ANALYSIS
# ============================================================
with tab_analysis:
    # Sidebar with all parameters
    with st.sidebar:
        st.header("⚙️ Parameters")

        st.subheader("Geometry")
        st.number_input("Displacer diameter [mm]", 30.0, 200.0,
                        step=1.0, key='D_displacer')
        st.number_input("Displacer stroke [mm]", 20.0, 300.0,
                        step=1.0, key='S_displacer')
        st.number_input("Power piston diameter [mm]", 30.0, 200.0,
                        step=1.0, key='D_power')
        st.number_input("Power piston stroke [mm]", 20.0, 300.0,
                        step=1.0, key='S_power')
        st.slider("Phase angle [°]", 30, 150, step=5, key='phi_deg')

        st.subheader("Regenerator")
        st.number_input("Regen diameter [mm]", 10.0, 150.0,
                        step=1.0, key='D_r')
        st.number_input("Regen length [mm]", 20.0, 500.0,
                        step=5.0, key='L_r')
        st.number_input("Wire diameter [mm]", 0.05, 5.0,
                        step=0.05, key='d_wire')
        st.slider("Porosity", 0.5, 0.99, step=0.01, key='porosity')

        st.subheader("Operating Conditions")
        st.selectbox("Working gas", ["Air", "Helium", "Hydrogen"], key='gas')
        st.number_input("Hot temperature T_h [K]", 400, 1500,
                        step=10, key='T_h')
        st.number_input("Cold temperature T_k [K]", 250, 400,
                        step=5, key='T_k')
        st.number_input("Mean pressure [bar]", 0.5, 50.0,
                        step=0.5, key='P_mean_bar')
        st.number_input("Frequency [Hz]", 1, 100, step=1, key='f')

        st.subheader("Losses")
        flow_loss = st.checkbox("Regen flow loss (Ergun)", True)
        regen_imp = st.checkbox("Regen imperfection", True)
        mech_loss = st.checkbox("Mech friction", True)
        wall_cond = st.checkbox("Wall conduction", True)
        leak_loss = st.checkbox("Seal leakage", True)
        st.slider("ε_reg", 0.5, 0.99, step=0.01, key='eps_reg')
        st.slider("η_mech", 0.5, 0.99, step=0.01, key='eta_mech')
        st.slider("C_leak", 0.0, 0.2, step=0.01, key='C_leak')

        st.subheader("Model")
        model_choice = st.radio(
            "Which model?",
            ["Schmidt (isothermal)",
             "Adiabatic (RK45 + P_mean iteration)",
             "BOTH — compare side-by-side"],
            index=2
        )

    # Build params from session state
    params = {k: st.session_state[k] for k in PROTOTYPE if k in st.session_state}
    # Fill in fixed values
    for k in ('V_loop_cold','V_loop_hot','V_cle','V_clc','L_displacer','gap','P_ref','k_metal','t_wall'):
        params[k] = PROTOTYPE[k]

    losses_flags = dict(flow=flow_loss, regen_imp=regen_imp, mechanical=mech_loss,
                        wall_cond=wall_cond, leakage=leak_loss, shuttle=False)
    run_both = model_choice.startswith("BOTH")

    # Run simulation(s)
    with st.spinner("Computing..."):
        if run_both:
            sim_s = simulate(params, model='schmidt', losses_flags=losses_flags)
            sim_a = simulate(params, model='adiabatic', losses_flags=losses_flags)
        else:
            primary_model = 'schmidt' if model_choice.startswith("Schmidt") else 'adiabatic'
            sim = simulate(params, model=primary_model, losses_flags=losses_flags)

    # Display results
    st.header("📊 Results")

    if run_both and sim_s and sim_a:
        cols = st.columns(4)
        cols[0].metric("Schmidt Power", f"{sim_s['losses']['P_brake']:.1f} W")
        cols[1].metric("Adiabatic Power", f"{sim_a['losses']['P_brake']:.1f} W")
        cols[2].metric("Schmidt η", f"{sim_s['losses']['eta_brake']*100:.2f}%")
        cols[3].metric("Adiabatic η", f"{sim_a['losses']['eta_brake']*100:.2f}%")

        optimism = sim_s['losses']['P_brake'] / sim_a['losses']['P_brake'] if sim_a['losses']['P_brake'] > 0 else 0
        iters = sim_a['result'].get('pressure_iterations', '?')
        st.info(f"**Isothermal Optimism Factor: {optimism:.3f}** "
                f"— Schmidt predicts {(optimism-1)*100:+.1f}% power vs Adiabatic. "
                f"Adiabatic: {iters} pressure iterations, "
                f"{sim_a['result']['cycles_to_converge']} cycles to thermal convergence.")

        c1, c2 = st.columns(2)
        for col, (name, sim) in zip([c1, c2], [('Schmidt', sim_s), ('Adiabatic', sim_a)]):
            L = sim['losses']
            with col:
                st.markdown(f"**{name}**")
                st.text(f"""M           : {L['M']*1000:.3f} g
P_mean      : {L['P_mean']/1e5:.3f} bar  (target {params['P_mean_bar']:.2f})
P_max/P_min : {L['P_max']/1e5:.2f} / {L['P_min']/1e5:.2f} bar
W_cycle     : {L['W_cycle']:.4f} J
W_shaft     : {L['W_shaft']:.4f} J
Brake Power : {L['P_brake']:.2f} W
Q_in        : {L['Q_in_W']:.2f} W
Efficiency  : {L['eta_brake']*100:.3f} %
% Carnot    : {L['frac_carnot']*100:.2f} %""")
    elif not run_both and sim:
        L = sim['losses']
        cols = st.columns(4)
        cols[0].metric("Brake Power", f"{L['P_brake']:.1f} W")
        cols[1].metric("Efficiency", f"{L['eta_brake']*100:.2f}%")
        cols[2].metric("Carnot", f"{L['eta_carnot']*100:.2f}%")
        cols[3].metric("Mass", f"{L['M']*1000:.2f} g")

        if sim['model'] == 'adiabatic':
            iters = sim['result'].get('pressure_iterations', '?')
            st.info(f"Adiabatic converged in {sim['result']['cycles_to_converge']} cycles, "
                    f"with {iters} pressure iterations. "
                    f"P_mean = {L['P_mean']/1e5:.3f} bar (target {params['P_mean_bar']:.2f}).")

        st.text(f"""WORK PER CYCLE [J]
  W_cycle (raw)     : {L['W_cycle']:.4f}
  - W_pump          : -{L['W_pump']:.4f}
  - W_leak          : -{L['W_leak']:.4f}
  - W_mech_loss     : -{L['W_mech_loss']:.4f}
  = W_shaft         : {L['W_shaft']:.4f}

HEAT INPUT [W]
  Q_e               : {L['Q_e']*params['f']:.2f}
  + Q_miss          : {L['Q_miss']*params['f']:.2f}
  + Q_cond          : {L['Q_cond_W']:.2f}
  = Q_in_total      : {L['Q_in_W']:.2f}""")

    # Plots
    st.subheader("📈 Plots")
    if run_both and sim_s and sim_a:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        Vs = (sim_s['result']['V_e'] + sim_s['result']['V_c']) * 1e6
        Va = (sim_a['result']['V_e'] + sim_a['result']['V_c']) * 1e6
        th = np.rad2deg(sim_s['result']['theta'])
        axes[0,0].plot(Vs, sim_s['result']['P']/1e5, color='#1F4E79', lw=2, label='Schmidt')
        axes[0,0].fill(Vs, sim_s['result']['P']/1e5, alpha=0.1, color='#1F4E79')
        axes[0,0].plot(Va, sim_a['result']['P']/1e5, color='#C00000', lw=2, label='Adiabatic')
        axes[0,0].fill(Va, sim_a['result']['P']/1e5, alpha=0.1, color='#C00000')
        axes[0,0].set_xlabel('Total Volume [cm³]'); axes[0,0].set_ylabel('Pressure [bar]')
        axes[0,0].set_title('P-V Diagram'); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)
        axes[0,1].plot(th, sim_s['result']['P']/1e5, color='#1F4E79', lw=2, label='Schmidt')
        axes[0,1].plot(th, sim_a['result']['P']/1e5, color='#C00000', lw=2, label='Adiabatic')
        axes[0,1].set_xlabel('Crank angle [°]'); axes[0,1].set_ylabel('Pressure [bar]')
        axes[0,1].set_title('Pressure vs θ'); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)
        axes[1,0].axhline(params['T_h'], color='#1F4E79', ls='--', lw=2, label='Schmidt T_e=T_h')
        axes[1,0].axhline(params['T_k'], color='#1F4E79', ls=':',  lw=2, label='Schmidt T_c=T_k')
        axes[1,0].plot(th, sim_a['result']['T_e'], color='#C00000', lw=2, label='Adia T_e')
        axes[1,0].plot(th, sim_a['result']['T_c'], color='#ED7D31', lw=2, label='Adia T_c')
        axes[1,0].set_xlabel('Crank angle [°]'); axes[1,0].set_ylabel('Temperature [K]')
        axes[1,0].set_title('Gas Temperatures'); axes[1,0].legend(fontsize=8); axes[1,0].grid(alpha=0.3)
        metrics = ['P_brake [W]', 'η [%]', '% Carnot']
        s_vals = [sim_s['losses']['P_brake'], sim_s['losses']['eta_brake']*100, sim_s['losses']['frac_carnot']*100]
        a_vals = [sim_a['losses']['P_brake'], sim_a['losses']['eta_brake']*100, sim_a['losses']['frac_carnot']*100]
        x = np.arange(len(metrics))
        axes[1,1].bar(x - 0.2, s_vals, 0.4, label='Schmidt', color='#1F4E79')
        axes[1,1].bar(x + 0.2, a_vals, 0.4, label='Adiabatic', color='#C00000')
        axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(metrics)
        axes[1,1].set_title('Performance'); axes[1,1].legend(); axes[1,1].grid(alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
    elif not run_both and sim:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        V_total = (sim['result']['V_e'] + sim['result']['V_c']) * 1e6
        th = np.rad2deg(sim['result']['theta'])
        axes[0,0].plot(V_total, sim['result']['P']/1e5, color='#1F4E79', lw=2)
        axes[0,0].fill(V_total, sim['result']['P']/1e5, alpha=0.15, color='#1F4E79')
        axes[0,0].set_xlabel('Total Volume [cm³]'); axes[0,0].set_ylabel('Pressure [bar]')
        axes[0,0].set_title('P-V Diagram'); axes[0,0].grid(alpha=0.3)
        axes[0,1].plot(th, sim['result']['P']/1e5, color='#C00000', lw=2)
        axes[0,1].set_xlabel('Crank angle [°]'); axes[0,1].set_ylabel('Pressure [bar]')
        axes[0,1].set_title('Pressure vs θ'); axes[0,1].grid(alpha=0.3)
        axes[1,0].plot(th, sim['result']['V_e']*1e6, color='#C00000', lw=2, label='V_e')
        axes[1,0].plot(th, sim['result']['V_c']*1e6, color='#1F4E79', lw=2, label='V_c')
        axes[1,0].set_xlabel('Crank angle [°]'); axes[1,0].set_ylabel('Volume [cm³]')
        axes[1,0].set_title('Cylinder Volumes'); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)
        L = sim['losses']
        labels = ['W_cycle']; values = [L['W_cycle']]
        if losses_flags['flow']:        labels.append('-W_pump'); values.append(-L['W_pump'])
        if losses_flags['leakage']:     labels.append('-W_leak'); values.append(-L['W_leak'])
        if losses_flags['mechanical']:  labels.append('-W_mech'); values.append(-L['W_mech_loss'])
        labels.append('W_shaft'); values.append(L['W_shaft'])
        colors = ['#70AD47'] + ['#ED7D31']*(len(labels)-2) + ['#1F4E79']
        axes[1,1].bar(labels, values, color=colors)
        axes[1,1].axhline(0, color='black', lw=0.6)
        axes[1,1].set_ylabel('Work [J/cycle]'); axes[1,1].set_title('Work Waterfall')
        axes[1,1].grid(alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)

# ============================================================
# TAB 2: OPTIMIZATION
# ============================================================
with tab_optimize:
    st.header("🎯 Optimization")
    st.markdown("""
    Lock parameters you want fixed (✓), leave others unlocked to be optimized.
    Set the range for each open parameter. Then click **Run Optimization**.
    """)

    st.subheader("Lock / Open parameters")
    open_specs = []
    lock_state = {}

    n_cols = 2
    cols = st.columns(n_cols)
    for i, (display_name, key, pmin, pmax, step, units) in enumerate(OPTIMIZABLE_PARAMS):
        col = cols[i % n_cols]
        with col:
            with st.container():
                current = st.session_state.get(key, PROTOTYPE.get(key, pmin))
                locked = st.checkbox(f"🔒 Lock **{display_name}** = {current} {units}",
                                     value=True, key=f"lock_{key}")
                lock_state[key] = locked
                if not locked:
                    cc1, cc2, cc3 = st.columns(3)
                    pmn = cc1.number_input(f"min", value=float(pmin), step=float(step),
                                            key=f"min_{key}", label_visibility="collapsed")
                    pmx = cc2.number_input(f"max", value=float(pmax), step=float(step),
                                            key=f"max_{key}", label_visibility="collapsed")
                    pst = cc3.number_input(f"step", value=float(step), step=float(step)/2,
                                            key=f"step_{key}", label_visibility="collapsed")
                    values = list(np.arange(pmn, pmx + pst/2, pst))
                    open_specs.append((key, values))

    st.subheader("Objective")
    obj_choice = st.radio("Maximize:",
                          ["Brake Power", "Brake Efficiency",
                           "Power × Efficiency (balanced)"],
                          horizontal=True)
    obj_map = {'Brake Power': 'power',
               'Brake Efficiency': 'efficiency',
               'Power × Efficiency (balanced)': 'balanced'}
    objective = obj_map[obj_choice]

    st.subheader("Model")
    opt_model = st.radio("Run optimization with:",
                         ["Schmidt (fast, recommended)",
                          "Adiabatic (slower, more accurate)"],
                         horizontal=True)
    opt_model_key = 'schmidt' if opt_model.startswith("Schmidt") else 'adiabatic'

    # Show estimated grid size
    if open_specs:
        total = 1
        for _, vals in open_specs:
            total *= len(vals)
        st.info(f"**Search space: {total:,} configurations** "
                f"({len(open_specs)} open parameters)")
        if total > 5000:
            st.warning(f"⚠️ Large search space — may take a few minutes. "
                       f"Consider increasing step sizes.")
    else:
        st.warning("No parameters open — uncheck at least one parameter above.")

    if st.button("🚀 Run Optimization", type="primary",
                 disabled=(len(open_specs) == 0)):
        # Build base params (use current values for locked params)
        base_params = {k: st.session_state[k] for k in PROTOTYPE if k in st.session_state}
        for k in ('V_loop_cold','V_loop_hot','V_cle','V_clc','L_displacer',
                  'gap','P_ref','k_metal','t_wall'):
            base_params[k] = PROTOTYPE[k]

        losses_flags_opt = dict(flow=True, regen_imp=True, mechanical=True,
                                wall_cond=True, leakage=True, shuttle=False)

        # Grid search
        keys = [s[0] for s in open_specs]
        grids = [s[1] for s in open_specs]

        progress = st.progress(0, text="Running grid search...")
        results_list = []
        total_combos = 1
        for g in grids: total_combos *= len(g)
        count = 0

        best_score = -float('inf')
        best_params = None
        best_sim = None

        for combo in product(*grids):
            p = dict(base_params)
            for k, v in zip(keys, combo):
                p[k] = v
            sim_opt = simulate(p, model=opt_model_key, losses_flags=losses_flags_opt)
            count += 1
            if count % max(1, total_combos // 100) == 0:
                progress.progress(count / total_combos,
                                  text=f"Evaluated {count}/{total_combos}")
            if sim_opt is None:
                continue
            L = sim_opt['losses']
            if L['W_shaft'] <= 0 or L['Q_in'] <= 0:
                continue
            if objective == 'power':
                score = L['P_brake']
            elif objective == 'efficiency':
                score = L['eta_brake']
            else:  # balanced
                score = L['P_brake'] * L['eta_brake']
            if score > best_score:
                best_score = score
                best_params = p
                best_sim = sim_opt
                results_list.append((score, p, L))

        progress.empty()

        if best_sim is None:
            st.error("❌ No valid solutions found in the search grid.")
        else:
            st.success(f"✅ Optimization complete! Best score: {best_score:.4f}")

            # Apply the optimum to session state
            if st.button("📥 Apply Optimum to Sliders"):
                for k in keys:
                    st.session_state[k] = best_params[k]
                st.rerun()

            # Show comparison
            current_params = {k: st.session_state[k] for k in PROTOTYPE if k in st.session_state}
            for k in ('V_loop_cold','V_loop_hot','V_cle','V_clc','L_displacer',
                      'gap','P_ref','k_metal','t_wall'):
                current_params[k] = PROTOTYPE[k]
            sim_current = simulate(current_params, model=opt_model_key, losses_flags=losses_flags_opt)

            st.subheader("Result")
            c1, c2, c3 = st.columns(3)
            c1.metric("Best Power",      f"{best_sim['losses']['P_brake']:.2f} W",
                      delta=f"{best_sim['losses']['P_brake'] - sim_current['losses']['P_brake']:+.2f} W")
            c2.metric("Best Efficiency", f"{best_sim['losses']['eta_brake']*100:.2f}%",
                      delta=f"{(best_sim['losses']['eta_brake'] - sim_current['losses']['eta_brake'])*100:+.2f}%")
            c3.metric("Best Score",      f"{best_score:.4f}")

            st.subheader("Optimal parameters")
            opt_df_rows = []
            for display_name, key, _, _, _, units in OPTIMIZABLE_PARAMS:
                if key in keys:
                    cur_val = current_params.get(key, PROTOTYPE[key])
                    new_val = best_params[key]
                    opt_df_rows.append({
                        'Parameter': display_name,
                        'Current': f"{cur_val} {units}",
                        'Optimum': f"{new_val} {units}",
                        'Change': f"{((new_val - cur_val)/cur_val*100 if cur_val != 0 else 0):+.1f} %" if isinstance(cur_val, (int, float)) and isinstance(new_val, (int, float)) else "—"
                    })
            st.table(opt_df_rows)

            # P-V comparison
            st.subheader("P-V Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            Vc = (sim_current['result']['V_e'] + sim_current['result']['V_c']) * 1e6
            Vb = (best_sim['result']['V_e']    + best_sim['result']['V_c']) * 1e6
            ax.plot(Vc, sim_current['result']['P']/1e5, color='#7F7F7F', lw=2, label='Current')
            ax.fill(Vc, sim_current['result']['P']/1e5, alpha=0.1, color='#7F7F7F')
            ax.plot(Vb, best_sim['result']['P']/1e5, color='#70AD47', lw=2.5, label='Optimum')
            ax.fill(Vb, best_sim['result']['P']/1e5, alpha=0.15, color='#70AD47')
            ax.set_xlabel('Total Volume [cm³]'); ax.set_ylabel('Pressure [bar]')
            ax.set_title('P-V: Current vs Optimum'); ax.legend(); ax.grid(alpha=0.3)
            st.pyplot(fig)

st.caption("💡 Tip: change parameters on the left; the Analysis tab updates automatically.")

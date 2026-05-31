"""
Stirling Engine Simulator — Streamlit App
==========================================
Simple web-based UI for the Stirling engine simulator.
No more terminal questions — just sliders and buttons.

To run:
    pip install streamlit
    streamlit run stirling_app.py

The browser will open automatically with the app.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
import streamlit as st

# ============================================================
# CORE PHYSICS (same as v2/v3 — no changes to the engine)
# ============================================================

PROTOTYPE = {
    'D_displacer':  0.075, 'S_displacer':  0.1015, 'L_displacer':  0.235,
    'D_power':      0.0656, 'S_power':      0.0616, 'phi_deg':      90.0,
    'D_r':          0.040, 'L_r':          0.236, 'd_wire':       0.001,
    'porosity':     0.9, 'gap':          0.00025,
    'gas':          'Air', 'T_h':          873.0, 'T_k':          300.0,
    'P_mean':       1.0e5, 'f':            10.0,
    'eps_reg':      0.85, 'eta_mech':     0.85, 'C_leak':       0.02,
    'P_ref':        1.0e5, 'k_metal':      26.0, 't_wall':        0.002,
    'V_loop_cold':  6.7278e-5, 'V_loop_hot':   7.1127e-5,
    'V_cle':        1.7671e-5, 'V_clc':        7.2041e-5,
}

GASES = {
    'Air':      {'R': 287,  'Cv': 718,   'Cp': 1005,  'gamma': 1.4,   'mu': 2.7e-5, 'k_gas': 0.04},
    'Helium':   {'R': 2077, 'Cv': 3116,  'Cp': 5193,  'gamma': 1.667, 'mu': 3.4e-5, 'k_gas': 0.18},
    'Hydrogen': {'R': 4124, 'Cv': 10160, 'Cp': 14284, 'gamma': 1.406, 'mu': 1.4e-5, 'k_gas': 0.22},
}


def build_geometry(params):
    g = dict(params)
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
    den = (V_c/T_ck
           + g * (geom['V_k']/T_k + geom['V_r']/T_r + geom['V_h']/T_h)
           + V_e/T_he)
    dP_dth = num / den
    dmc_dth = (P*dV_c_dth + V_c*dP_dth/g) / (R_g * T_ck)
    dme_dth = (P*dV_e_dth + V_e*dP_dth/g) / (R_g * T_he)
    dTc_dth = T_c * (dP_dth/P + dV_c_dth/V_c - dmc_dth/m_c)
    dTe_dth = T_e * (dP_dth/P + dV_e_dth/V_e - dme_dth/m_e)
    return [dP_dth, dTc_dth, dTe_dth, dmc_dth, dme_dth]


def adiabatic_cycle(geom, gas, P_target, max_cycles=15, tol=0.5):
    T_h, T_k = gas['T_h'], gas['T_k']
    T_r = (T_h - T_k) / np.log(T_h / T_k)
    gas['T_r'] = T_r
    theta = np.deg2rad(np.arange(361))
    V_e_arr = geom['V_cle'] + (geom['V_swe']/2)*(1 + np.cos(theta))
    V_c_arr = (geom['V_clc']
               + (geom['V_swc']/2)*(1 + np.cos(theta - geom['phi']))
               + (geom['V_swe']/2)*(1 - np.cos(theta)))
    Sigma = V_c_arr/T_k + geom['V_r']/T_r + V_e_arr/T_h
    M = P_target / (gas['R'] * (1.0/Sigma).mean())
    P0 = P_target; Tc0 = T_k; Te0 = T_h
    mc0 = P0 * V_c_arr[0] / (gas['R'] * Tc0)
    me0 = P0 * V_e_arr[0] / (gas['R'] * Te0)
    y0 = [P0, Tc0, Te0, mc0, me0]
    theta_eval = np.linspace(0, 2*np.pi, 361)
    for cycle in range(max_cycles):
        try:
            sol = solve_ivp(adiabatic_rhs, (0, 2*np.pi), y0, t_eval=theta_eval,
                            args=(geom, gas, M), method='RK45',
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
                T_c=Tc_arr, T_e=Te_arr, M=M, T_r=T_r,
                model='Adiabatic (RK45)', cycles_to_converge=cycle+1)


def compute_losses(result, geom, gas, params, losses_flags):
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
        V_dot = dV_e * 360 * params['f']
        u = V_dot / (A_reg * eps_por)
        rho = P_mid / (gas['R'] * result['T_r'])
        dP_drop = geom['L_r']*(R_v*gas['mu']*u + R_i*rho*u*np.abs(u))
        out['W_pump'] = float(np.sum(np.abs(dP_drop * dV_e)))
    else:
        out['W_pump'] = 0.0
    W_after_flow = W_cycle - out['W_pump']
    if losses_flags['leakage']:
        out['W_leak'] = W_after_flow * params['C_leak'] * (P_mean / params['P_ref'])
    else:
        out['W_leak'] = 0.0
    W_after_leak = W_after_flow - out['W_leak']
    if losses_flags['mechanical']:
        out['W_mech_loss'] = W_after_leak * (1 - params['eta_mech'])
    else:
        out['W_mech_loss'] = 0.0
    out['W_shaft'] = W_after_leak - out['W_mech_loss']
    out['P_brake'] = out['W_shaft'] * params['f']
    out['Q_e'] = W_e
    out['Q_miss'] = M*gas['Cv']*(gas['T_h']-gas['T_k'])*(1-params['eps_reg']) if losses_flags['regen_imp'] else 0.0
    if losses_flags['wall_cond']:
        D_outer = geom['D_r'] + 2*params['t_wall']
        A_ring = math.pi*((D_outer/2)**2 - (geom['D_r']/2)**2)
        out['Q_cond_W'] = params['k_metal']*A_ring*(gas['T_h']-gas['T_k'])/geom['L_r']
        out['Q_cond'] = out['Q_cond_W']/params['f']
    else:
        out['Q_cond'] = 0.0; out['Q_cond_W'] = 0.0
    out['Q_shuttle'] = 0.0  # disabled until disc-rod displacer model is added
    out['Q_in'] = out['Q_e'] + out['Q_miss'] + out['Q_cond'] + out['Q_shuttle']
    out['Q_in_W'] = out['Q_in'] * params['f']
    out['eta_brake'] = out['W_shaft']/out['Q_in'] if out['Q_in'] > 0 else 0
    out['eta_carnot'] = 1 - gas['T_k']/gas['T_h']
    out['frac_carnot'] = out['eta_brake']/out['eta_carnot'] if out['eta_carnot'] > 0 else 0
    return out


def simulate(params, model='schmidt', losses_flags=None):
    if losses_flags is None:
        losses_flags = dict(flow=True, regen_imp=True, mechanical=True,
                            wall_cond=True, leakage=True, shuttle=False)
    geom = build_geometry(params)
    gas = dict(GASES[params['gas']])
    gas['T_h'] = params['T_h']; gas['T_k'] = params['T_k']
    P_target = params['P_mean']
    if model == 'schmidt':
        result = schmidt_cycle(geom, gas, P_target)
    else:
        result = adiabatic_cycle(geom, gas, P_target)
        if result is None:
            return None
    losses_out = compute_losses(result, geom, gas, params, losses_flags)
    return dict(result=result, losses=losses_out, geom=geom, gas=gas, params=params,
                losses_flags=losses_flags, model=model)


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Stirling Engine Simulator", layout="wide",
                   initial_sidebar_state="expanded")

st.title("🔥 Stirling Engine Simulator")
st.caption("Gamma-type engine analysis — choose your model, set your parameters, see results.")

# ---- SIDEBAR: All parameters ----
with st.sidebar:
    st.header("⚙️ Parameters")

    st.subheader("Geometry")
    D_displacer = st.number_input("Displacer diameter [mm]", 30, 200, 75, 1) * 1e-3
    S_displacer = st.number_input("Displacer stroke [mm]", 20, 300, 102, 1) * 1e-3
    D_power     = st.number_input("Power piston diameter [mm]", 30, 200, 66, 1) * 1e-3
    S_power     = st.number_input("Power piston stroke [mm]", 20, 300, 62, 1) * 1e-3
    phi_deg     = st.slider("Phase angle [°]", 30, 150, 90, 5)

    st.subheader("Regenerator")
    D_r     = st.number_input("Regen diameter [mm]", 10, 150, 40, 1) * 1e-3
    L_r     = st.number_input("Regen length [mm]", 20, 500, 236, 5) * 1e-3
    d_wire  = st.number_input("Wire diameter [mm]", 0.05, 5.0, 1.0, 0.05) * 1e-3
    porosity = st.slider("Porosity", 0.5, 0.99, 0.9, 0.01)

    st.subheader("Operating Conditions")
    gas_name = st.selectbox("Working gas", ["Air", "Helium", "Hydrogen"])
    T_h      = st.number_input("Hot temperature T_h [K]", 400, 1500, 873, 10)
    T_k      = st.number_input("Cold temperature T_k [K]", 250, 400, 300, 5)
    P_mean   = st.number_input("Mean pressure [bar]", 0.5, 50.0, 1.0, 0.5) * 1e5
    f        = st.number_input("Frequency [Hz]", 1, 100, 10, 1)

    st.subheader("Losses")
    flow_loss      = st.checkbox("Regenerator flow loss (Ergun)", True)
    regen_imp      = st.checkbox("Regenerator imperfection", True)
    mech_loss      = st.checkbox("Mechanical friction", True)
    wall_cond      = st.checkbox("Wall heat conduction", True)
    leak_loss      = st.checkbox("Seal leakage", True)

    if regen_imp:
        eps_reg = st.slider("ε_reg (regen effectiveness)", 0.5, 0.99, 0.85, 0.01)
    else:
        eps_reg = 0.85
    if mech_loss:
        eta_mech = st.slider("η_mech (mechanical efficiency)", 0.5, 0.99, 0.85, 0.01)
    else:
        eta_mech = 0.85
    if leak_loss:
        C_leak = st.slider("C_leak", 0.0, 0.2, 0.02, 0.01)
    else:
        C_leak = 0.0

    st.subheader("Model Choice")
    model_choice = st.radio(
        "Which model?",
        ["Schmidt (isothermal, fast)",
         "Adiabatic (RK45, accurate)",
         "BOTH — compare them"],
        index=2
    )

# ---- Build params dict ----
params = dict(PROTOTYPE)
params.update({
    'D_displacer': D_displacer, 'S_displacer': S_displacer,
    'D_power': D_power, 'S_power': S_power, 'phi_deg': phi_deg,
    'D_r': D_r, 'L_r': L_r, 'd_wire': d_wire, 'porosity': porosity,
    'gas': gas_name, 'T_h': T_h, 'T_k': T_k, 'P_mean': P_mean, 'f': f,
    'eps_reg': eps_reg, 'eta_mech': eta_mech, 'C_leak': C_leak,
})

losses_flags = dict(flow=flow_loss, regen_imp=regen_imp, mechanical=mech_loss,
                    wall_cond=wall_cond, leakage=leak_loss, shuttle=False)

run_both = model_choice.startswith("BOTH")
primary_model = 'schmidt' if model_choice.startswith("Schmidt") or run_both else 'adiabatic'

# ---- MAIN AREA: Run and show results ----
st.header("📊 Results")

with st.spinner("Computing..."):
    if run_both:
        sim_s = simulate(params, model='schmidt', losses_flags=losses_flags)
        sim_a = simulate(params, model='adiabatic', losses_flags=losses_flags)
        sims_to_show = {'Schmidt': sim_s, 'Adiabatic': sim_a}
    else:
        sim = simulate(params, model=primary_model, losses_flags=losses_flags)
        sims_to_show = {sim['result']['model']: sim}

# ---- Key metrics ----
if run_both and sim_s and sim_a:
    cols = st.columns(4)
    cols[0].metric("Schmidt Power",  f"{sim_s['losses']['P_brake']:.1f} W")
    cols[1].metric("Adiabatic Power", f"{sim_a['losses']['P_brake']:.1f} W")
    cols[2].metric("Schmidt η",      f"{sim_s['losses']['eta_brake']*100:.2f} %")
    cols[3].metric("Adiabatic η",    f"{sim_a['losses']['eta_brake']*100:.2f} %")

    optimism = sim_s['losses']['P_brake'] / sim_a['losses']['P_brake'] if sim_a['losses']['P_brake'] > 0 else 0
    st.info(f"**Isothermal Optimism Factor: {optimism:.3f}** "
            f"— Schmidt predicts {(optimism-1)*100:+.1f}% power vs Adiabatic. "
            f"Adiabatic converged in {sim_a['result']['cycles_to_converge']} cycles.")
elif not run_both and sim:
    L = sim['losses']
    cols = st.columns(4)
    cols[0].metric("Brake Power", f"{L['P_brake']:.1f} W")
    cols[1].metric("Brake Efficiency", f"{L['eta_brake']*100:.2f} %")
    cols[2].metric("Carnot Efficiency", f"{L['eta_carnot']*100:.2f} %")
    cols[3].metric("Mass", f"{L['M']*1000:.2f} g")

# ---- Detailed results ----
st.subheader("Detailed Results")

if run_both and sim_s and sim_a:
    df_cols = st.columns(2)
    for col, (name, sim) in zip(df_cols, sims_to_show.items()):
        L = sim['losses']
        with col:
            st.markdown(f"**{name}**")
            st.text(f"""
M           : {L['M']*1000:.3f} g
P_mean      : {L['P_mean']/1e5:.3f} bar
P_max/P_min : {L['P_max']/1e5:.2f} / {L['P_min']/1e5:.2f} bar

W_cycle (raw)       : {L['W_cycle']:.4f} J
- W_pump            : {L['W_pump']:.4f} J
- W_leak            : {L['W_leak']:.4f} J
- W_mech_loss       : {L['W_mech_loss']:.4f} J
= W_shaft           : {L['W_shaft']:.4f} J

Q_in                : {L['Q_in_W']:.2f} W
  Q_e               : {L['Q_e']*params['f']:.2f} W
  Q_miss            : {L['Q_miss']*params['f']:.2f} W
  Q_cond            : {L['Q_cond_W']:.2f} W

Brake Power         : {L['P_brake']:.2f} W
Efficiency          : {L['eta_brake']*100:.3f} %
Fraction of Carnot  : {L['frac_carnot']*100:.2f} %
""")
elif not run_both and sim:
    L = sim['losses']
    st.text(f"""
M           : {L['M']*1000:.3f} g
P_mean      : {L['P_mean']/1e5:.3f} bar
P_max/P_min : {L['P_max']/1e5:.2f} / {L['P_min']/1e5:.2f} bar

WORK PER CYCLE [J]
  W_cycle (raw)     : {L['W_cycle']:.4f}
  - W_pump          : -{L['W_pump']:.4f}
  - W_leak          : -{L['W_leak']:.4f}
  - W_mech_loss     : -{L['W_mech_loss']:.4f}
  = W_shaft         : {L['W_shaft']:.4f}

HEAT INPUT [W]
  Q_e               : {L['Q_e']*params['f']:.2f}
  + Q_miss          : {L['Q_miss']*params['f']:.2f}
  + Q_cond          : {L['Q_cond_W']:.2f}
  = Q_in_total      : {L['Q_in_W']:.2f}

POWER & EFFICIENCY
  Brake Power       : {L['P_brake']:.2f} W
  Brake η           : {L['eta_brake']*100:.3f} %
  Carnot η          : {L['eta_carnot']*100:.3f} %
  Fraction Carnot   : {L['frac_carnot']*100:.2f} %
""")

# ---- Plots ----
st.subheader("📈 Plots")

if run_both and sim_s and sim_a:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    Vs = (sim_s['result']['V_e'] + sim_s['result']['V_c']) * 1e6
    Va = (sim_a['result']['V_e'] + sim_a['result']['V_c']) * 1e6
    theta = np.rad2deg(sim_s['result']['theta'])

    # P-V
    axes[0,0].plot(Vs, sim_s['result']['P']/1e5, color='#1F4E79', lw=2, label='Schmidt')
    axes[0,0].fill(Vs, sim_s['result']['P']/1e5, alpha=0.1, color='#1F4E79')
    axes[0,0].plot(Va, sim_a['result']['P']/1e5, color='#C00000', lw=2, label='Adiabatic')
    axes[0,0].fill(Va, sim_a['result']['P']/1e5, alpha=0.1, color='#C00000')
    axes[0,0].set_xlabel('Total Volume [cm³]')
    axes[0,0].set_ylabel('Pressure [bar]')
    axes[0,0].set_title('P-V Diagram')
    axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

    # P-theta
    axes[0,1].plot(theta, sim_s['result']['P']/1e5, color='#1F4E79', lw=2, label='Schmidt')
    axes[0,1].plot(theta, sim_a['result']['P']/1e5, color='#C00000', lw=2, label='Adiabatic')
    axes[0,1].set_xlabel('Crank angle [°]')
    axes[0,1].set_ylabel('Pressure [bar]')
    axes[0,1].set_title('Pressure vs θ')
    axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

    # Temperatures
    axes[1,0].axhline(T_h, color='#1F4E79', ls='--', lw=2, label='Schmidt T_e = T_h')
    axes[1,0].axhline(T_k, color='#1F4E79', ls=':',  lw=2, label='Schmidt T_c = T_k')
    axes[1,0].plot(theta, sim_a['result']['T_e'], color='#C00000', lw=2, label='Adia T_e')
    axes[1,0].plot(theta, sim_a['result']['T_c'], color='#ED7D31', lw=2, label='Adia T_c')
    axes[1,0].set_xlabel('Crank angle [°]')
    axes[1,0].set_ylabel('Temperature [K]')
    axes[1,0].set_title('Gas Temperatures')
    axes[1,0].legend(fontsize=8); axes[1,0].grid(alpha=0.3)

    # Bar comparison
    metrics = ['P_brake [W]', 'η [%]', '% Carnot']
    s_vals = [sim_s['losses']['P_brake'], sim_s['losses']['eta_brake']*100, sim_s['losses']['frac_carnot']*100]
    a_vals = [sim_a['losses']['P_brake'], sim_a['losses']['eta_brake']*100, sim_a['losses']['frac_carnot']*100]
    x = np.arange(len(metrics))
    axes[1,1].bar(x - 0.2, s_vals, 0.4, label='Schmidt', color='#1F4E79')
    axes[1,1].bar(x + 0.2, a_vals, 0.4, label='Adiabatic', color='#C00000')
    axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(metrics)
    axes[1,1].set_title('Performance Comparison')
    axes[1,1].legend(); axes[1,1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    st.pyplot(fig)

elif not run_both and sim:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    V_total = (sim['result']['V_e'] + sim['result']['V_c']) * 1e6
    theta = np.rad2deg(sim['result']['theta'])

    # P-V
    axes[0,0].plot(V_total, sim['result']['P']/1e5, color='#1F4E79', lw=2)
    axes[0,0].fill(V_total, sim['result']['P']/1e5, alpha=0.15, color='#1F4E79')
    axes[0,0].set_xlabel('Total Volume [cm³]')
    axes[0,0].set_ylabel('Pressure [bar]')
    axes[0,0].set_title('P-V Diagram')
    axes[0,0].grid(alpha=0.3)

    # P-theta
    axes[0,1].plot(theta, sim['result']['P']/1e5, color='#C00000', lw=2)
    axes[0,1].set_xlabel('Crank angle [°]')
    axes[0,1].set_ylabel('Pressure [bar]')
    axes[0,1].set_title('Pressure vs θ')
    axes[0,1].grid(alpha=0.3)

    # Volumes
    axes[1,0].plot(theta, sim['result']['V_e']*1e6, color='#C00000', lw=2, label='V_e')
    axes[1,0].plot(theta, sim['result']['V_c']*1e6, color='#1F4E79', lw=2, label='V_c')
    axes[1,0].set_xlabel('Crank angle [°]')
    axes[1,0].set_ylabel('Volume [cm³]')
    axes[1,0].set_title('Cylinder Volumes')
    axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

    # Waterfall
    L = sim['losses']
    labels = ['W_cycle']
    values = [L['W_cycle']]
    if losses_flags['flow']:        labels.append('-W_pump'); values.append(-L['W_pump'])
    if losses_flags['leakage']:     labels.append('-W_leak'); values.append(-L['W_leak'])
    if losses_flags['mechanical']:  labels.append('-W_mech'); values.append(-L['W_mech_loss'])
    labels.append('W_shaft'); values.append(L['W_shaft'])
    colors = ['#70AD47'] + ['#ED7D31']*(len(labels)-2) + ['#1F4E79']
    axes[1,1].bar(labels, values, color=colors)
    axes[1,1].axhline(0, color='black', lw=0.6)
    axes[1,1].set_ylabel('Work [J/cycle]')
    axes[1,1].set_title('Work Waterfall')
    axes[1,1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    st.pyplot(fig)

st.caption("💡 Tip: Change any parameter on the left, results update automatically.")

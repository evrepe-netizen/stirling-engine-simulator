"""
app.py — Stirling Engine Simulator v4 (Streamlit UI)
=====================================================
Imports physics from physics.py and optimization from optimization.py.

Tabs:
  1. Analysis   — Schmidt vs Adiabatic, P-V diagram, animation, Top-3 geometry
  2. Validation — 5 self-consistency checks
  3. Optimization — Coarse/Fine, LHS, or Bayesian search
  4. Export     — Excel download

To run:
    streamlit run app.py
"""

import math, io, base64, warnings, tempfile, os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import streamlit as st

from animation_v4 import build_engine_animation
from physics import (
    PROTOTYPE, GASES,
    to_si, build_geometry,
    simulate,
    validate_mass_conservation, validate_first_law,
    validate_carnot, validate_pressure_scaling,
)
from optimization import (
    OPTIMIZABLE_PARAMS, geometry_sensitivity,
    coarse_fine_search, lhs_search, bayesian_search,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stirling Engine Simulator v4",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────────
if 'initialized' not in st.session_state:
    for k, v in PROTOTYPE.items():
        st.session_state[k] = v
    st.session_state.initialized = True

def reset_to_prototype():
    for k, v in PROTOTYPE.items():
        st.session_state[k] = v

# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_reset = st.columns([4, 1])
with col_title:
    st.title("🔥 Stirling Engine Simulator v4")
    st.caption("Gamma-type engine — Analysis · Validation · Optimization · Export")
with col_reset:
    st.write(""); st.write("")
    if st.button("🔄 Reset to Prototype", type="secondary"):
        reset_to_prototype(); st.rerun()

tab_analysis, tab_validation, tab_optimize, tab_export = st.tabs(
    ["📊 Analysis", "✅ Validation", "🎯 Optimization", "📥 Export"]
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")

    st.subheader("Geometry")
    st.number_input("Displacer diameter [mm]",   30.0, 200.0, step=1.0, key='D_displacer')
    st.number_input("Displacer stroke [mm]",      20.0, 300.0, step=1.0, key='S_displacer')
    st.number_input("Power piston diameter [mm]", 30.0, 200.0, step=1.0, key='D_power')
    st.number_input("Power piston stroke [mm]",   20.0, 300.0, step=1.0, key='S_power')
    st.number_input("Phase angle [°]",            45.0, 135.0, step=5.0, key='phi_deg')

    st.subheader("Regenerator")
    st.number_input("Regen diameter [mm]", 10.0, 150.0, step=1.0,  key='D_r')
    st.number_input("Regen length [mm]",   20.0, 500.0, step=5.0,  key='L_r')
    st.number_input("Wire diameter [mm]",  0.05,   5.0, step=0.05, key='d_wire')
    st.slider("Porosity", 0.5, 0.99, step=0.01, key='porosity')

    st.subheader("Operating Conditions")
    st.selectbox("Working gas", ["Air", "Helium", "Hydrogen"], key='gas')
    st.number_input("Hot temperature T_h [K]",  400,  1500, step=10,  key='T_h')
    st.number_input("Cold temperature T_k [K]", 250,   400, step=5,   key='T_k')
    st.number_input("Mean pressure [bar]",       0.5,  50.0, step=0.5, key='P_mean_bar')
    st.number_input("Frequency [Hz]",            1,    100,  step=1,   key='f')

    st.subheader("Losses")
    flow_loss = st.checkbox("Regen flow loss (Ergun)", True)
    regen_imp = st.checkbox("Regen imperfection",      True)
    mech_loss = st.checkbox("Mech friction",           True)
    wall_cond = st.checkbox("Wall conduction",         True)
    leak_loss = st.checkbox("Seal leakage",            True)
    st.slider("ε_reg",  0.5,  0.99, step=0.01, key='eps_reg')
    st.slider("η_mech", 0.5,  0.99, step=0.01, key='eta_mech')
    st.slider("C_leak", 0.0,  0.20, step=0.01, key='C_leak')

    st.subheader("Model")
    model_choice = st.radio("Which model?", [
        "Schmidt (isothermal)",
        "Adiabatic (RK45)",
        "BOTH — compare side-by-side",
    ], index=2)

# ── Build shared params ───────────────────────────────────────────────────────
params = {k: st.session_state[k] for k in PROTOTYPE if k in st.session_state}
for k in ('V_loop_cold','V_loop_hot','V_cle','V_clc',
          'L_displacer','gap','P_ref','k_metal','t_wall'):
    params[k] = PROTOTYPE[k]

losses_flags = dict(flow=flow_loss, regen_imp=regen_imp, mechanical=mech_loss,
                    wall_cond=wall_cond, leakage=leak_loss, shuttle=False)
run_both = model_choice.startswith("BOTH")

with st.spinner("Computing..."):
    if run_both:
        sim_s = simulate(params, model='schmidt',   losses_flags=losses_flags)
        sim_a = simulate(params, model='adiabatic', losses_flags=losses_flags)
        sim   = sim_s
    else:
        primary = 'schmidt' if model_choice.startswith("Schmidt") else 'adiabatic'
        sim   = simulate(params, model=primary, losses_flags=losses_flags)
        sim_s = sim if primary == 'schmidt'   else None
        sim_a = sim if primary == 'adiabatic' else None



# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    st.header("📊 Results")

    if run_both and sim_s and sim_a:
        Ls = sim_s['losses']; La = sim_a['losses']
        c0,c1,c2,c3 = st.columns(4)
        c0.metric("Schmidt Power",   f"{Ls['P_brake']:.1f} W")
        c1.metric("Adiabatic Power", f"{La['P_brake']:.1f} W")
        c2.metric("Schmidt η",       f"{Ls['eta_brake']*100:.2f}%")
        c3.metric("Adiabatic η",     f"{La['eta_brake']*100:.2f}%")

        M_g = Ls['M'] * 1000
        t_ratio = params['T_h'] / params['T_k']
        if La['W_cycle'] > Ls['W_cycle']:
            st.info(
                f"⚖️ **Both models use the same gas mass M = {M_g:.3f} g** "
                f"(computed by Schmidt for P_mean = {params['P_mean_bar']:.2f} bar). "
                f"W_adiabatic ({La['W_cycle']:.3f} J) > W_schmidt ({Ls['W_cycle']:.3f} J) — "
                f"expected for T_h/T_k = {t_ratio:.2f} > 1.8: in the adiabatic model, "
                f"T_e swings above T_h and T_c swings below T_k, enlarging the P-V loop "
                f"(Urieli & Berchowitz 1984, Ch. 4). "
                f"Adiabatic P_mean = {La['P_mean']/1e5:.3f} bar (output, not forced). "
                f"The real engine falls below both ideal models."
            )
        else:
            st.info(
                f"⚖️ Both models use M = {M_g:.3f} g. "
                f"W_schmidt ({Ls['W_cycle']:.3f} J) ≥ W_adiabatic ({La['W_cycle']:.3f} J) — "
                f"T_h/T_k = {t_ratio:.2f} ≤ 1.8. "
                f"Adiabatic P_mean = {La['P_mean']/1e5:.3f} bar (output)."
            )

        col_s, col_a = st.columns(2)
        for col, name, L in [(col_s,'Schmidt',Ls),(col_a,'Adiabatic',La)]:
            with col:
                st.markdown(f"**{name}**")
                st.code(
                    f"M           : {L['M']*1000:.3f} g\n"
                    f"P_mean      : {L['P_mean']/1e5:.3f} bar\n"
                    f"P_max/min   : {L['P_max']/1e5:.2f} / {L['P_min']/1e5:.2f} bar\n"
                    f"W_cycle     : {L['W_cycle']:.4f} J\n"
                    f"W_shaft     : {L['W_shaft']:.4f} J\n"
                    f"Brake Power : {L['P_brake']:.2f} W\n"
                    f"Q_in        : {L['Q_in_W']:.2f} W\n"
                    f"Efficiency  : {L['eta_brake']*100:.3f} %\n"
                    f"% Carnot    : {L['frac_carnot']*100:.2f} %",
                    language='text'
                )

        # ── Plots ────────────────────────────────────────────────────────────
        st.subheader("📈 Diagrams")
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        Vs = (sim_s['result']['V_e'] + sim_s['result']['V_c']) * 1e6
        Va = (sim_a['result']['V_e'] + sim_a['result']['V_c']) * 1e6
        th = np.rad2deg(sim_s['result']['theta'])

        ax = axes[0, 0]
        ax.plot(Vs, sim_s['result']['P']/1e5, '#1565C0', lw=2, label='Schmidt')
        ax.fill(Vs, sim_s['result']['P']/1e5, alpha=0.1, color='#1565C0')
        ax.plot(Va, sim_a['result']['P']/1e5, '#C62828', lw=2, label='Adiabatic')
        ax.fill(Va, sim_a['result']['P']/1e5, alpha=0.1, color='#C62828')
        ax.set(xlabel='V_total [cm³]', ylabel='P [bar]', title='P-V Diagram')
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[0, 1]
        ax.plot(th, sim_s['result']['P']/1e5, '#1565C0', lw=2, label='Schmidt')
        ax.plot(th, sim_a['result']['P']/1e5, '#C62828', lw=2, label='Adiabatic')
        ax.set(xlabel='θ [°]', ylabel='P [bar]', title='Pressure vs θ')
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[1, 0]
        ax.axhline(params['T_h'], color='#C62828', ls='--', lw=1.5, label='Schmidt T_e')
        ax.axhline(params['T_k'], color='#1565C0', ls='--', lw=1.5, label='Schmidt T_c')
        ax.plot(th, sim_a['result']['T_e'], '#D32F2F', lw=2, label='Adia T_e')
        ax.plot(th, sim_a['result']['T_c'], '#1976D2', lw=2, label='Adia T_c')
        ax.set(xlabel='θ [°]', ylabel='T [K]', title='Gas Temperatures')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        metrics = ['Brake Power [W]', 'η [%]', '% Carnot']
        sv = [Ls['P_brake'], Ls['eta_brake']*100, Ls['frac_carnot']*100]
        av = [La['P_brake'], La['eta_brake']*100, La['frac_carnot']*100]
        x = np.arange(len(metrics))
        ax = axes[1, 1]
        ax.bar(x-0.2, sv, 0.4, label='Schmidt',   color='#1565C0')
        ax.bar(x+0.2, av, 0.4, label='Adiabatic', color='#C62828')
        ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=8)
        ax.set_title('Performance Comparison'); ax.legend(); ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    elif sim:
        L = sim['losses']
        c0,c1,c2,c3 = st.columns(4)
        c0.metric("Brake Power",  f"{L['P_brake']:.1f} W")
        c1.metric("Efficiency",   f"{L['eta_brake']*100:.2f}%")
        c2.metric("Carnot limit", f"{L['eta_carnot']*100:.2f}%")
        c3.metric("Gas mass",     f"{L['M']*1000:.2f} g")

    # ── Engine Animation ──────────────────────────────────────────────────────
    st.subheader("🎬 Engine Animation")
    with st.spinner("Rendering animation..."):
        try:
            gif_b64 = build_engine_animation(build_geometry(to_si(params)), params)
            st.markdown(
                f'<img src="data:image/gif;base64,{gif_b64}" '
                f'style="width:100%;max-width:780px;border-radius:8px;" />',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.warning(f"Animation error: {e}")

    # ── Top-3 Geometry Recommendations ───────────────────────────────────────
    if sim:
        st.subheader("🔧 Top-3 Geometric Improvements")
        st.caption("Analysis focuses on geometric parameters only. "
                   "Operating conditions and efficiency assumptions are held fixed.")
        with st.spinner("Running sensitivity analysis..."):
            sens = geometry_sensitivity(params, losses_flags)

        if sens:
            for rank, (key, name, units, base_val, best_val, delta_W, pct) in enumerate(sens[:3], 1):
                direction = "↑ Increase" if best_val > base_val else "↓ Decrease"
                arrow = "🟢" if delta_W > 0 else "🔴"
                with st.expander(
                    f"#{rank}  {name}  —  {arrow} {delta_W:+.2f} W brake power  ({pct:+.1f}%)",
                    expanded=(rank == 1)
                ):
                    st.markdown(
                        f"**{direction} {name}** from **{base_val:.1f} {units}** "
                        f"to **{best_val:.1f} {units}**\n\n"
                        f"Estimated brake power change: **{delta_W:+.2f} W** ({pct:+.1f}%)\n\n"
                        f"*Based on ±10% local sensitivity analysis (Schmidt model).*"
                    )
        else:
            st.info("Could not compute sensitivity — check parameters.")




# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — VALIDATION
# ════════════════════════════════════════════════════════════════════════════
with tab_validation:
    st.header("✅ Validation Suite")
    if sim is None:
        st.error("Simulation failed — check parameters."); st.stop()

    t_ratio = params['T_h'] / params['T_k']

    with st.expander("1️⃣  Mass Conservation  (Δm/m_avg < 2 %)", expanded=True):
        delta, ok = validate_mass_conservation(sim['result'], sim['geom'], sim['gas'])
        (st.success if ok else st.error)(f"{'✅' if ok else '❌'}  Mass variation: **{delta:.3f} %**"
                                         + (" < 2 % ✓" if ok else " > 2 % limit"))

    with st.expander("2️⃣  First Law  (W_e + W_c = W_cycle, error < 0.1 %)", expanded=True):
        err, ok = validate_first_law(sim['losses'])
        (st.success if ok else st.error)(f"{'✅' if ok else '❌'}  First-law error: **{err:.6f} %**")

    with st.expander("3️⃣  Carnot Bound  (η_brake ≤ η_Carnot)", expanded=True):
        eta_b, eta_c, ok = validate_carnot(sim['losses'])
        (st.success if ok else st.error)(
            f"{'✅' if ok else '❌'}  η_brake = {eta_b*100:.3f} %  "
            f"{'≤' if ok else '>'} η_Carnot = {eta_c*100:.2f} %  "
            f"({sim['losses']['frac_carnot']*100:.1f} % of Carnot)"
        )

    with st.expander("4️⃣  Zero ΔT Limit  (W → 0 as T_h → T_k)", expanded=True):
        lf0 = dict(flow=False,regen_imp=False,mechanical=False,wall_cond=False,leakage=False,shuttle=False)
        pz  = dict(params); pz['T_h'] = params['T_k'] + 5
        sz  = simulate(pz, model='schmidt', losses_flags=lf0)
        if sz:
            ratio = abs(sz['losses']['W_cycle']) / max(abs(sim['losses']['W_cycle']), 1e-9)
            ok_z  = ratio < 0.05
            (st.success if ok_z else st.error)(
                f"{'✅' if ok_z else '❌'}  W(ΔT=5K) = {sz['losses']['W_cycle']:.5f} J  "
                f"({ratio*100:.2f} % of nominal)"
            )

    with st.expander("5️⃣  Schmidt vs Adiabatic  (physics note)", expanded=True):
        if run_both and sim_s and sim_a:
            Ws = sim_s['losses']['W_cycle']; Wa = sim_a['losses']['W_cycle']
            M_g = sim_s['losses']['M']*1000
            st.info(
                f"⚖️ **Both models: M = {M_g:.3f} g** (Option A — same gas mass).\n\n"
                f"Schmidt W = {Ws:.4f} J  |  "
                f"Adiabatic W = {Wa:.4f} J  |  "
                f"Adiabatic P_mean = {sim_a['losses']['P_mean']/1e5:.3f} bar (output).\n\n"
            )
            if Wa > Ws:
                st.info(
                    f"W_adia > W_schmidt — expected for T_h/T_k = {t_ratio:.2f} > 1.8.  "
                    f"T_e ∈ [{sim_a['result']['T_e'].min():.0f}, {sim_a['result']['T_e'].max():.0f}] K, "
                    f"T_c ∈ [{sim_a['result']['T_c'].min():.0f}, {sim_a['result']['T_c'].max():.0f}] K.  "
                    "Real engine is below both ideals (Urieli & Berchowitz 1984, Ch. 4)."
                )
            else:
                st.success(f"✅ W_schmidt ≥ W_adiabatic — T_h/T_k = {t_ratio:.2f} ≤ 1.8.")
        else:
            st.info(f"Select **BOTH** models in sidebar to compare. T_h/T_k = {t_ratio:.2f}.")

    with st.expander("6️⃣  Pressure Linearity  (W ∝ P_mean)", expanded=True):
        with st.spinner("Running scaling test..."):
            W1, W2, ratio_p, ok_p = validate_pressure_scaling(params, losses_flags)
        if W1:
            (st.success if ok_p else st.error)(
                f"{'✅' if ok_p else '❌'}  "
                f"W({params['P_mean_bar']:.1f} bar) = {W1:.4f} J, "
                f"W({params['P_mean_bar']*2:.1f} bar) = {W2:.4f} J — "
                f"ratio = {ratio_p:.5f} (expected 2.000)"
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — OPTIMIZATION
# ════════════════════════════════════════════════════════════════════════════
with tab_optimize:
    st.header("🎯 Optimization")

    st.subheader("Lock / Open parameters")
    open_specs = []
    n_cols = 2
    cols   = st.columns(n_cols)
    for i, (display_name, key, pmin, pmax, step, units) in enumerate(OPTIMIZABLE_PARAMS):
        col = cols[i % n_cols]
        with col:
            current = st.session_state.get(key, PROTOTYPE.get(key, pmin))
            locked  = st.checkbox(f"🔒 Lock **{display_name}** = {current} {units}",
                                   value=True, key=f"lock_{key}")
            if not locked:
                cc1, cc2 = st.columns(2)
                pmn = cc1.number_input("min", value=float(pmin), step=float(step),
                                        key=f"min_{key}", label_visibility="collapsed")
                pmx = cc2.number_input("max", value=float(pmax), step=float(step),
                                        key=f"max_{key}", label_visibility="collapsed")
                open_specs.append((key, pmn, pmx))

    st.subheader("Strategy")
    strategy = st.radio("Search method:", [
        "Coarse → Fine grid (recommended, fast)",
        "Latin Hypercube Sampling (LHS, smarter)",
        "Bayesian Optimization (advanced, requires scikit-optimize)",
    ], horizontal=False)

    if strategy.startswith("Latin"):
        n_lhs = st.slider("Number of LHS samples", 100, 2000, 500, 100)
    elif strategy.startswith("Bayesian"):
        n_bay = st.slider("Number of Bayesian calls", 20, 200, 60, 10)

    st.subheader("Objective")
    obj_choice = st.radio("Maximize:", [
        "Brake Power", "Brake Efficiency", "Power × Efficiency (balanced)"
    ], horizontal=True)
    obj_map = {'Brake Power':'power','Brake Efficiency':'efficiency',
               'Power × Efficiency (balanced)':'balanced'}
    objective = obj_map[obj_choice]

    st.subheader("Model")
    opt_model_key = 'schmidt' if st.radio(
        "Model:", ["Schmidt (fast)", "Adiabatic (slower)"], horizontal=True
    ).startswith("Schmidt") else 'adiabatic'

    st.caption(
        "**Excluded from automated optimization:** porosity, ε_reg, η_mech, C_leak — "
        "coupled material/manufacturing properties. Adjust manually in the sidebar."
    )
    if not open_specs:
        st.warning("Unlock at least one parameter above.")
    else:
        st.info(f"{len(open_specs)} parameter(s) open for optimization.")

    if st.button("🚀 Run Optimization", type="primary", disabled=(len(open_specs)==0)):
        base_params = dict(params)
        opt_flags   = dict(flow=True, regen_imp=True, mechanical=True,
                           wall_cond=True, leakage=True, shuttle=False)
        prog_bar = st.progress(0, text="Starting...")

        def cb(frac, msg):
            prog_bar.progress(min(frac, 1.0), text=msg)

        if strategy.startswith("Coarse"):
            bp, bl, all_r = coarse_fine_search(base_params, open_specs, objective,
                                                opt_model_key, opt_flags, cb)
        elif strategy.startswith("Latin"):
            bp, bl, all_r = lhs_search(base_params, open_specs, objective,
                                        opt_model_key, opt_flags, n_lhs, cb)
        else:
            bp, bl, all_r = bayesian_search(base_params, open_specs, objective,
                                             opt_model_key, opt_flags, n_bay, cb)
        prog_bar.empty()

        if bl is None:
            st.error("❌ No valid configuration found.")
        else:
            st.success(f"✅ Done!  Best brake power: {bl['P_brake']:.2f} W, "
                       f"η = {bl['eta_brake']*100:.2f}%")
            sim_cur = simulate(base_params, model=opt_model_key, losses_flags=opt_flags)
            c1,c2 = st.columns(2)
            if sim_cur:
                c1.metric("Best Power",    f"{bl['P_brake']:.2f} W",
                          delta=f"{bl['P_brake']-sim_cur['losses']['P_brake']:+.2f} W")
                c2.metric("Best η",        f"{bl['eta_brake']*100:.2f}%",
                          delta=f"{(bl['eta_brake']-sim_cur['losses']['eta_brake'])*100:+.2f}%")

            # Show changed parameters
            rows = []
            for _, key, *_ in OPTIMIZABLE_PARAMS:
                if any(s[0]==key for s in open_specs):
                    rows.append({'Parameter': key,
                                 'Current': f"{base_params.get(key,'?')}",
                                 'Optimum': f"{bp.get(key,'?')}"})
            st.table(rows)

            # P-V comparison
            sim_best = simulate(bp, model=opt_model_key, losses_flags=opt_flags)
            if sim_cur and sim_best:
                fig2, ax2 = plt.subplots(figsize=(9,5))
                Vc=(sim_cur['result']['V_e']+sim_cur['result']['V_c'])*1e6
                Vb=(sim_best['result']['V_e']+sim_best['result']['V_c'])*1e6
                ax2.plot(Vc,sim_cur['result']['P']/1e5,'#888',lw=2,label='Current')
                ax2.fill(Vc,sim_cur['result']['P']/1e5,alpha=0.1,color='#888')
                ax2.plot(Vb,sim_best['result']['P']/1e5,'#2E7D32',lw=2.5,label='Optimum')
                ax2.fill(Vb,sim_best['result']['P']/1e5,alpha=0.15,color='#2E7D32')
                ax2.set(xlabel='V_total [cm³]',ylabel='P [bar]',title='P-V: Current vs Optimum')
                ax2.legend(); ax2.grid(alpha=0.3)
                st.pyplot(fig2); plt.close(fig2)

            # Store for export
            st.session_state['opt_result'] = dict(
                best_params=bp, best_losses=bl, all_results=all_r,
                base_params=base_params, strategy=strategy, objective=obj_choice
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPORT
# ════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.header("📥 Export to Excel")
    st.markdown("Download a complete Excel workbook with all simulation results.")

    if sim is None:
        st.error("No simulation results — go to Analysis tab first."); st.stop()

    if st.button("📊 Generate Excel", type="primary"):
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment
            from openpyxl.utils import get_column_letter
        except ImportError:
            st.error("openpyxl not installed. Run: pip install openpyxl"); st.stop()

        wb = openpyxl.Workbook()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── Sheet 1 — Summary ────────────────────────────────────────────────
        ws1 = wb.active; ws1.title = "Summary"
        header_fill = PatternFill("solid", fgColor="1565C0")
        header_font = Font(color="FFFFFF", bold=True)

        def _hdr(ws, row, col, text):
            c = ws.cell(row=row, column=col, value=text)
            c.fill = header_fill; c.font = header_font
            c.alignment = Alignment(horizontal='center')

        def _row(ws, r, label, value, unit=""):
            ws.cell(r,1,label); ws.cell(r,2,value); ws.cell(r,3,unit)

        _hdr(ws1,1,1,"Stirling Engine Simulator v4 — Summary"); ws1.merge_cells('A1:C1')
        _row(ws1,2,"Timestamp",ts)
        _row(ws1,3,"Model",sim['model'])
        _row(ws1,4,"Gas",params['gas'])
        ws1.append([])
        _hdr(ws1,6,1,"INPUT PARAMETERS")
        input_rows = [
            ("D_displacer",   params['D_displacer'],   "mm"),
            ("S_displacer",   params['S_displacer'],   "mm"),
            ("D_power",       params['D_power'],       "mm"),
            ("S_power",       params['S_power'],       "mm"),
            ("phi_deg",       params['phi_deg'],       "°"),
            ("D_r",           params['D_r'],           "mm"),
            ("L_r",           params['L_r'],           "mm"),
            ("d_wire",        params['d_wire'],        "mm"),
            ("porosity",      params['porosity'],      ""),
            ("T_h",           params['T_h'],           "K"),
            ("T_k",           params['T_k'],           "K"),
            ("P_mean_bar",    params['P_mean_bar'],    "bar"),
            ("f",             params['f'],             "Hz"),
            ("eps_reg",       params['eps_reg'],       ""),
            ("eta_mech",      params['eta_mech'],      ""),
        ]
        for i,(label,val,unit) in enumerate(input_rows, 7):
            _row(ws1,i,label,val,unit)
        r = 7+len(input_rows)+1
        _hdr(ws1,r,1,"KEY OUTPUTS"); r+=1
        L = sim['losses']
        for label,val,unit in [
            ("W_cycle",     L['W_cycle'],             "J"),
            ("W_shaft",     L['W_shaft'],             "J"),
            ("P_brake",     L['P_brake'],             "W"),
            ("eta_brake",   L['eta_brake']*100,       "%"),
            ("eta_carnot",  L['eta_carnot']*100,      "%"),
            ("frac_carnot", L['frac_carnot']*100,     "%"),
            ("M_gas",       L['M']*1000,              "g"),
            ("P_mean_out",  L['P_mean']/1e5,          "bar"),
        ]:
            _row(ws1,r,label,round(val,5),unit); r+=1
        ws1.column_dimensions['A'].width = 22
        ws1.column_dimensions['B'].width = 18
        ws1.column_dimensions['C'].width = 8

        # ── Sheet 2 — Detailed losses ────────────────────────────────────────
        ws2 = wb.create_sheet("Detailed Results")
        _hdr(ws2,1,1,"Loss Component"); _hdr(ws2,1,2,"J/cycle"); _hdr(ws2,1,3,"W")
        loss_rows = [
            ("W_pump (flow)",    L['W_pump'],     L['W_pump']*params['f']),
            ("W_leak (leakage)", L['W_leak'],     L['W_leak']*params['f']),
            ("W_mech_loss",      L['W_mech_loss'],L['W_mech_loss']*params['f']),
            ("Q_miss (regen)",   L['Q_miss'],     L['Q_miss']*params['f']),
            ("Q_cond (wall)",    L['Q_cond'],     L['Q_cond_W']),
            ("W_shaft (output)", L['W_shaft'],    L['P_brake']),
        ]
        for i,(label,jc,w) in enumerate(loss_rows,2):
            ws2.cell(i,1,label); ws2.cell(i,2,round(jc,5)); ws2.cell(i,3,round(w,3))

        # Sensitivity / top-3
        ws2.append([]); ws2.append(["Top-3 Geometry Sensitivity"])
        sens = geometry_sensitivity(params, losses_flags)
        ws2.append(["Parameter","Base value","Best value","ΔP_brake [W]","Change [%]"])
        for key,name,units,bv,bst,dw,pct in sens[:3]:
            ws2.append([f"{name} ({key})", f"{bv:.2f} {units}",
                        f"{bst:.2f} {units}", round(dw,3), round(pct,2)])

        ws2.column_dimensions['A'].width = 25
        ws2.column_dimensions['B'].width = 14

        # ── Sheet 3 — Cycle data ─────────────────────────────────────────────
        ws3 = wb.create_sheet("Cycle Data")
        headers3 = ["theta_deg","P_bar","V_e_cm3","V_c_cm3","T_e_K","T_c_K","V_total_cm3"]
        for j,h in enumerate(headers3,1): _hdr(ws3,1,j,h)
        R = sim['result']
        for i in range(len(R['theta'])):
            ws3.append([
                round(math.degrees(R['theta'][i]),2),
                round(R['P'][i]/1e5,5),
                round(R['V_e'][i]*1e6,4),
                round(R['V_c'][i]*1e6,4),
                round(R['T_e'][i],2),
                round(R['T_c'][i],2),
                round((R['V_e'][i]+R['V_c'][i])*1e6,4),
            ])

        # ── Sheet 4 — Optimization (if available) ────────────────────────────
        if 'opt_result' in st.session_state:
            opt = st.session_state['opt_result']
            ws4 = wb.create_sheet("Optimization")
            ws4.append(["Strategy", opt['strategy']])
            ws4.append(["Objective", opt['objective']])
            ws4.append([])
            ws4.append(["Parameter","Base","Optimum"])
            for _,key,*_ in OPTIMIZABLE_PARAMS:
                if key in opt['best_params']:
                    ws4.append([key, opt['base_params'].get(key,''),
                                 opt['best_params'].get(key,'')])
            ws4.append([])
            ws4.append(["Best P_brake [W]", round(opt['best_losses']['P_brake'],3)])
            ws4.append(["Best eta [%]",     round(opt['best_losses']['eta_brake']*100,3)])

        # ── Sheet 5 — Validation ─────────────────────────────────────────────
        ws5 = wb.create_sheet("Validation")
        ws5.append(["Check","Result","Pass/Fail"])
        delta,ok_m = validate_mass_conservation(sim['result'],sim['geom'],sim['gas'])
        ws5.append(["Mass conservation", f"{delta:.3f}%", "PASS" if ok_m else "FAIL"])
        err1L,ok1L = validate_first_law(sim['losses'])
        ws5.append(["First Law", f"{err1L:.6f}%", "PASS" if ok1L else "FAIL"])
        etab,etac,okc = validate_carnot(sim['losses'])
        ws5.append(["Carnot bound", f"η={etab*100:.3f}% ≤ {etac*100:.2f}%", "PASS" if okc else "FAIL"])
        W1p,W2p,rp,okp = validate_pressure_scaling(params, losses_flags)
        ws5.append(["Pressure linearity", f"ratio={rp:.5f}", "PASS" if okp else "FAIL"])

        # ── Save and offer download ───────────────────────────────────────────
        buf = io.BytesIO()
        wb.save(buf); buf.seek(0)
        fname = f"stirling_results_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.xlsx"
        st.download_button(
            label="⬇️ Download Excel",
            data=buf.getvalue(),
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.success(f"✅ Excel ready: {fname}")

st.caption("💡 Stirling Engine Simulator v4 — physics.py · optimization.py · app.py")

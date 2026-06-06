"""
optimization_v8.py — Smart optimization for Stirling engine parameters
=======================================================================
Three strategies:
  1. Coarse→Fine two-stage grid search  (default, always available)
  2. Latin Hypercube Sampling (LHS)     (scipy.stats.qmc — part of scipy)
  3. Bayesian Optimization              (scikit-optimize, optional)

Geometry-only sensitivity analysis for Top-3 recommendations.
Imports only from physics_v8.py — no Streamlit.

v8 changes vs V7:
  - gap moved to [mm] units in OPTIMIZABLE_PARAMS (0.10–0.50 mm, step 0.05)
    for human-readable UI display. _to_physics_params() converts gap_mm → m
    before every simulate() call so physics remain in SI.
  - gap and L_displacer_effective added to GEOMETRIC_KEYS and param_meta so
    geometry_sensitivity() evaluates them alongside other geometric parameters.

NOTE on excluded parameters:
  Porosity, eps_reg, eta_mech, C_leak are coupled material/manufacturing
  properties. Adjust them manually in the sidebar.
"""

import numpy as np
from itertools import product as itertools_product

from physics_v9_4 import simulate, simulate_fixed_heat, PROTOTYPE

# Keys whose optimizer values are in mm but must be converted to m for physics.
# Currently only 'gap'; L_displacer_effective is kept in metres for SI consistency.
_MM_TO_M_KEYS = {'gap'}

# ── Parameter definitions ─────────────────────────────────────────────────────
# (display_name, key, min, max, default_step, units)
# NOTE: 'gap' is listed in mm here for UI readability.
#       _to_physics_params() converts it to metres before simulate() calls.
OPTIMIZABLE_PARAMS = [
    ('Displacer diameter',         'D_displacer',           30,    150,    5,     'mm'),
    ('Displacer stroke',           'S_displacer',           30,    200,    10,    'mm'),
    ('Power piston diameter',      'D_power',               30,    150,    5,     'mm'),
    ('Power piston stroke',        'S_power',               20,    150,    10,    'mm'),
    ('Phase angle',                'phi_deg',               60,    120,    15,    '°'),
    ('Regen diameter',             'D_r',                   20,    100,    10,    'mm'),
    ('Regen length',               'L_r',                   50,    400,    50,    'mm'),
    ('Wire diameter',              'd_wire',                0.5,   3.0,    0.5,   'mm'),
    ('Hot temperature',            'T_h',                   573,   1273,   100,   'K'),
    ('Mean pressure',              'P_mean_bar',            1.0,   30.0,   5.0,   'bar'),
    ('Frequency',                  'f',                     5,     50,     5,     'Hz'),
    # Shuttle-loss parameters.
    # gap: stored and displayed in mm (0.10–0.50), converted to m for physics.
    # Bounds enforced to prevent optimizer from making gap unphysically large.
    ('Displacer radial gap',       'gap',                   0.10,  0.50,   0.05,  'mm'),
    ('Effective displacer length', 'L_displacer_effective', 0.05,  0.235,  0.02,  'm'),
    ('Heat Input (Max/Target)',    'Q_in_max',              50,    3000,   50,    'W'),
]
# NOTE: porosity, eps_reg, eta_mech, C_leak are intentionally excluded.

# Geometric params for sensitivity analysis — v8: gap and L_displacer_effective included.
GEOMETRIC_KEYS = {
    'D_displacer', 'S_displacer', 'D_power', 'S_power',
    'phi_deg', 'D_r', 'L_r', 'd_wire',
    'gap', 'L_displacer_effective',
}


def _to_physics_params(p):
    """
    Return a copy of params dict with optimizer mm-unit keys converted to SI
    before passing to simulate(). Currently converts 'gap': mm → m.
    """
    q = dict(p)
    if 'gap' in q:
        q['gap'] = q['gap'] * 1e-3   # mm → m
    return q

EXCLUDED_NOTE = (
    "**Excluded from automated optimization:** porosity, regenerator effectiveness "
    "(ε_reg), mechanical efficiency (η_mech), and seal leakage coefficient (C_leak) "
    "are coupled material/manufacturing properties. Modify them manually in the sidebar "
    "when exploring trade-offs."
)


def _score(losses, params, objective):
    """
    Scalar score to maximize.
    Applies a heat_factor penalty: if required heat exceeds the budget
    (params['Q_in_max']), available power and efficiency are scaled down
    proportionally. This prevents the optimizer from finding configurations
    that are theoretically powerful but physically infeasible given the
    available heat supply.
    """
    if losses['W_shaft'] <= 0 or losses['Q_in'] <= 0:
        return -float('inf')

    Q_req = losses['Q_in_W']
    Q_max = params.get('Q_in_max', 3000)
    heat_factor = min(1.0, Q_max / Q_req) if Q_req > 0 else 0.0

    P_avail  = losses['P_brake']  * heat_factor
    eta_eff  = losses['eta_brake'] * heat_factor

    if objective == 'power':
        return P_avail
    elif objective == 'efficiency':
        return eta_eff
    else:   # balanced
        return P_avail * eta_eff


# ── Shared simulate helper ────────────────────────────────────────────────────
def _run_sim(p, model_key, losses_flags, driving_mode):
    """
    Run the correct simulation depending on driving mode.
    In Fixed Heat Input mode, uses simulate_fixed_heat with p['Q_in_max'].
    In Fixed T_h mode (default), uses the standard simulate().
    Always applies _to_physics_params() for mm→m gap conversion.
    """
    phys_p = _to_physics_params(p)
    if driving_mode == "Fixed Heat Input (Q_in)":
        return simulate_fixed_heat(
            phys_p, phys_p.get('Q_in_max', 500),
            model=model_key, losses_flags=losses_flags
        )
    return simulate(phys_p, model=model_key, losses_flags=losses_flags)


# ── 1. Coarse → Fine grid search ─────────────────────────────────────────────
def coarse_fine_search(base_params, open_specs, objective, model_key,
                       losses_flags, driving_mode="Fixed Hot Temperature (T_h)",
                       progress_cb=None):
    """
    Two-stage grid search.
      Stage 1: 4 values per parameter (coarse sweep).
      Stage 2: 5 values in ±20% neighborhood of best coarse point.

    open_specs: list of (key, min_val, max_val).
    driving_mode: passed through to _run_sim to choose Fixed T_h vs Fixed Q_in.
    Returns (best_params, best_losses, all_results_list).
    """
    keys   = [s[0] for s in open_specs]
    bounds = [(s[1], s[2]) for s in open_specs]

    coarse_grids = [np.linspace(lo, hi, 4) for lo, hi in bounds]
    total_c = 1
    for g in coarse_grids: total_c *= len(g)

    best_score  = -float('inf')
    best_params = None
    best_combo  = None
    count = 0

    for combo in itertools_product(*coarse_grids):
        p = dict(base_params)
        for k, v in zip(keys, combo):
            p[k] = float(v)
        sim = _run_sim(p, model_key, losses_flags, driving_mode)
        count += 1
        if progress_cb and count % max(1, total_c // 40) == 0:
            progress_cb(0.5 * count / total_c, f"Coarse: {count}/{total_c}")
        if sim is None:
            continue
        s = _score(sim['losses'], p, objective)
        if s > best_score:
            best_score  = s
            best_params = dict(p)
            best_combo  = combo

    if best_combo is None:
        return None, None, []

    fine_grids = []
    for v, (lo, hi) in zip(best_combo, bounds):
        span = (hi - lo) * 0.2
        f_lo = max(lo, v - span)
        f_hi = min(hi, v + span)
        fine_grids.append(np.linspace(f_lo, f_hi, 5))

    total_f = 1
    for g in fine_grids: total_f *= len(g)
    count_f = 0
    all_results = []

    for combo in itertools_product(*fine_grids):
        p = dict(base_params)
        for k, v in zip(keys, combo):
            p[k] = float(v)
        sim = _run_sim(p, model_key, losses_flags, driving_mode)
        count_f += 1
        if progress_cb and count_f % max(1, total_f // 40) == 0:
            progress_cb(0.5 + 0.5 * count_f / total_f, f"Fine: {count_f}/{total_f}")
        if sim is None:
            continue
        s = _score(sim['losses'], p, objective)
        all_results.append((s, dict(p), sim['losses']))
        if s > best_score:
            best_score  = s
            best_params = dict(p)

    all_results.sort(key=lambda x: x[0], reverse=True)

    sim_best    = _run_sim(best_params, model_key, losses_flags, driving_mode)
    best_losses = sim_best['losses'] if sim_best else (
        all_results[0][2] if all_results else None
    )
    return best_params, best_losses, all_results


# ── 2. Latin Hypercube Sampling ───────────────────────────────────────────────
def lhs_search(base_params, open_specs, objective, model_key,
               losses_flags, n_samples=500,
               driving_mode="Fixed Hot Temperature (T_h)", progress_cb=None):
    """
    Latin Hypercube Sampling across the parameter space.
    Returns (best_params, best_losses, all_results_list).
    """
    try:
        from scipy.stats.qmc import LatinHypercube
    except ImportError:
        raise ImportError("scipy >= 1.7 required for LHS (scipy.stats.qmc)")

    keys     = [s[0] for s in open_specs]
    bounds   = np.array([(s[1], s[2]) for s in open_specs])
    n_params = len(keys)

    sampler = LatinHypercube(d=n_params, seed=42)
    unit    = sampler.random(n=n_samples)
    samples = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])

    best_score  = -float('inf')
    best_params = None
    all_results = []

    for i, row in enumerate(samples):
        p = dict(base_params)
        for k, v in zip(keys, row):
            p[k] = float(v)
        sim = _run_sim(p, model_key, losses_flags, driving_mode)
        if progress_cb and i % max(1, n_samples // 50) == 0:
            progress_cb(i / n_samples, f"LHS: {i}/{n_samples}")
        if sim is None:
            continue
        s = _score(sim['losses'], p, objective)
        all_results.append((s, dict(p), sim['losses']))
        if s > best_score:
            best_score  = s
            best_params = dict(p)

    all_results.sort(key=lambda x: x[0], reverse=True)
    sim_best    = _run_sim(best_params, model_key, losses_flags, driving_mode) if best_params else None
    best_losses = sim_best['losses'] if sim_best else (all_results[0][2] if all_results else None)
    return best_params, best_losses, all_results


# ── 3. Bayesian Optimization (optional) ──────────────────────────────────────
def bayesian_search(base_params, open_specs, objective, model_key,
                    losses_flags, n_calls=60,
                    driving_mode="Fixed Hot Temperature (T_h)", progress_cb=None):
    """
    Gaussian-process Bayesian optimization via scikit-optimize.
    Falls back to LHS if skopt is not installed.
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError:
        if progress_cb:
            progress_cb(0, "scikit-optimize not installed — falling back to LHS")
        return lhs_search(base_params, open_specs, objective, model_key,
                          losses_flags, n_samples=n_calls * 8,
                          driving_mode=driving_mode, progress_cb=progress_cb)

    keys       = [s[0] for s in open_specs]
    dimensions = [Real(s[1], s[2], name=s[0]) for s in open_specs]
    all_results = []
    call_count  = [0]
    best_score  = [-float('inf')]
    best_params = [None]

    def objective_fn(values):
        p = dict(base_params)
        for k, v in zip(keys, values):
            p[k] = float(v)
        sim = _run_sim(p, model_key, losses_flags, driving_mode)
        call_count[0] += 1
        if progress_cb and call_count[0] % 5 == 0:
            progress_cb(call_count[0] / n_calls, f"Bayesian: {call_count[0]}/{n_calls}")
        if sim is None:
            return 0.0
        s = _score(sim['losses'], p, objective)
        all_results.append((s, dict(p), sim['losses']))
        if s > best_score[0]:
            best_score[0]  = s
            best_params[0] = dict(p)
        return -s

    gp_minimize(objective_fn, dimensions, n_calls=n_calls,
                random_state=42, verbose=False)

    all_results.sort(key=lambda x: x[0], reverse=True)
    bp = best_params[0]
    if bp is None and all_results:
        bp = all_results[0][1]
    sim_best    = _run_sim(bp, model_key, losses_flags, driving_mode) if bp else None
    best_losses = sim_best['losses'] if sim_best else (all_results[0][2] if all_results else None)
    return bp, best_losses, all_results


# ── Geometry-only sensitivity analysis (v7: full-range global sweep) ─────────
def geometry_sensitivity(base_params, losses_flags, n_points=20):
    """
    Global sensitivity: sweep each geometric parameter across its ENTIRE
    allowed range (n_points evenly spaced values) with all other parameters
    held constant, and find the absolute best value for each.

    v7 change vs V6: replaced local ±10% perturbation with a full-range sweep
    so each recommendation is a true global optimum for that single parameter.

    Returns list of (key, display_name, units, base_val, best_val, delta_W, pct)
    sorted by delta_W descending (largest improvement first).
    """
    sim_base = simulate(base_params, model='schmidt', losses_flags=losses_flags)
    if sim_base is None:
        return []
    P_base = sim_base['losses']['P_brake']

    param_meta = {
        'D_displacer':            ('Displacer diameter',    'mm',  30,    150),
        'S_displacer':            ('Displacer stroke',      'mm',  30,    200),
        'D_power':                ('Power piston diam.',    'mm',  30,    150),
        'S_power':                ('Power piston stroke',   'mm',  20,    150),
        'phi_deg':                ('Phase angle',           '°',   60,    120),
        'D_r':                    ('Regen diameter',        'mm',  20,    100),
        'L_r':                    ('Regen length',          'mm',  50,    400),
        'd_wire':                 ('Wire diameter',         'mm',  0.5,   3.0),
        # Shuttle-loss geometry — v8 addition to sensitivity sweep.
        # gap is stored in mm here for display; converted to m before simulate().
        'gap':                    ('Displacer gap',         'mm',  0.10,  0.50),
        'L_displacer_effective':  ('Effective disp. length','m',   0.05,  0.235),
    }

    results = []
    for key in GEOMETRIC_KEYS:
        if key not in param_meta:
            continue
        name, units, pmin, pmax = param_meta[key]
        base_val = base_params.get(key, PROTOTYPE.get(key))
        if base_val is None:
            continue

        # For gap: base_val is in metres in params dict; display range is mm.
        # Convert base_val to mm for sweep, convert back to m for simulate().
        gap_key = (key == 'gap')
        if gap_key:
            base_val_display = base_val * 1e3   # m → mm for display
            sweep = np.linspace(pmin, pmax, n_points)   # mm sweep
        else:
            base_val_display = base_val
            sweep = np.linspace(pmin, pmax, n_points)

        best_delta = -float('inf')
        best_val   = base_val_display

        for val in sweep:
            p = dict(base_params)
            if gap_key:
                p[key] = float(val) * 1e-3   # mm → m for physics
            else:
                p[key] = float(val)
            sim = simulate(p, model='schmidt', losses_flags=losses_flags)
            if sim is None:
                continue
            delta = sim['losses']['P_brake'] - P_base
            if delta > best_delta:
                best_delta = delta
                best_val   = float(val)   # keep in mm for gap display

        if best_delta == -float('inf'):
            continue

        pct      = best_delta / P_base * 100 if P_base > 0 else 0
        base_for_display = base_val_display if gap_key else base_val
        results.append((key, name, units, base_for_display, best_val, best_delta, pct))

    results.sort(key=lambda x: x[5], reverse=True)
    return results

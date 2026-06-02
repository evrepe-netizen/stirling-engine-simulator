"""
optimization.py — Smart optimization for Stirling engine parameters
====================================================================
Three strategies:
  1. Coarse→Fine two-stage grid search  (default, always available)
  2. Latin Hypercube Sampling (LHS)     (scipy.stats.qmc — part of scipy)
  3. Bayesian Optimization              (scikit-optimize, optional)

Geometry-only sensitivity analysis for Top-3 recommendations.
Imports only from physics.py — no Streamlit.

NOTE on excluded parameters:
  Porosity, eps_reg, eta_mech, C_leak are coupled material/manufacturing
  properties that cannot be independently optimized without updating the
  full regenerator sub-model. They are excluded from automated optimization.
  The user should modify them manually when exploring trade-offs.
"""

import numpy as np
from itertools import product as itertools_product

from physics import simulate, PROTOTYPE

# ── Parameter definitions ─────────────────────────────────────────────────────
# (display_name, key, min, max, default_step, units)
OPTIMIZABLE_PARAMS = [
    ('Displacer diameter',    'D_displacer', 30,  150,  5,   'mm'),
    ('Displacer stroke',      'S_displacer', 30,  200,  10,  'mm'),
    ('Power piston diameter', 'D_power',     30,  150,  5,   'mm'),
    ('Power piston stroke',   'S_power',     20,  150,  10,  'mm'),
    ('Phase angle',           'phi_deg',     60,  120,  15,  '°'),
    ('Regen diameter',        'D_r',         20,  100,  10,  'mm'),
    ('Regen length',          'L_r',         50,  400,  50,  'mm'),
    ('Wire diameter',         'd_wire',      0.5, 3.0,  0.5, 'mm'),
    ('Hot temperature',       'T_h',         573, 1273, 100, 'K'),
    ('Mean pressure',         'P_mean_bar',  1.0, 30.0, 5.0, 'bar'),
    ('Frequency',             'f',           5,   50,   5,   'Hz'),
]
# NOTE: porosity, eps_reg, eta_mech, C_leak are intentionally excluded.

# Geometric params for sensitivity analysis (porosity excluded — coupled property)
GEOMETRIC_KEYS = {
    'D_displacer', 'S_displacer', 'D_power', 'S_power',
    'phi_deg', 'D_r', 'L_r', 'd_wire',
}

EXCLUDED_NOTE = (
    "**Excluded from automated optimization:** porosity, regenerator effectiveness "
    "(ε_reg), mechanical efficiency (η_mech), and seal leakage coefficient (C_leak) "
    "are coupled material/manufacturing properties. Modify them manually in the sidebar "
    "when exploring trade-offs."
)


def _score(losses, objective):
    """Scalar score to maximize."""
    if losses['W_shaft'] <= 0 or losses['Q_in'] <= 0:
        return -float('inf')
    if objective == 'power':
        return losses['P_brake']
    elif objective == 'efficiency':
        return losses['eta_brake']
    else:   # balanced
        return losses['P_brake'] * losses['eta_brake']


# ── 1. Coarse → Fine grid search ─────────────────────────────────────────────
def coarse_fine_search(base_params, open_specs, objective, model_key,
                       losses_flags, progress_cb=None):
    """
    Two-stage grid search.
      Stage 1: 4 values per parameter (coarse sweep).
      Stage 2: 5 values in ±20% neighborhood of best coarse point.

    open_specs: list of (key, min_val, max_val).
    progress_cb: optional callable(fraction, message).
    Returns (best_params, best_losses, all_results_list).
    best_losses is always re-derived from best_params to guarantee consistency.
    """
    keys   = [s[0] for s in open_specs]
    bounds = [(s[1], s[2]) for s in open_specs]

    # ── Stage 1: coarse (4 values per param) ─────────────────────────────────
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
        sim = simulate(p, model=model_key, losses_flags=losses_flags)
        count += 1
        if progress_cb and count % max(1, total_c // 40) == 0:
            progress_cb(0.5 * count / total_c, f"Coarse: {count}/{total_c}")
        if sim is None:
            continue
        s = _score(sim['losses'], objective)
        if s > best_score:
            best_score = s
            best_params = dict(p)
            best_combo  = combo

    if best_combo is None:
        return None, None, []

    # ── Stage 2: fine (5 values in ±20% around best coarse point) ────────────
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
        sim = simulate(p, model=model_key, losses_flags=losses_flags)
        count_f += 1
        if progress_cb and count_f % max(1, total_f // 40) == 0:
            progress_cb(0.5 + 0.5 * count_f / total_f, f"Fine: {count_f}/{total_f}")
        if sim is None:
            continue
        s = _score(sim['losses'], objective)
        all_results.append((s, dict(p), sim['losses']))
        if s > best_score:
            best_score  = s
            best_params = dict(p)

    all_results.sort(key=lambda x: x[0], reverse=True)

    # Always re-run best_params to get consistent losses
    sim_best  = simulate(best_params, model=model_key, losses_flags=losses_flags)
    best_losses = sim_best['losses'] if sim_best else (
        all_results[0][2] if all_results else None
    )
    return best_params, best_losses, all_results


# ── 2. Latin Hypercube Sampling ───────────────────────────────────────────────
def lhs_search(base_params, open_specs, objective, model_key,
               losses_flags, n_samples=500, progress_cb=None):
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
        sim = simulate(p, model=model_key, losses_flags=losses_flags)
        if progress_cb and i % max(1, n_samples // 50) == 0:
            progress_cb(i / n_samples, f"LHS: {i}/{n_samples}")
        if sim is None:
            continue
        s = _score(sim['losses'], objective)
        all_results.append((s, dict(p), sim['losses']))
        if s > best_score:
            best_score = s
            best_params = dict(p)

    all_results.sort(key=lambda x: x[0], reverse=True)
    sim_best    = simulate(best_params, model=model_key, losses_flags=losses_flags) if best_params else None
    best_losses = sim_best['losses'] if sim_best else (all_results[0][2] if all_results else None)
    return best_params, best_losses, all_results


# ── 3. Bayesian Optimization (optional) ──────────────────────────────────────
def bayesian_search(base_params, open_specs, objective, model_key,
                    losses_flags, n_calls=60, progress_cb=None):
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
                          losses_flags, n_samples=n_calls * 8, progress_cb=progress_cb)

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
        sim = simulate(p, model=model_key, losses_flags=losses_flags)
        call_count[0] += 1
        if progress_cb and call_count[0] % 5 == 0:
            progress_cb(call_count[0] / n_calls, f"Bayesian: {call_count[0]}/{n_calls}")
        if sim is None:
            return 0.0
        s = _score(sim['losses'], objective)
        all_results.append((s, dict(p), sim['losses']))
        if s > best_score[0]:
            best_score[0] = s
            best_params[0] = dict(p)
        return -s   # skopt minimizes

    gp_minimize(objective_fn, dimensions, n_calls=n_calls,
                random_state=42, verbose=False)

    all_results.sort(key=lambda x: x[0], reverse=True)
    bp = best_params[0]
    if bp is None and all_results:
        bp = all_results[0][1]
    sim_best    = simulate(bp, model=model_key, losses_flags=losses_flags) if bp else None
    best_losses = sim_best['losses'] if sim_best else (all_results[0][2] if all_results else None)
    return bp, best_losses, all_results


# ── Geometry-only sensitivity analysis ───────────────────────────────────────
def geometry_sensitivity(base_params, losses_flags, perturbation=0.10):
    """
    Local sensitivity: perturb each geometric parameter by ±10% and measure
    impact on P_brake.  Only considers GEOMETRIC_KEYS (porosity excluded).

    Returns list of (key, display_name, units, base_val, best_val, delta_W, pct)
    sorted by delta_W descending (best improvement first).
    Only parameters where at least one direction gives a positive delta are shown.
    """
    sim_base = simulate(base_params, model='schmidt', losses_flags=losses_flags)
    if sim_base is None:
        return []
    P_base = sim_base['losses']['P_brake']

    param_meta = {
        'D_displacer': ('Displacer diameter',    'mm', 30,  150),
        'S_displacer': ('Displacer stroke',      'mm', 30,  200),
        'D_power':     ('Power piston diameter', 'mm', 30,  150),
        'S_power':     ('Power piston stroke',   'mm', 20,  150),
        'phi_deg':     ('Phase angle',           '°',  60,  120),
        'D_r':         ('Regen diameter',        'mm', 20,  100),
        'L_r':         ('Regen length',          'mm', 50,  400),
        'd_wire':      ('Wire diameter',         'mm', 0.5, 3.0),
    }

    results = []
    for key in GEOMETRIC_KEYS:
        if key not in param_meta:
            continue
        name, units, pmin, pmax = param_meta[key]
        base_val = base_params.get(key, PROTOTYPE.get(key))
        if base_val is None:
            continue

        # Try both directions, keep the one with HIGHEST delta (prefer positive)
        best_delta = -float('inf')
        best_val   = base_val

        for sign in (+1, -1):
            val = base_val * (1 + sign * perturbation)
            val = float(np.clip(val, pmin, pmax))
            if abs(val - base_val) < 1e-9:
                continue
            p   = dict(base_params); p[key] = val
            sim = simulate(p, model='schmidt', losses_flags=losses_flags)
            if sim is None:
                continue
            delta = sim['losses']['P_brake'] - P_base
            if delta > best_delta:
                best_delta = delta
                best_val   = val

        if best_delta == -float('inf'):
            continue

        pct = best_delta / P_base * 100 if P_base > 0 else 0
        results.append((key, name, units, base_val, best_val, best_delta, pct))

    # Sort: positive improvements first, then by magnitude
    results.sort(key=lambda x: x[5], reverse=True)
    return results

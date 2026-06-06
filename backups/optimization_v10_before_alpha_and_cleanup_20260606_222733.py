"""
optimization_v10.py — Smart optimization for Stirling engine parameters
========================================================================
v10 additions:
  - prototype2_search: Stage 2 — locks D_displacer, S_displacer, L_displacer
  - stage3_search:     Stage 3 — full geometry, Q_in feasibility filter
  - stage4_search:     Stage 4 — operating conditions on Prototype 2 geometry
  - STAGE2_LOCKED, STAGE4_OPEN: constant sets for UI enforcement
"""

import numpy as np
from itertools import product as itertools_product

from physics_v10 import simulate, simulate_fixed_heat, PROTOTYPE

# Keys whose optimizer values are in mm but must be converted to m for physics.
_MM_TO_M_KEYS = {'gap'}

# ── Parameter definitions ─────────────────────────────────────────────────────
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
    ('Displacer radial gap',       'gap',                   0.10,  0.50,   0.05,  'mm'),
    ('Effective displacer length', 'L_displacer_effective', 0.05,  0.235,  0.02,  'm'),
    ('Heat Input (Max/Target)',    'Q_in_max',              50,    3000,   50,    'W'),
]

GEOMETRIC_KEYS = {
    'D_displacer', 'S_displacer', 'D_power', 'S_power',
    'phi_deg', 'D_r', 'L_r', 'd_wire',
    'gap', 'L_displacer_effective',
}

EXCLUDED_NOTE = (
    "**Excluded from automated optimization:** porosity, regenerator effectiveness "
    "(ε_reg), mechanical efficiency (η_mech), and seal leakage coefficient (C_leak) "
    "are coupled material/manufacturing properties. Modify them manually in the sidebar."
)


def _to_physics_params(p):
    """Convert optimizer mm-unit keys to SI before simulate()."""
    q = dict(p)
    if 'gap' in q:
        q['gap'] = q['gap'] * 1e-3
    return q


def _score(losses, params, objective):
    """
    Scalar score to maximize.
    Applies heat_factor penalty if Q_in exceeds Q_in_max budget.
    """
    if losses['W_shaft'] <= 0 or losses['Q_in'] <= 0:
        return -float('inf')

    Q_req = losses['Q_in_W']
    Q_max = params.get('Q_in_max', 3000)
    heat_factor = min(1.0, Q_max / Q_req) if Q_req > 0 else 0.0

    P_avail = losses['P_brake']   * heat_factor
    eta_eff = losses['eta_brake'] * heat_factor

    if objective == 'power':
        return P_avail
    elif objective == 'efficiency':
        return eta_eff
    else:
        return P_avail * eta_eff


def _run_sim(p, model_key, losses_flags, driving_mode):
    """
    Run correct simulation depending on driving mode.
    Always applies _to_physics_params() for mm→m conversion.
    """
    phys_p = _to_physics_params(p)
    if driving_mode == "Fixed Heat Input (Q_in)":
        return simulate_fixed_heat(
            phys_p, phys_p.get('Q_in_max', 500),
            model=model_key, losses_flags=losses_flags)
    return simulate(phys_p, model=model_key, losses_flags=losses_flags)


# ── 1. Coarse → Fine grid search ─────────────────────────────────────────────
def coarse_fine_search(base_params, open_specs, objective, model_key,
                       losses_flags, driving_mode="Fixed Hot Temperature (T_h)",
                       progress_cb=None):
    """
    Two-stage grid search.
      Stage 1: 4 values per parameter (coarse sweep).
      Stage 2: 5 values in ±20% neighborhood of best coarse point.
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
        all_results[0][2] if all_results else None)
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


# ── 3. Bayesian Optimization ──────────────────────────────────────────────────
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


# ── Geometry-only sensitivity analysis ───────────────────────────────────────
def geometry_sensitivity(base_params, losses_flags, n_points=20):
    """
    Global sensitivity: sweep each geometric parameter across its full allowed
    range. Returns list of (key, name, units, base_val, best_val, delta_W, pct).
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

        gap_key = (key == 'gap')
        if gap_key:
            base_val_display = base_val * 1e3
            sweep = np.linspace(pmin, pmax, n_points)
        else:
            base_val_display = base_val
            sweep = np.linspace(pmin, pmax, n_points)

        best_delta = -float('inf')
        best_val   = base_val_display

        for val in sweep:
            p = dict(base_params)
            if gap_key:
                p[key] = float(val) * 1e-3
            else:
                p[key] = float(val)
            sim = simulate(p, model='schmidt', losses_flags=losses_flags)
            if sim is None:
                continue
            delta = sim['losses']['P_brake'] - P_base
            if delta > best_delta:
                best_delta = delta
                best_val   = float(val)

        if best_delta == -float('inf'):
            continue

        pct = best_delta / P_base * 100 if P_base > 0 else 0
        base_for_display = base_val_display if gap_key else base_val
        results.append((key, name, units, base_for_display, best_val, best_delta, pct))

    results.sort(key=lambda x: x[5], reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE FUNCTIONS (v10 addition)
# ═══════════════════════════════════════════════════════════════════════════════

STAGE2_LOCKED = ('D_displacer', 'S_displacer', 'L_displacer')
# These three parameters represent the existing physical displacer.
# Prototype 2 must never change them.

STAGE4_OPEN = ('P_mean_bar', 'T_h', 'f')
# Stage 4 opens operating conditions on fixed Prototype 2 geometry.


def _enforce_stage2_locks(open_specs, base_params):
    """
    Verify that no locked Stage 2 parameter appears in open_specs.
    Raises ValueError with a clear message if violated.
    """
    open_keys = [s[0] for s in open_specs]
    for locked in STAGE2_LOCKED:
        if locked in open_keys:
            raise ValueError(
                f"Parameter '{locked}' is locked in Stage 2 (Prototype 2). "
                f"It represents the existing physical displacer and cannot be optimized. "
                f"Remove it from open_specs."
            )


def prototype2_search(base_params, open_specs, objective, model_key,
                      losses_flags, method='coarse_fine',
                      n_samples=300, progress_cb=None):
    """
    Stage 2 — Prototype 2 optimization.

    Locks D_displacer, S_displacer, L_displacer to their base_params values.
    Optimizes everything else specified in open_specs.

    Parameters
    ----------
    base_params : dict  must contain the locked displacer parameters
    open_specs  : list  of (key, min_val, max_val) — must NOT include locked params
    objective   : str   'power', 'efficiency', or 'balanced'
    model_key   : str   'schmidt' or 'adiabatic'
    losses_flags: dict
    method      : str   'coarse_fine', 'lhs', or 'bayesian'
    n_samples   : int   used if method='lhs'
    progress_cb : callable or None

    Returns
    -------
    (best_params, best_losses, all_results)
    best_params will have STAGE2_LOCKED values identical to base_params.
    """
    _enforce_stage2_locks(open_specs, base_params)

    if method == 'lhs':
        best_params, best_losses, all_results = lhs_search(
            base_params, open_specs, objective, model_key,
            losses_flags, n_samples=n_samples, progress_cb=progress_cb)
    elif method == 'bayesian':
        best_params, best_losses, all_results = bayesian_search(
            base_params, open_specs, objective, model_key,
            losses_flags, n_calls=max(20, n_samples // 5),
            progress_cb=progress_cb)
    else:
        best_params, best_losses, all_results = coarse_fine_search(
            base_params, open_specs, objective, model_key,
            losses_flags, progress_cb=progress_cb)

    # Post-check: verify locked parameters are unchanged
    if best_params is not None:
        for locked in STAGE2_LOCKED:
            if locked in base_params:
                assert abs(best_params.get(locked, 0) - base_params[locked]) < 1e-9, \
                    f"BUG: {locked} was modified during Prototype 2 optimization"

    return best_params, best_losses, all_results


def stage3_search(base_params, open_specs, objective, Q_in_max,
                  model_key='schmidt', losses_flags=None,
                  method='lhs', n_samples=400, progress_cb=None):
    """
    Stage 3 — Full geometry optimization under fixed operating conditions.

    Gas = Air, P = 1 bar, f = 10 Hz (fixed in base_params before calling).
    Q_in <= Q_in_max is a hard feasibility requirement for final results.

    Returns
    -------
    (best_params, best_losses, all_results_feasible, all_results_raw)
    """
    if losses_flags is None:
        losses_flags = dict(flow=True, regen_imp=True, mechanical=True,
                            wall_cond=True, leakage=True, shuttle=False)

    p_with_budget = dict(base_params)
    p_with_budget['Q_in_max'] = Q_in_max

    if method == 'lhs':
        best_params, best_losses, all_results_raw = lhs_search(
            p_with_budget, open_specs, objective, model_key,
            losses_flags, n_samples=n_samples, progress_cb=progress_cb)
    elif method == 'bayesian':
        best_params, best_losses, all_results_raw = bayesian_search(
            p_with_budget, open_specs, objective, model_key,
            losses_flags, n_calls=max(20, n_samples // 5),
            progress_cb=progress_cb)
    else:
        best_params, best_losses, all_results_raw = coarse_fine_search(
            p_with_budget, open_specs, objective, model_key,
            losses_flags, progress_cb=progress_cb)

    all_results_feasible = [
        (score, p, l) for score, p, l in all_results_raw
        if l.get('Q_in_W', 9999) <= Q_in_max
    ]
    all_results_feasible.sort(key=lambda x: x[0], reverse=True)

    return best_params, best_losses, all_results_feasible, all_results_raw


def stage4_search(proto2_params, objective, Q_in_max=1500.0,
                  P_max=10.0, f_max=25.0, model_key='schmidt',
                  losses_flags=None, n_samples=200, progress_cb=None):
    """
    Stage 4 — Operating conditions optimization on Prototype 2 geometry.

    Geometry fixed to proto2_params. Opens P_mean_bar, T_h, f.
    Runs separately for Air, Helium, Hydrogen.

    Returns
    -------
    dict with keys 'Air', 'Helium', 'Hydrogen'
    Each value: dict with best_params, best_losses, all_feasible, all_raw.
    """
    if losses_flags is None:
        losses_flags = dict(flow=True, regen_imp=True, mechanical=True,
                            wall_cond=True, leakage=True, shuttle=False)

    open_specs = [
        ('P_mean_bar', 1.0, P_max),
        ('T_h',        573, 1273),
        ('f',          5.0, f_max),
    ]

    results = {}
    gases = ['Air', 'Helium', 'Hydrogen']

    for idx, gas_name in enumerate(gases):
        if progress_cb:
            progress_cb(idx / len(gases), f"Optimizing for {gas_name}...")

        p = dict(proto2_params)
        p['gas']       = gas_name
        p['Q_in_max']  = Q_in_max

        _, _, all_feasible, all_raw = stage3_search(
            p, open_specs, objective, Q_in_max,
            model_key=model_key, losses_flags=losses_flags,
            method='lhs', n_samples=n_samples,
            progress_cb=None)

        best = all_feasible[0] if all_feasible else (None, None, None)
        results[gas_name] = {
            'best_params':  best[1] if best[0] is not None else None,
            'best_losses':  best[2] if best[0] is not None else None,
            'all_feasible': all_feasible,
            'all_raw':      all_raw,
        }

    if progress_cb:
        progress_cb(1.0, "Stage 4 complete.")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# V10 named-design helpers: Max Power / Max Efficiency / Balanced
# ─────────────────────────────────────────────────────────────────────────────

def _loss_eta(losses):
    """Return net/brake efficiency from a losses dict."""
    if losses is None:
        return -1e99
    if 'eta_brake' in losses:
        return losses.get('eta_brake', -1e99)
    q = losses.get('Q_in_W', 0)
    return losses.get('P_brake', -1e99) / q if q else -1e99


def _balanced_score_from_losses(losses, ref_losses=None, w_P=0.5):
    """
    Dimensionless balanced score:
    Score = w_P*(P/P_ref) + (1-w_P)*(eta/eta_ref)

    If reference values are unavailable, use raw P and eta normalization
    from 1.0 to avoid crashing.
    """
    if losses is None:
        return -1e99

    P = losses.get('P_brake', -1e99)
    eta = _loss_eta(losses)

    if ref_losses:
        P_ref = max(ref_losses.get('P_brake', 0), 1e-12)
        eta_ref = max(_loss_eta(ref_losses), 1e-12)
    else:
        P_ref = 1.0
        eta_ref = 1.0

    return w_P * (P / P_ref) + (1.0 - w_P) * (eta / eta_ref)


def select_named_designs(all_results, Q_in_max=1500.0, ref_losses=None, w_P=0.5):
    """
    Select final named designs from feasible candidates only.

    Final named designs must satisfy:
    Q_in_W <= Q_in_max
    P_brake > 0
    """
    feasible = []
    infeasible = []

    for item in all_results or []:
        if not isinstance(item, (tuple, list)) or len(item) < 3:
            continue

        score, params, losses = item[0], item[1], item[2]

        if losses is None or params is None:
            continue

        q = losses.get('Q_in_W', None)
        P = losses.get('P_brake', None)

        if q is None or P is None:
            continue

        if q <= Q_in_max and P > 0:
            feasible.append((score, params, losses))
        else:
            infeasible.append((score, params, losses))

    if not feasible:
        return {
            'max_power': None,
            'max_efficiency': None,
            'balanced': None,
            'feasible_results': [],
            'infeasible_results': infeasible,
        }

    def eta_of(item):
        losses = item[2]
        if 'eta_brake' in losses:
            return losses['eta_brake']
        return losses.get('P_brake', 0) / max(losses.get('Q_in_W', 1e-12), 1e-12)

    def P_of(item):
        return item[2].get('P_brake', -1e99)

    def balanced_of(item):
        return _balanced_score_from_losses(item[2], ref_losses=ref_losses, w_P=w_P)

    max_power = max(feasible, key=P_of)
    max_efficiency = max(feasible, key=eta_of)
    balanced = max(feasible, key=balanced_of)

    return {
        'max_power': {
            'score': max_power[0],
            'params': max_power[1],
            'losses': max_power[2],
        },
        'max_efficiency': {
            'score': max_efficiency[0],
            'params': max_efficiency[1],
            'losses': max_efficiency[2],
        },
        'balanced': {
            'score': balanced[0],
            'params': balanced[1],
            'losses': balanced[2],
        },
        'feasible_results': feasible,
        'infeasible_results': infeasible,
    }


def stage3_search_named(base_params, open_specs, Q_in_max,
                        model_key='schmidt', losses_flags=None,
                        method='lhs', n_samples=400, progress_cb=None,
                        w_P=0.5, ref_losses=None):
    """
    V10 Stage 3 wrapper:
    Runs a broad geometry search once, then selects:
    - Max Power
    - Max Efficiency
    - Balanced

    Operating conditions should already be fixed in base_params:
    gas=Air, P_mean_bar=1, f=10.
    """
    # Use balanced as the internal search objective to sample tradeoff reasonably.
    bp, bl, feasible, raw = stage3_search(
        base_params, open_specs, 'balanced', Q_in_max,
        model_key=model_key,
        losses_flags=losses_flags,
        method=method,
        n_samples=n_samples,
        progress_cb=progress_cb
    )

    named = select_named_designs(raw, Q_in_max=Q_in_max,
                                 ref_losses=ref_losses, w_P=w_P)
    named['raw_results'] = raw
    named['best_params'] = named['balanced']['params'] if named['balanced'] else bp
    named['best_losses'] = named['balanced']['losses'] if named['balanced'] else bl
    return named


def stage4_search_named(proto2_params, Q_in_max=1500.0,
                        P_max=10.0, f_max=25.0,
                        model_key='schmidt', losses_flags=None,
                        n_samples=200, progress_cb=None,
                        w_P=0.5, ref_losses=None):
    """
    V10 Stage 4 wrapper:
    For each gas, optimize operating conditions on fixed Prototype 2 geometry
    and return three feasible named designs:
    - Max Power
    - Max Efficiency
    - Balanced

    Output:
    {
      'Air': {'max_power':..., 'max_efficiency':..., 'balanced':..., ...},
      'Helium': {...},
      'Hydrogen': {...}
    }
    """
    if losses_flags is None:
        losses_flags = dict(flow=True, regen_imp=True, mechanical=True,
                            wall_cond=True, leakage=True, shuttle=False)

    open_specs = [
        ('P_mean_bar', 1.0, P_max),
        ('T_h',        573, 1273),
        ('f',          5.0, f_max),
    ]

    gases = ['Air', 'Helium', 'Hydrogen']
    results = {}

    for idx, gas_name in enumerate(gases):
        if progress_cb:
            progress_cb(idx / len(gases), f"Optimizing {gas_name}...")

        p = dict(proto2_params)
        p['gas'] = gas_name
        p['Q_in_max'] = Q_in_max

        bp, bl, feasible, raw = stage3_search(
            p, open_specs, 'balanced', Q_in_max,
            model_key=model_key,
            losses_flags=losses_flags,
            method='lhs',
            n_samples=n_samples,
            progress_cb=None
        )

        named = select_named_designs(raw, Q_in_max=Q_in_max,
                                     ref_losses=ref_losses, w_P=w_P)
        named['raw_results'] = raw
        results[gas_name] = named

    if progress_cb:
        progress_cb(1.0, "Stage 4 named optimization complete.")

    return results


"""US sensitivity experiments: runtime + error vs. grid size / #levels / #forecasters.

Run from ``notebooks/run_experiments.ipynb`` (step 3); plotted as Figures 5 and 8.

Three timing axes (all US, week 4, eta_multiplier=1.0):
  1. grid ``unit``       -> also the *error* sensitivity (pseudo-omni objective)
  2. number of quantile levels (symmetric, includes 0.5)
  3. number of base forecasters (top-k by data availability)

The error objective is computed in the **original count scale** so grid sizes
are comparable (see :func:`metrics.pseudo_omni_objective`).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Callable, List, Optional, Sequence

import numpy as np

import config as cfg
import data_io
from algorithms import run_omni_v2, run_ql_pb_opt, _grid_from_preds
from metrics import pseudo_omni_objective, mean_ql_objective

# Bump when the *content* of result rows changes (e.g. new fields), so cached
# rows from older versions are recomputed rather than reused.
ROWS_VERSION = 3


# ---------------------------------------------------------------------------
# Result caching (hash of parameters -> pickle in the results folder)
# ---------------------------------------------------------------------------

def _params_hash(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def load_or_run(prefix: str, params: dict, compute_fn: Callable[[], object],
                cache_dir: Optional[str] = None, force: bool = False):
    """Return cached rows for ``params`` if present, else compute and cache.

    The cache file is ``{cache_dir}/{prefix}_{hash(params)}.pkl`` containing
    ``{"params": params, "rows": <result>}``. Set ``force=True`` to recompute
    and overwrite. Any change to ``params`` yields a new hash (a fresh cache).
    """
    cache_dir = cache_dir or cfg.SENSITIVITY_DIR
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{prefix}_{_params_hash(params)}.pkl")
    if not force and os.path.exists(path):
        cached = pickle.load(open(path, "rb"))
        print(f"[cache hit ] {prefix}: {os.path.basename(path)}")
        return cached["rows"]
    rows = compute_fn()
    pickle.dump({"params": params, "rows": rows}, open(path, "wb"))
    print(f"[cache miss] {prefix}: saved {os.path.basename(path)}")
    return rows


def _timed_omni_v2(preds_TNF, y_arr, unit, alpha_list, eta_multiplier,
                   seed, n_repeats, thetas=None):
    """Run omni-v2 ``n_repeats`` times (fixed seed) and return
    (last_result, mean_runtime, all_runtimes)."""
    runtimes: List[float] = []
    res = None
    for _ in range(n_repeats):
        res = run_omni_v2(preds_TNF, y_arr, unit, alpha_list,
                          eta_multiplier=eta_multiplier, seed=seed, thetas=thetas)
        runtimes.append(res["runtime_sec"])
    return res, float(np.mean(runtimes)), runtimes


# ---------------------------------------------------------------------------
# Axis 1: grid unit  (runtime AND error)
# ---------------------------------------------------------------------------

def run_grid_unit_experiment(loaded: dict, units: Optional[Sequence[int]] = None,
                             eta_multiplier=cfg.SENSITIVITY_ETA_MULTIPLIER,
                             seed: int = 0, n_repeats: int = 1) -> List[dict]:
    """Run omni-v2 per grid unit and score it with the pseudo-omni objective.

    ``eta_multiplier`` may be a single number or a list. A scalar is converted to
    a one-entry list (so the reported ``objective`` equals that single run — the
    US study passes ``1.0`` and thus keeps the fixed-``c_O`` behaviour). With a
    list, every learning rate is run for each grid unit and the reported
    ``objective`` / ``objective_trace`` are the **best (minimum) over the sweep**,
    so the curve reflects the best-achievable error per grid; per-eta values are
    also kept in ``objective_by_eta`` / ``runtime_by_eta``.
    """
    units = list(units or cfg.SENSITIVITY_UNITS)
    eta_list = [float(eta_multiplier)] if np.ndim(eta_multiplier) == 0 else list(eta_multiplier)
    preds, y, alpha = loaded["preds_TNF"], loaded["y_arr"], loaded["alpha_list"]
    rows = []
    for unit in units:
        etas, runtimes, objs, argmaxes, traces = [], [], [], [], []
        preds_by_eta, mql_objs, mql_traces, stats_by_eta = [], [], [], []
        for eta in eta_list:
            res, rt, rts = _timed_omni_v2(preds, y, unit, alpha, eta, seed, n_repeats)
            obj = pseudo_omni_objective(res["phat_history_orig"], y, preds, alpha)
            mql = mean_ql_objective(res["phat_history_orig"], y, preds, alpha)
            etas.append(eta); runtimes.append(rt)
            objs.append(obj["objective"]); argmaxes.append(obj["argmax_alpha"])
            traces.append(obj["objective_trace"])
            preds_by_eta.append(res["phat_history_orig"])
            mql_objs.append(mql["objective"]); mql_traces.append(mql["objective_trace"])
            # Everything we can cheaply keep per learning rate (runs are slow).
            stats_by_eta.append({
                "eta_multiplier": eta, "eta": res["eta"],
                "runtime_sec": rt, "runtimes": rts,
                "pseudo_omni": obj,            # full dict: relative + absolute + per-level
                "mean_ql": mql,                # full dict: relative + absolute + per-forecaster
                # algorithm-internal traces (scaled grid units, as returned by run_omni_v2)
                "omni_score_trace": res["omni_score_trace"],          # (T,) grid omni error (abs)
                "minimax_value_history": res["minimax_value_history"],  # (T,)
                "pb_loss_history_orig": res["pb_loss_history"] * unit * res["m"],  # (T, N) expected pinball, counts
            })
        best = int(np.argmin(objs))
        mql_best = int(np.argmin(mql_objs))
        abs_objs = [s["pseudo_omni"]["objective_abs"] for s in stats_by_eta]
        rows.append({
            "unit": unit, "m": res["m"], "thetas": res["thetas"],
            "runtime_sec": runtimes[best], "runtime_by_eta": runtimes,
            "etas": etas, "objective_by_eta": objs,
            "objective": objs[best], "best_eta": etas[best],
            "argmax_alpha": argmaxes[best], "objective_trace": traces[best],
            # absolute pseudo-omni error (max over levels of our mean pinball)
            "objective_abs_by_eta": abs_objs, "objective_abs": min(abs_objs),
            # predictions (original counts) so other metrics can be computed later
            "preds_orig_by_eta": preds_by_eta,
            # level-averaged QL regret vs. best single forecaster
            "mean_ql_objective_by_eta": mql_objs,
            "mean_ql_objective": mql_objs[mql_best], "mean_ql_best_eta": etas[mql_best],
            "mean_ql_objective_trace": mql_traces[mql_best],
            # full per-eta statistics
            "stats_by_eta": stats_by_eta,
        })
        print(f"unit={unit:<7g}  m={res['m']:6d}  time={runtimes[best]:7.2f}s  "
              f"objective={objs[best]:10.3f} (eta={etas[best]:g})  "
              f"(worst level alpha={argmaxes[best]})  "
              f"mean_ql={mql_objs[mql_best]:9.3f} (eta={etas[mql_best]:g})")
    return rows


def run_grid_unit_experiment_cached(loaded: dict, units: Optional[Sequence[int]] = None,
                                    eta_multiplier: float = cfg.SENSITIVITY_ETA_MULTIPLIER,
                                    seed: int = 0, n_repeats: int = 1,
                                    cache_dir: Optional[str] = None,
                                    force: bool = False) -> List[dict]:
    """Cached :func:`run_grid_unit_experiment`.

    Keyed by (geo, week, units, eta_multiplier, seed, n_repeats). On a cache hit
    the stored ``grid_rows`` are loaded from the results folder instead of
    recomputing the (slow) omni-v2 sweep; pass ``force=True`` to recompute.
    """
    units = list(units or cfg.SENSITIVITY_UNITS)
    eta_key = float(eta_multiplier) if np.ndim(eta_multiplier) == 0 else [float(e) for e in eta_multiplier]
    params = {
        "experiment": "grid_unit_omni_v2",
        "geo": loaded.get("geo"), "week": loaded.get("week"),
        "units": units, "eta_multiplier": eta_key,
        "seed": seed, "n_repeats": n_repeats,
        "rows_version": ROWS_VERSION,
    }
    if loaded.get("dataset") is not None:
        params["dataset"] = loaded.get("dataset")
    return load_or_run(
        "grid_unit_omni_v2", params,
        lambda: run_grid_unit_experiment(loaded, units=units, eta_multiplier=eta_multiplier,
                                         seed=seed, n_repeats=n_repeats),
        cache_dir=cache_dir, force=force,
    )


# ---------------------------------------------------------------------------
# Axis 1 (baselines): QL-opt / Pinball-opt error + runtime vs. grid unit
# ---------------------------------------------------------------------------

def run_grid_unit_qlpb_experiment(loaded: dict, units: Optional[Sequence[int]] = None,
                                  eta_multiplier=1.0, round_Y_F: bool = True,
                                  seed: int = 0, n_repeats: int = 1) -> List[dict]:
    """Run ``run_ql_pb_opt`` per grid unit over a sweep of learning rates and
    score both QL-opt and Pinball-opt with the same pseudo-omni objective as
    omni-v2.

    ``eta_multiplier`` may be a single number or a list. A scalar is converted to
    a one-entry list; the function then loops over every learning rate for each
    grid unit. The reported ``pinball_objective`` / ``ql_objective`` (and their
    traces) are the **best (minimum) over the learning-rate sweep**, so the
    baselines are tuned before being compared to omni-v2; per-eta values are also
    kept in ``*_by_eta``.

    Predictions are produced on the rounded grid (``round_Y_F=True`` -> each grid
    unit yields a different rounded target/base forecast) and scored in the
    ORIGINAL count scale (``preds * unit`` vs. un-rounded Y / forecasts), so the
    curve overlays directly on the omni-v2 error-vs-unit plot.

    ``compute_omni_traces=False`` avoids the huge base-forecaster elementary-score
    tensor (~15 GB at unit=25); the objective only needs the predictions.
    """
    units = list(units or cfg.SENSITIVITY_UNITS)
    eta_list = [float(eta_multiplier)] if np.ndim(eta_multiplier) == 0 else list(eta_multiplier)
    preds, y, alpha = loaded["preds_TNF"], loaded["y_arr"], loaded["alpha_list"]

    rows = []
    for unit in units:
        etas, runtimes, pin_objs, ql_objs = [], [], [], []
        pin_traces, ql_traces = [], []
        pin_preds, ql_preds = [], []
        pin_mql, ql_mql, pin_mql_traces, ql_mql_traces = [], [], [], []
        stats_by_eta = []
        for eta in eta_list:
            rts: List[float] = []
            res = None
            for _ in range(n_repeats):
                res = run_ql_pb_opt(preds, y, unit, alpha, eta_multiplier=eta,
                                    round_Y_F=round_Y_F, compute_omni_traces=False)
                rts.append(res["runtime_sec"])
            pin = pseudo_omni_objective(res["pinball_preds_history_orig"], y, preds, alpha)
            qlo = pseudo_omni_objective(res["ql_preds_history_orig"], y, preds, alpha)
            pin_m = mean_ql_objective(res["pinball_preds_history_orig"], y, preds, alpha)
            ql_m = mean_ql_objective(res["ql_preds_history_orig"], y, preds, alpha)
            etas.append(eta)
            runtimes.append(float(np.mean(rts)))
            pin_objs.append(pin["objective"]); pin_traces.append(pin["objective_trace"])
            ql_objs.append(qlo["objective"]); ql_traces.append(qlo["objective_trace"])
            pin_preds.append(res["pinball_preds_history_orig"])
            ql_preds.append(res["ql_preds_history_orig"])
            pin_mql.append(pin_m["objective"]); pin_mql_traces.append(pin_m["objective_trace"])
            ql_mql.append(ql_m["objective"]); ql_mql_traces.append(ql_m["objective_trace"])
            stats_by_eta.append({
                "eta_multiplier": eta, "eta": res["eta"],
                "runtime_sec": float(np.mean(rts)), "runtimes": rts,
                "pinball": {"pseudo_omni": pin, "mean_ql": pin_m},   # full dicts
                "ql":      {"pseudo_omni": qlo, "mean_ql": ql_m},
                "ql_v_history": res["ql_v_history"],                  # (T, F) QL-opt weights
            })

        pin_best = int(np.argmin(pin_objs))
        ql_best = int(np.argmin(ql_objs))
        pin_mql_best = int(np.argmin(pin_mql))
        ql_mql_best = int(np.argmin(ql_mql))
        pin_abs = [s["pinball"]["pseudo_omni"]["objective_abs"] for s in stats_by_eta]
        ql_abs = [s["ql"]["pseudo_omni"]["objective_abs"] for s in stats_by_eta]
        rows.append({
            "unit": unit, "m": res["m"],
            "runtime_sec": float(np.mean(runtimes)),
            "etas": etas,
            "runtime_by_eta": runtimes,
            "pinball_objective_by_eta": pin_objs,
            "ql_objective_by_eta": ql_objs,
            # best over the learning-rate sweep (used for the overlay plot)
            "pinball_objective": pin_objs[pin_best], "pinball_best_eta": etas[pin_best],
            "ql_objective": ql_objs[ql_best], "ql_best_eta": etas[ql_best],
            "pinball_objective_trace": pin_traces[pin_best],
            "ql_objective_trace": ql_traces[ql_best],
            # absolute pseudo-omni error (max over levels of mean pinball)
            "pinball_objective_abs_by_eta": pin_abs, "pinball_objective_abs": min(pin_abs),
            "ql_objective_abs_by_eta": ql_abs, "ql_objective_abs": min(ql_abs),
            # predictions (original counts) per eta so other metrics can be computed later
            "pinball_preds_orig_by_eta": pin_preds,
            "ql_preds_orig_by_eta": ql_preds,
            # level-averaged QL regret vs. best single forecaster (best over the sweep)
            "pinball_mean_ql_by_eta": pin_mql, "ql_mean_ql_by_eta": ql_mql,
            "pinball_mean_ql": pin_mql[pin_mql_best], "pinball_mean_ql_best_eta": etas[pin_mql_best],
            "ql_mean_ql": ql_mql[ql_mql_best], "ql_mean_ql_best_eta": etas[ql_mql_best],
            "pinball_mean_ql_trace": pin_mql_traces[pin_mql_best],
            "ql_mean_ql_trace": ql_mql_traces[ql_mql_best],
            # full per-eta statistics
            "stats_by_eta": stats_by_eta,
        })
        print(f"unit={unit:<7g}  m={res['m']:6d}  |  "
              f"pinball_obj={pin_objs[pin_best]:9.3f} (eta={etas[pin_best]:g})  "
              f"ql_obj={ql_objs[ql_best]:9.3f} (eta={etas[ql_best]:g})  |  "
              f"pinball_mean_ql={pin_mql[pin_mql_best]:9.3f} (eta={etas[pin_mql_best]:g})")
    return rows


# ---------------------------------------------------------------------------
# Axis 2: number of quantile levels  (runtime)
# ---------------------------------------------------------------------------

def run_quantile_experiment(loaded: dict, k_list: Optional[Sequence[int]] = None,
                            unit: Optional[int] = None,
                            eta_multiplier: float = cfg.SENSITIVITY_ETA_MULTIPLIER,
                            seed: int = 0, n_repeats: int = 1) -> List[dict]:
    k_list = list(k_list or cfg.SENSITIVITY_N_QUANTILES)
    full_alpha = np.asarray(loaded["alpha_list"])
    preds, y = loaded["preds_TNF"], loaded["y_arr"]
    unit = unit or loaded["unit"]
    rows = []
    for k in k_list:
        sub = cfg.symmetric_quantile_subset(k, full_alpha)
        idx = np.array([int(np.where(full_alpha == a)[0][0]) for a in sub])
        preds_sub = preds[:, idx, :]
        res, rt, rts = _timed_omni_v2(preds_sub, y, unit, sub, eta_multiplier, seed, n_repeats)
        rows.append({"n_quantiles": k, "unit": unit, "m": res["m"],
                     "runtime_sec": rt, "runtimes": rts, "alphas": list(np.round(sub, 3))})
        print(f"n_quantiles={k:3d}  m={res['m']:6d}  time={rt:7.2f}s")
    return rows


# ---------------------------------------------------------------------------
# Axis 3: number of base forecasters  (runtime)
# ---------------------------------------------------------------------------

def run_forecaster_experiment(geo: str = cfg.SENSITIVITY_GEO,
                              week: int = cfg.SENSITIVITY_WEEK,
                              fracs: Optional[Sequence[float]] = None,
                              unit: Optional[int] = None,
                              eta_multiplier: float = cfg.SENSITIVITY_ETA_MULTIPLIER,
                              seed: int = 0, n_repeats: int = 1,
                              fix_grid: bool = True) -> List[dict]:
    """Sweep the number of base forecasters (top-k by availability).

    With ``fix_grid=True`` the theta grid is computed once from the full
    forecaster set and reused for every subset, so runtime isolates the effect
    of F (rather than an incidental change in the grid size m).
    """
    fracs = list(fracs or cfg.SENSITIVITY_FORECASTER_FRACS)
    ranking = data_io.forecaster_availability_ranking(geo, week)
    F_total = len(ranking)

    full = data_io.load_geo_arrays(geo, week)
    unit = unit or full["unit"]
    thetas_fixed = _grid_from_preds(full["preds_TNF"], unit)[0] if fix_grid else None

    rows = []
    for frac in fracs:
        k = max(1, int(round(frac * F_total)))
        names = ranking[:k]
        loaded = data_io.load_geo_arrays(geo, week, forecaster_names=names)
        res, rt, rts = _timed_omni_v2(loaded["preds_TNF"], loaded["y_arr"], unit,
                                      loaded["alpha_list"], eta_multiplier, seed,
                                      n_repeats, thetas=thetas_fixed)
        rows.append({"frac": frac, "n_forecasters": k, "unit": unit, "m": res["m"],
                     "runtime_sec": rt, "runtimes": rts, "forecasters": names})
        print(f"frac={frac:.0%}  n_forecasters={k:3d}/{F_total}  m={res['m']:6d}  time={rt:7.2f}s")
    return rows

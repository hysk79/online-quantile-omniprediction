"""Online quantile-omniprediction algorithms.

- :func:`run_omni_v2`   - Algorithm 1 (omniprediction under all proper losses
  for multiple quantile levels, grid-based); "Our algorithm" in the paper.
- :func:`run_omni_wql`  - two-player-game omniprediction for weighted quantile
  losses (grid-free; Appendix "Omniprediction under weighted quantile losses").
- :func:`run_ql_pb_opt` - Hedge on each quantile level in parallel; its
  ``pinball_*`` outputs are the "Hedge-QL" baseline in the paper.

Every algorithm takes a preassembled ``(T, N, F)`` prediction array in original
scale (built once by ``data_io.assemble_preds_array``) plus a target vector.
Each ``run_*`` returns a results dict that also carries ``runtime_sec`` (the
wall-clock time of the online loop).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from metrics import (
    elementary_scores_grid_N,
    elementary_scores_grid_N_F,
    elementary_scores_grid_T_N,
    omni_error_from_scores,
    pinball_loss,
    ql_error_from_pb_loss,
    wql_omni_error_from_pb_loss,
)
from solver_omni_v2 import multi_q_minmax_solver_v2, minimax_value_neg
from solver_wql import efficeint_solve_weighted_hinge_split

# If the full (T, N, m, F) indicator tensor would exceed this many entries, we
# compute indicators per time step instead of precomputing them (keeps memory
# bounded for fine grids, e.g. US at unit=25 -> ~1e9 booleans).
_PRECOMPUTE_INDICATOR_LIMIT = 3e8


def _grid_from_preds(preds_TNF: np.ndarray, unit: int):
    """Replicate the original theta grid construction from forecast extremes.

    ``forecast_min`` uses the lowest quantile level, ``forecast_max`` the
    highest (columns are assumed sorted by ascending alpha).
    """
    forecast_min = float(preds_TNF[:, 0, :].min())
    forecast_max = float(preds_TNF[:, -1, :].max())
    Y_rounded_min = min(int(np.floor(forecast_min / unit)), 0)
    Y_rounded_max = int(np.ceil(forecast_max / unit))
    m = Y_rounded_max - Y_rounded_min
    thetas = np.arange(Y_rounded_min, Y_rounded_max) + 0.5
    return thetas, m


# ============================================================================
# Grid-based multi-quantile omniprediction (v2) -- the main algorithm
# ============================================================================

def run_omni_v2(preds_TNF: np.ndarray, y_arr: np.ndarray, unit: int,
                alpha_list, eta_multiplier: float = 1.0,
                precompute_indicators="auto", seed: Optional[int] = None,
                thetas: Optional[np.ndarray] = None,
                verbose: bool = False) -> dict:
    """Run grid-based multi-quantile omniprediction (formerly
    ``omniprediction_multiq_online_v2``).

    Parameters
    ----------
    preds_TNF : (T, N, F) base forecasts, ORIGINAL count scale.
    y_arr : (T,) targets, ORIGINAL count scale.
    unit : grid spacing (counts) used to discretize the outcome space.
    alpha_list : (N,) quantile levels, ascending.
    eta_multiplier : Hedge learning-rate multiplier.
    precompute_indicators : "auto" | True | False. Whether to materialize the
        full (T, N, m, F) indicator tensor (faster) or recompute per step
        (lower memory).
    seed : optional RNG seed for the randomized per-step prediction.
    thetas : optional explicit scaled theta grid; overrides the grid derived
        from ``preds_TNF``. Useful to hold the grid fixed while varying the
        forecaster set (so runtime reflects F, not an incidental change in m).
    """
    if seed is not None:
        np.random.seed(seed)

    alpha_list = np.asarray(alpha_list, dtype=np.float64)
    assert alpha_list.ndim == 1
    preds_TNF = np.asarray(preds_TNF, dtype=np.float64)
    y_arr = np.asarray(y_arr, dtype=np.float64)

    T, N, F = preds_TNF.shape
    assert alpha_list.shape[0] == N
    assert y_arr.shape[0] == T

    if thetas is None:
        thetas, m = _grid_from_preds(preds_TNF, unit)
    else:
        thetas = np.asarray(thetas, dtype=np.float64)
        m = thetas.shape[0]
    eta = eta_multiplier * np.sqrt(np.log(m * N * F) / T)

    # Scale + round to the integer grid once (vectorized).
    Y_scaled = np.round(y_arr / unit)
    preds_scaled = np.round(preds_TNF / unit)  # (T, N, F)

    if precompute_indicators == "auto":
        precompute_indicators = (T * N * m * F) <= _PRECOMPUTE_INDICATOR_LIMIT

    if verbose:
        print("=" * 70)
        print("OMNIPREDICTION MULTI-QUANTILE (v2)")
        print(f"  T={T}, N={N}, F={F}, unit={unit}, m={m}")
        print(f"  eta_multiplier={eta_multiplier}, eta={eta:.5f}")
        print(f"  precompute_indicators={precompute_indicators}")

    indicators_TNmF = None
    if precompute_indicators:
        indicators_TNmF = thetas[None, None, :, None] < preds_scaled[:, :, None, :]

    # Uniform initial weights over (level, theta, forecaster).
    w = np.ones((N, m, F)) / (N * m * F)

    phat_history = np.zeros((T, N))
    k_star_history = np.zeros((T, N))
    k_star_prob_history = np.zeros((T, N))
    minimax_value_history = np.zeros((T,))
    omni_error_history = np.zeros((T, N, m))
    pb_loss_history = np.zeros((T, N))

    t0 = time.perf_counter()
    for t in range(T):
        y_t = Y_scaled[t]
        forecaster_preds = preds_scaled[t, :, :]  # (N, F)

        if indicators_TNmF is not None:
            indic_t = indicators_TNmF[t, :, :, :]
        else:
            indic_t = thetas[None, :, None] < forecaster_preds[:, None, :]
        weighted_indicators_NmF = w * indic_t

        phat_dict_list, Vn_values = multi_q_minmax_solver_v2(
            theta_weights_F=w, thetas=thetas, weighted_indicators_NmF=weighted_indicators_NmF,
        )
        minimax_value_history[t] = minimax_value_neg(alpha_list=alpha_list, Vn_values=Vn_values)

        phat = np.array([pd["phat"] for pd in phat_dict_list])
        k_star = np.array([pd["k_star"] for pd in phat_dict_list])
        k_star_prob = np.array([pd["k_star_prob"] for pd in phat_dict_list])

        if np.min(phat[1:] - phat[:-1]) < -1e-15:
            print(f"Warning: phat not monotone at t={t}: {np.min(phat[1:] - phat[:-1])}")

        phat_history[t, :] = phat
        k_star_history[t, :] = k_star
        k_star_prob_history[t, :] = k_star_prob

        phat_score = (
            k_star_prob[:, None] * elementary_scores_grid_N(k_star, y_t, thetas, alpha_list)
            + (1 - k_star_prob[:, None]) * elementary_scores_grid_N(k_star + 1, y_t, thetas, alpha_list)
        )
        pb_loss_t = (
            k_star_prob * pinball_loss(k_star, y_t, alpha_list[None, :]) / m
            + (1 - k_star_prob) * pinball_loss(k_star + 1, y_t, alpha_list[None, :]) / m
        )
        omni_error_history[t, :, :] = phat_score
        pb_loss_history[t, :] = pb_loss_t

        f_scores = elementary_scores_grid_N_F(forecaster_preds, y_t, thetas, alpha_list)

        # Hedge update in log space.
        log_w = np.log(w + 1e-10)
        log_w += eta * (phat_score[:, :, None] - f_scores)
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        w /= np.sum(w)
    runtime_sec = time.perf_counter() - t0

    omni_score_trace = omni_error_from_scores(omni_error_history)
    assert omni_score_trace.shape == (T,)

    return {
        "phat_history": phat_history,                        # (T, N) scaled units
        "phat_history_orig": phat_history * unit,            # (T, N) original counts
        "minimax_value_history": minimax_value_history,      # (T,)
        "omni_error_history": np.max(omni_error_history, axis=(1, 2)),  # (T,)
        "omni_score_trace": omni_score_trace,                # (T,)
        "pb_loss_history": pb_loss_history,                  # (T, N)
        "thetas": thetas,
        "y_arr": y_arr,
        "Y_scaled": Y_scaled,
        "T": T, "N": N, "F": F, "m": m, "unit": unit,
        "eta": eta, "eta_multiplier": eta_multiplier,
        "alpha_list": alpha_list,
        "runtime_sec": runtime_sec,
    }


# ============================================================================
# Grid-free weighted-quantile-loss omniprediction (WQL)
# ============================================================================

def run_omni_wql(preds_TNF: np.ndarray, y_arr: np.ndarray, unit: int,
                 alpha_list, eta_multiplier: float = 1.0, verbose: bool = False) -> dict:
    """Run WQL omniprediction (formerly ``omniprediction_multiq_wql``).

    Grid-free: predictions/targets are divided by ``unit`` (for numerical
    conditioning) but *not* rounded to the grid; ``m`` only normalizes losses.
    """
    alpha_list = np.asarray(alpha_list, dtype=np.float64)
    preds_TNF = np.asarray(preds_TNF, dtype=np.float64)
    y_arr = np.asarray(y_arr, dtype=np.float64)
    T, N, F = preds_TNF.shape
    assert alpha_list.shape[0] == N and y_arr.shape[0] == T

    _, m = _grid_from_preds(preds_TNF, unit)
    eta = eta_multiplier * np.sqrt(np.log(N * F) / T)

    Y_scaled = y_arr / unit
    preds_scaled = preds_TNF / unit  # (T, N, F), no rounding

    forecasters_pb_loss_history = np.zeros((T, N, F))
    for t in range(T):
        forecasters_pb_loss_history[t, :, :] = pinball_loss(
            p=preds_scaled[t, :, :], y=Y_scaled[t], alpha=alpha_list[:, None]) / m
    forecasters_score_trace = wql_omni_error_from_pb_loss(forecasters_pb_loss_history)
    best_forecaster_score_trace = forecasters_score_trace.min(axis=1)

    w = np.ones((N, F)) / (N * F)
    phat_history = np.zeros((T, N))
    minimax_value_history = np.zeros((T,))
    omni_pb_loss_history = np.zeros((T, N))

    t0 = time.perf_counter()
    for t in range(T):
        y_t = Y_scaled[t]
        forecaster_preds = preds_scaled[t, :, :]
        if np.min(w) < 1e-15:
            print(f"Minimum weight too small: {np.min(w)}")

        phat, Vn_diff_new = efficeint_solve_weighted_hinge_split(
            weights_NF=w, forecasts_NF=forecaster_preds)
        phat_history[t, :] = phat
        minimax_value_history[t] = np.sum(alpha_list * Vn_diff_new)

        p_scores = pinball_loss(p=phat, y=y_t, alpha=alpha_list) / m
        f_scores = pinball_loss(p=forecaster_preds, y=y_t, alpha=alpha_list[:, None]) / m
        omni_pb_loss_history[t, :] = p_scores

        log_w = np.log(w + 1e-12)
        log_w += eta * (p_scores[:, None] - f_scores)
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        w = np.maximum(w, 1e-12)
        w /= np.sum(w)
    runtime_sec = time.perf_counter() - t0

    omni_score_trace = wql_omni_error_from_pb_loss(omni_pb_loss_history)

    return {
        "phat_history": phat_history,
        "phat_history_orig": phat_history * unit,
        "minimax_value_history": minimax_value_history,
        "omni_pb_loss_history": omni_pb_loss_history,
        "omni_score_trace": omni_score_trace,
        "omni_score_trace_rel": omni_score_trace - best_forecaster_score_trace,
        "forecasters_score_trace": forecasters_score_trace,
        "best_forecaster_score_trace": best_forecaster_score_trace,
        "y_arr": y_arr, "Y_scaled": Y_scaled,
        "T": T, "N": N, "F": F, "m": m, "unit": unit,
        "eta": eta, "eta_multiplier": eta_multiplier,
        "alpha_list": alpha_list,
        "runtime_sec": runtime_sec,
    }


# ============================================================================
# QL-opt and Pinball-opt (fixed-weight Hedge baselines)
# ============================================================================

def run_ql_pb_opt(preds_TNF: np.ndarray, y_arr: np.ndarray, unit: int,
                  alpha_list, eta_multiplier: float = 1.0,
                  round_Y_F: bool = True, compute_omni_traces: bool = True,
                  verbose: bool = False) -> dict:
    """Run QL-opt and Pinball-opt (formerly ``ql_pb_opt``).

    ``compute_omni_traces`` (default True) controls whether the grid-based
    elementary-score omni traces are computed. Those require a ``(T, N, m, F)``
    tensor for the base forecasters, which is ~15 GB for US at ``unit=25`` — set
    it to False when you only need the prediction sequences (e.g. to score them
    with :func:`metrics.pseudo_omni_objective`).
    """
    alpha_list = np.asarray(alpha_list, dtype=np.float64)
    preds_TNF = np.asarray(preds_TNF, dtype=np.float64)
    y_arr = np.asarray(y_arr, dtype=np.float64)
    T, N, F = preds_TNF.shape
    assert alpha_list.shape[0] == N and y_arr.shape[0] == T

    thetas, m = _grid_from_preds(preds_TNF, unit)
    eta = eta_multiplier * np.sqrt(np.log(F) / T)

    if round_Y_F:
        Y_scaled = np.round(y_arr / unit)
        preds_scaled = np.round(preds_TNF / unit)
    else:
        Y_scaled = y_arr / unit
        preds_scaled = preds_TNF / unit

    # Base-forecaster pinball losses drive the Hedge updates (small: (T, N, F)).
    forecasters_pb_loss_history = np.zeros((T, N, F))
    for t in range(T):
        forecasters_pb_loss_history[t] = pinball_loss(preds_scaled[t], Y_scaled[t], alpha_list[:, None]) / m

    # Grid-based omni error trace of the base forecasters (optional; needs a
    # (T, N, m, F) tensor, which is huge for fine grids).
    forecasters_score_trace = None
    best_forecaster_score_trace = None
    if compute_omni_traces:
        forecasters_escore_history = np.zeros((T, N, m, F))
        for t in range(T):
            forecasters_escore_history[t] = elementary_scores_grid_N_F(preds_scaled[t], Y_scaled[t], thetas, alpha_list)
        forecasters_score_trace = omni_error_from_scores(forecasters_escore_history)
        best_forecaster_score_trace = forecasters_score_trace.min(axis=1)

    pinball_v = np.ones((N, F)) / F
    ql_v = np.ones(F) / F
    ql_v_history = np.zeros((T, F))
    pinball_preds_history = np.zeros((T, N))
    ql_preds_history = np.zeros((T, N))
    pinball_pb_loss_history = np.zeros((T, N))
    ql_pb_loss_history = np.zeros((T, N))

    t0 = time.perf_counter()
    for t in range(T):
        y_t = Y_scaled[t]
        forecaster_preds = preds_scaled[t, :, :]

        # Pinball-opt (independent Hedge per level).
        pinball_preds_t = (forecaster_preds * pinball_v).sum(axis=1)
        pinball_preds_history[t, :] = pinball_preds_t
        pinball_pb_loss_history[t, :] = pinball_loss(pinball_preds_t, y_t, alpha_list) / m
        f_pb_losses = forecasters_pb_loss_history[t, :, :]

        pinball_v = np.log(pinball_v + 1e-10)
        pinball_v -= eta * f_pb_losses
        pinball_v -= np.max(pinball_v, axis=1, keepdims=True)
        pinball_v = np.exp(pinball_v)
        pinball_v /= np.sum(pinball_v, axis=1, keepdims=True)

        # QL-opt (single Hedge shared across levels).
        ql_preds_t = (forecaster_preds * ql_v[None, :]).sum(axis=1)
        ql_preds_history[t, :] = ql_preds_t
        ql_pb_loss_history[t, :] = pinball_loss(ql_preds_t, y_t, alpha_list) / m
        ql_v_history[t, :] = ql_v

        f_ql_loss = np.mean(f_pb_losses, axis=0)
        ql_v = np.log(ql_v + 1e-10)
        ql_v -= eta * f_ql_loss
        ql_v -= np.max(ql_v)
        ql_v = np.exp(ql_v)
        ql_v /= np.sum(ql_v)
    runtime_sec = time.perf_counter() - t0

    out = {
        "pinball_preds_history": pinball_preds_history,          # (T, N) scaled units
        "ql_preds_history": ql_preds_history,                    # (T, N) scaled units
        "pinball_preds_history_orig": pinball_preds_history * unit,
        "ql_preds_history_orig": ql_preds_history * unit,
        "ql_v_history": ql_v_history,
        "pinball_pb_loss_history": pinball_pb_loss_history,
        "ql_pb_loss_history": ql_pb_loss_history,
        "eta": eta, "eta_multiplier": eta_multiplier, "round_Y_F": round_Y_F,
        "y_arr": y_arr, "Y_scaled": Y_scaled,
        "T": T, "N": N, "F": F, "m": m, "unit": unit,
        "alpha_list": alpha_list,
        "runtime_sec": runtime_sec,
    }

    if compute_omni_traces:
        pinball_escores = elementary_scores_grid_T_N(pinball_preds_history, Y_scaled, thetas, alpha_list)
        ql_escores = elementary_scores_grid_T_N(ql_preds_history, Y_scaled, thetas, alpha_list)
        pinball_omni_score_trace = np.max(np.cumsum(pinball_escores, axis=0), axis=(1, 2)) / np.arange(1, T + 1)
        ql_omni_score_trace = np.max(np.cumsum(ql_escores, axis=0), axis=(1, 2)) / np.arange(1, T + 1)
        out.update({
            "pinball_omni_history": np.max(pinball_escores, axis=(1, 2)),
            "ql_omni_history": np.max(ql_escores, axis=(1, 2)),
            "pinball_omni_score_trace": pinball_omni_score_trace,
            "ql_omni_score_trace": ql_omni_score_trace,
            "pinball_omni_score_trace_rel": pinball_omni_score_trace - best_forecaster_score_trace,
            "ql_omni_score_trace_rel": ql_omni_score_trace - best_forecaster_score_trace,
            "forecasters_score_trace": forecasters_score_trace,
            "best_forecaster_score_trace": best_forecaster_score_trace,
        })
    return out

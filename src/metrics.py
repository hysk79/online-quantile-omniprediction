"""Loss functions, elementary scores, and error traces.

Also provides :func:`pseudo_omni_objective` (max-regret weighted-quantile-loss
objective) and :func:`mean_ql_objective` (level-averaged QL), used to score
algorithm output in the sensitivity experiments.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Pinball / quantile loss
# ---------------------------------------------------------------------------

def pinball_loss(p, y, alpha):
    """Pinball (quantile) loss; broadcasts over arbitrary shapes."""
    p = np.asarray(p)
    y = np.asarray(y)
    alpha = np.asarray(alpha)
    return np.maximum(alpha * (y - p), (1 - alpha) * (p - y))


def ql_loss(p, y, alpha_list):
    """Mean pinball loss across quantile levels. ``p``: (n, N), ``y``: (n,)."""
    p = np.asarray(p)
    y = np.asarray(y)
    alpha_list = np.asarray(alpha_list)
    loss = pinball_loss(p, y[:, None], alpha_list[None, :])
    return np.mean(loss, axis=1)


# ---------------------------------------------------------------------------
# Elementary scores (indicator-based, used by the omni-v2 / QL-pb algorithms)
# ---------------------------------------------------------------------------

def elementary_scores_grid_N(p_vec_N, y, thetas, alpha_list):
    """(N,) predictions -> (N, m) elementary scores at a single time step."""
    N = alpha_list.shape[0]
    m = thetas.shape[0]
    alpha_vec = alpha_list[:, None]
    thetas_row = thetas[None, :]
    assert p_vec_N.shape == (N,), f"p_vec_N.shape: {p_vec_N.shape}, N: {N}"

    p_vec = p_vec_N[:, None]                              # (N, 1)
    theta_lt_y = (thetas_row < y).astype(float)          # (1, m)
    term1 = (y < p_vec).astype(float) - alpha_vec        # (N, 1)
    term2 = (thetas_row < p_vec).astype(float) - theta_lt_y  # (N, m)
    return term1 * term2                                 # (N, m)


def elementary_scores_grid_N_F(p_grid_N_F, y, thetas, alpha_list):
    """(N, F) predictions -> (N, m, F) elementary scores at a single time step."""
    N = alpha_list.shape[0]
    F = p_grid_N_F.shape[1]
    assert p_grid_N_F.shape == (N, F)

    p_grid = p_grid_N_F[:, None, :]        # (N, 1, F)
    theta_grid = thetas[None, :, None]     # (1, m, 1)
    alpha_grid = alpha_list[:, None, None]  # (N, 1, 1)
    term1 = (y < p_grid).astype(float) - alpha_grid                                    # (N, 1, F)
    term2 = (theta_grid < p_grid).astype(float) - (theta_grid < y).astype(float)       # (N, m, F)
    return term1 * term2                   # (N, m, F)


def elementary_scores_grid_T_N(p_grid_T_N, y, thetas, alpha_list):
    """(T, N) predictions -> (T, N, m) elementary scores over the whole horizon."""
    T = y.shape[0]
    N = alpha_list.shape[0]
    assert p_grid_T_N.shape == (T, N), f"p_grid_T_N.shape: {p_grid_T_N.shape}, T: {T}, N: {N}"

    p_grid = p_grid_T_N[:, :, None]        # (T, N, 1)
    y_arr = y[:, None, None]               # (T, 1, 1)
    theta_grid = thetas[None, None, :]     # (1, 1, m)
    alpha_grid = alpha_list[None, :, None]  # (1, N, 1)
    term1 = (y_arr < p_grid).astype(float) - alpha_grid
    term2 = (theta_grid < p_grid).astype(float) - (theta_grid < y_arr).astype(float)
    return term1 * term2                   # (T, N, m)


# ---------------------------------------------------------------------------
# Error traces
# ---------------------------------------------------------------------------

def omni_error_from_scores(scores: np.ndarray) -> np.ndarray:
    """Running omniprediction error: max over (level, theta) of the cumulative
    mean elementary score. ``scores`` is (T, N, m) or (T, N, m, F)."""
    if scores.ndim == 3:
        return np.max(scores.cumsum(axis=0), axis=(1, 2)) / (np.arange(scores.shape[0]) + 1)
    elif scores.ndim == 4:
        return np.max(scores.cumsum(axis=0), axis=(1, 2)) / (np.arange(scores.shape[0])[:, None] + 1)
    raise ValueError(f"scores.ndim: {scores.ndim}")


def omni_score_trace_for_preds(preds_TN_scaled: np.ndarray, y_scaled: np.ndarray,
                               thetas: np.ndarray, alpha_list: np.ndarray) -> np.ndarray:
    """Running grid omniprediction error for a single ``(T, N)`` prediction seq.

    Mirrors the ``pinball_omni_score_trace`` computed inside
    :func:`algorithms.run_ql_pb_opt`, factored out so the notebook can score
    reference forecasters / arbitrary prediction sequences on a fixed ``thetas``
    grid without rebuilding a big ``(T, N, m, F)`` tensor.

    ``preds_TN_scaled`` and ``y_scaled`` must already be divided by ``unit`` (and
    rounded, to match how the base forecasters were scored).
    """
    T = y_scaled.shape[0]
    escores = elementary_scores_grid_T_N(preds_TN_scaled, y_scaled, thetas, alpha_list)
    return np.max(np.cumsum(escores, axis=0), axis=(1, 2)) / np.arange(1, T + 1)


def count_quantile_crossings(q_TN: np.ndarray, alpha_list: np.ndarray, tol: float = 1e-9):
    """Count quantile crossings per time step for a ``(T, N)`` prediction seq.

    A crossing for a level pair ``(i, j)`` with ``alpha_i < alpha_j`` occurs when
    the predicted quantiles are out of order, ``q_i(t) > q_j(t)``. All
    ``N*(N-1)/2`` ordered pairs are checked (not just adjacent levels).

    Parameters
    ----------
    q_TN : (T, N) predictions.
    alpha_list : (N,) quantile levels (any order; sorted internally).
    tol : slack so exact ties are not counted as crossings.

    Returns
    -------
    (any_cross, n_pair_cross) : both ``(T,)`` int arrays.
      * ``any_cross[t]``     = 1 if *any* crossing occurs at step ``t``, else 0.
      * ``n_pair_cross[t]``  = number of level pairs that cross at step ``t``.
    """
    q_TN = np.asarray(q_TN, dtype=np.float64)
    alpha_list = np.asarray(alpha_list, dtype=np.float64)
    order = np.argsort(alpha_list)
    q = q_TN[:, order]                                   # levels ascending in alpha
    T, N = q.shape
    iu = np.triu_indices(N, k=1)                         # all (i, j) with i < j
    # crossing if lower-level prediction exceeds higher-level prediction.
    pair_cross = (q[:, iu[0]] - q[:, iu[1]]) > tol       # (T, n_pairs)
    n_pair_cross = pair_cross.sum(axis=1).astype(int)    # (T,)
    any_cross = (n_pair_cross > 0).astype(int)           # (T,)
    return any_cross, n_pair_cross


def instantaneous_ql(preds, y, alpha_list) -> np.ndarray:
    """Per-step mean pinball loss across levels (original scale).

    ``preds`` is ``(T, N)`` -> returns ``(T,)``; ``(T, N, F)`` -> ``(T, F)``.
    """
    preds = np.asarray(preds, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    alpha_list = np.asarray(alpha_list, dtype=np.float64)
    if preds.ndim == 2:
        loss = pinball_loss(preds, y[:, None], alpha_list[None, :])       # (T, N)
        return loss.mean(axis=1)
    elif preds.ndim == 3:
        loss = pinball_loss(preds, y[:, None, None], alpha_list[None, :, None])  # (T, N, F)
        return loss.mean(axis=1)
    raise ValueError(f"preds.ndim: {preds.ndim}")


def ql_error_from_pb_loss(pb_loss: np.ndarray) -> np.ndarray:
    """Running QL error: average over levels of the cumulative-mean pinball loss."""
    if pb_loss.ndim == 2:
        return np.average(pb_loss.cumsum(axis=0), axis=1) / (np.arange(pb_loss.shape[0]) + 1)
    elif pb_loss.ndim == 3:
        return np.average(pb_loss.cumsum(axis=0), axis=1) / (np.arange(pb_loss.shape[0])[:, None] + 1)
    raise ValueError(f"pb_loss.ndim: {pb_loss.ndim}")


def wql_omni_error_from_pb_loss(pb_loss: np.ndarray) -> np.ndarray:
    """Running weighted-QL omni error: max over levels of the cumulative-mean
    pinball loss. ``pb_loss`` is (T, N) or (T, N, F)."""
    if pb_loss.ndim == 2:
        return np.max(pb_loss.cumsum(axis=0), axis=1) / (np.arange(pb_loss.shape[0]) + 1)
    elif pb_loss.ndim == 3:
        return np.max(pb_loss.cumsum(axis=0), axis=1) / (np.arange(pb_loss.shape[0])[:, None] + 1)
    raise ValueError(f"pb_loss.ndim: {pb_loss.ndim}")


# ---------------------------------------------------------------------------
# Pseudo-omniprediction objective (max regret over weighted quantile losses)
# ---------------------------------------------------------------------------

def pseudo_omni_objective(q_TN: np.ndarray, y_T: np.ndarray,
                          preds_TNF: np.ndarray, alpha_list: np.ndarray) -> dict:
    r"""Max-regret weighted-quantile-loss omniprediction objective.

    Evaluates a prediction sequence ``q(t) = (q_n(t))_n`` (e.g. the omni-v2
    output) against base forecasters under the class of weighted quantile
    losses:

    .. math::
        \sup_{\ell \in C_{wql},\, b \in [B]}
            \frac{1}{T} \sum_{t=1}^T [\ell(q(t), Y_t) - \ell(f_b(t), Y_t)].

    Because a weighted quantile loss is linear in its per-level weights, the
    supremum over the loss class is attained at a single quantile level, so the
    objective reduces to

    .. math::
        \max_{n \in [N]}  (1/T) \sum_t \rho_{\tau_n}(q_n(t), Y_t)  
        - \min_{b} (1/T) \sum_t \rho_{\tau_n}(f_b^n(t), Y_t).

    All inputs must be on the **same (original count) scale** so the objective
    is comparable across grid sizes. Pinball losses are *not* divided by ``m``.

    Parameters
    ----------
    q_TN : (T, N) our predictions, one per (time, quantile level).
    y_T : (T,) realized targets.
    preds_TNF : (T, N, F) base-forecaster predictions.
    alpha_list : (N,) quantile levels.

    Returns
    -------
    dict with the scalar ``objective`` at the final horizon, plus the running
    trace, the per-level regret, and the argmax level.
    """
    q_TN = np.asarray(q_TN, dtype=np.float64)
    y_T = np.asarray(y_T, dtype=np.float64)
    preds_TNF = np.asarray(preds_TNF, dtype=np.float64)
    alpha_list = np.asarray(alpha_list, dtype=np.float64)

    T, N = q_TN.shape
    assert y_T.shape == (T,)
    assert preds_TNF.shape[:2] == (T, N), f"preds_TNF.shape={preds_TNF.shape}, expected ({T},{N},F)"

    # Per-step pinball losses (original scale, un-normalized).
    our_pb = pinball_loss(q_TN, y_T[:, None], alpha_list[None, :])              # (T, N)
    f_pb = pinball_loss(preds_TNF, y_T[:, None, None], alpha_list[None, :, None])  # (T, N, F)

    steps = np.arange(1, T + 1)
    our_cummean = our_pb.cumsum(axis=0) / steps[:, None]                        # (T, N)
    f_cummean = f_pb.cumsum(axis=0) / steps[:, None, None]                      # (T, N, F)

    # Regret vs. the best base forecaster at each level, then max over levels.
    best_f_cummean = f_cummean.min(axis=2)                                     # (T, N)
    regret_TN = our_cummean - best_f_cummean                                   # (T, N)
    objective_trace = regret_TN.max(axis=1)                                    # (T,)

    regret_final = regret_TN[-1]                                               # (N,)
    argmax_level = int(np.argmax(regret_final))

    # Absolute (non-relative) version: max over levels of our mean pinball loss,
    # and the same statistic for each base forecaster.
    our_abs_trace = our_cummean.max(axis=1)                                    # (T,)
    f_abs_trace = f_cummean.max(axis=1)                                        # (T, F)

    return {
        # --- relative (regret vs. per-level best forecaster) ---
        "objective": float(objective_trace[-1]),
        "objective_trace": objective_trace,               # (T,)
        "regret_per_level": regret_final,                 # (N,)
        "regret_per_level_trace": regret_TN,              # (T, N)
        "argmax_level": argmax_level,
        "argmax_alpha": float(alpha_list[argmax_level]),
        # --- absolute ---
        "objective_abs": float(our_abs_trace[-1]),
        "objective_abs_trace": our_abs_trace,             # (T,)
        "forecasters_objective_abs": f_abs_trace[-1],     # (F,)
        "forecasters_objective_abs_trace": f_abs_trace,   # (T, F)
        "best_forecaster_objective_abs": float(f_abs_trace[-1].min()),
        # --- per-level components ---
        "our_mean_pinball": our_cummean[-1],              # (N,)
        "our_mean_pinball_trace": our_cummean,            # (T, N)
        "best_forecaster_mean_pinball": best_f_cummean[-1],  # (N,)
        "best_forecaster_mean_pinball_trace": best_f_cummean,  # (T, N)
        "forecasters_mean_pinball": f_cummean[-1],        # (N, F)
    }


def mean_ql_objective(q_TN: np.ndarray, y_T: np.ndarray,
                      preds_TNF: np.ndarray, alpha_list: np.ndarray) -> dict:
    r"""Level-averaged quantile-loss regret (the "Total QL (Rel.)" metric).

    Same inputs / scale conventions as :func:`pseudo_omni_objective`, but the
    max over quantile levels is replaced by the *average* over levels, i.e. the
    usual (mean) quantile loss, reported relative to the best base forecaster:

    .. math::
        \frac{1}{T}\sum_t \mathrm{QL}(q(t), Y_t) - \min_b \frac{1}{T}\sum_t \mathrm{QL}(f_b(t), Y_t),
        \qquad \mathrm{QL}(q, y) = \frac{1}{N}\sum_n \rho_{\alpha_n}(q_n, y).

    ``objective`` uses the best *single* forecaster on mean QL (what the paper
    plots). ``objective_per_level_best`` instead subtracts the per-level best
    forecaster (mean over levels of the per-level regret used in
    :func:`pseudo_omni_objective`); it is a stricter comparator.
    """
    q_TN = np.asarray(q_TN, dtype=np.float64)
    y_T = np.asarray(y_T, dtype=np.float64)
    preds_TNF = np.asarray(preds_TNF, dtype=np.float64)
    alpha_list = np.asarray(alpha_list, dtype=np.float64)

    T, N = q_TN.shape
    assert y_T.shape == (T,)
    assert preds_TNF.shape[:2] == (T, N), f"preds_TNF.shape={preds_TNF.shape}, expected ({T},{N},F)"

    our_pb = pinball_loss(q_TN, y_T[:, None], alpha_list[None, :])              # (T, N)
    f_pb = pinball_loss(preds_TNF, y_T[:, None, None], alpha_list[None, :, None])  # (T, N, F)

    steps = np.arange(1, T + 1)
    our_ql_trace = our_pb.mean(axis=1).cumsum() / steps                          # (T,)
    f_ql_cummean = f_pb.mean(axis=1).cumsum(axis=0) / steps[:, None]            # (T, F)
    best_single_trace = f_ql_cummean.min(axis=1)                                # (T,)
    per_level_best_trace = (f_pb.cumsum(axis=0) / steps[:, None, None]).min(axis=2).mean(axis=1)  # (T,)

    return {
        # --- relative ---
        "objective": float(our_ql_trace[-1] - best_single_trace[-1]),
        "objective_trace": our_ql_trace - best_single_trace,                    # (T,)
        "objective_per_level_best": float(our_ql_trace[-1] - per_level_best_trace[-1]),
        "objective_per_level_best_trace": our_ql_trace - per_level_best_trace,  # (T,)
        # --- absolute ---
        "objective_abs": float(our_ql_trace[-1]),
        "objective_abs_trace": our_ql_trace,                                    # (T,)
        "our_mean_ql": float(our_ql_trace[-1]),
        "our_mean_ql_trace": our_ql_trace,                                      # (T,)
        "forecasters_mean_ql": f_ql_cummean[-1],                                # (F,)
        "forecasters_mean_ql_trace": f_ql_cummean,                              # (T, F)
        "best_forecaster_mean_ql": float(best_single_trace[-1]),
        "best_forecaster_mean_ql_trace": best_single_trace,                     # (T,)
        "best_forecaster_idx": int(np.argmin(f_ql_cummean[-1])),
        "per_level_best_mean_ql": float(per_level_best_trace[-1]),
    }

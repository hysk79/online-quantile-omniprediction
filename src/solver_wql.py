"""Minimax solver for the grid-free weighted-quantile-loss (WQL) omni step.

Ported from ``multi_q_minimax_solver_wql.py`` (dead commented-out variant
dropped). Solves, per step, the weighted-hinge split that yields the WQL
minimax prediction for every quantile level at once.
"""
from __future__ import annotations

import numpy as np
from typing import List


def multi_q_minmax_solver_wql(weights_NF: np.ndarray, forecasts_NF: np.ndarray):
    """Solve the WQL minmax problem across all quantile levels (randomized-free)."""
    N, F = weights_NF.shape
    assert weights_NF.shape == forecasts_NF.shape
    assert np.isclose(np.sum(weights_NF), 1.0)

    Vn_dic = solve_weighted_hinge_split_all_n(
        weights_NF=weights_NF, forecasts_NF=forecasts_NF,
        tol=min(weights_NF.min() / 2, 1e-10),
    )
    Vn_values = Vn_dic['minimum']
    j_optimal = Vn_dic['theta_interval'][:, 0]
    assert Vn_values.shape == j_optimal.shape == (N + 1,)
    assert np.all(j_optimal[:-1] <= j_optimal[1:]), f"j_optimal must be non-decreasing, got {j_optimal}"

    weights_N = weights_NF.sum(axis=1)
    weighted_forecasts_N = (forecasts_NF * weights_NF).sum(axis=1)
    numerator_N = weighted_forecasts_N + Vn_values[:-1] - Vn_values[1:]
    phat = np.concatenate([[-np.inf], numerator_N / weights_N])

    if not np.all(phat[:-1] <= phat[1:] + 1e-10):
        print(f'phat is not ordered, max violation: {np.max(phat[:-1] - phat[1:])}')
        raise ValueError('phat is not ordered')

    return phat[1:], Vn_values


def efficeint_solve_weighted_hinge_split(
    weights_NF: np.ndarray,
    forecasts_NF: np.ndarray,
    tol: float = 1e-12,
    verbose: bool = False,
):
    """Efficient single-pass solver returning (phat_N, Vn_diff_N)."""
    weights = np.asarray(weights_NF, dtype=np.float64)
    forecasts = np.asarray(forecasts_NF, dtype=np.float64)

    if weights.shape != forecasts.shape:
        raise ValueError(f"Shape mismatch: weights {weights.shape}, forecasts {forecasts.shape}")
    if weights.ndim != 2:
        raise ValueError(f"Expected 2D arrays of shape (N, F), got ndim={weights.ndim}")
    if np.any(weights < -tol):
        raise ValueError("weights_NF must be nonnegative")

    N, F = weights.shape

    w_sum_N = np.concatenate([[0.0], np.sum(weights, axis=1)])
    w_cumsum_N_front = np.cumsum(w_sum_N)
    w_cumsum_N_back = w_sum_N.sum() - w_cumsum_N_front
    wf_N = np.concatenate([[0.0], np.sum(weights * forecasts, axis=1)])

    weights_total = w_sum_N.sum()
    if not np.isclose(weights_total, 1.0, rtol=0.0, atol=tol):
        print(f"weights_NF must sum to 1.0, got {weights_total}")

    w_1d = weights.reshape(-1)
    f_1d = forecasts.reshape(-1)
    order = np.argsort(f_1d, kind="mergesort")
    f_sorted = f_1d[order]
    w_sorted = w_1d[order]

    # Group identical forecast values and accumulate weight / weighted-value.
    f_uniq_vals: List[float] = []
    f_uniq_vals_w_cumsum: List[float] = []
    f_uniq_vals_wf_cumsum: List[float] = []
    M = f_sorted.size
    running_w_sum = 0.0
    running_wf_sum = 0.0
    i = 0
    while i < M:
        v = float(f_sorted[i])
        j = i
        while j < M and f_sorted[j] == v:
            j += 1
        here_w_sum = w_sorted[i:j].sum()
        running_w_sum += here_w_sum
        running_wf_sum += here_w_sum * v
        f_uniq_vals.append(float(v))
        f_uniq_vals_w_cumsum.append(running_w_sum)
        f_uniq_vals_wf_cumsum.append(running_wf_sum)
        i = j
    wf_total = wf_N[1:].sum()
    assert np.isclose(running_w_sum, 1.0, rtol=0.0, atol=1e-10), f'running_w_sum: {running_w_sum}'
    assert np.isclose(running_wf_sum, wf_total, rtol=0.0, atol=1e-10), f'running_wf_sum: {running_wf_sum}'

    K = len(f_uniq_vals)
    theta_stars = np.empty(N + 1, dtype=float)
    theta_stars[0] = f_uniq_vals[0]
    theta_stars[N] = f_uniq_vals[K - 1]
    theta_stars_idx = np.empty(N + 1, dtype=int)
    theta_stars_idx[0] = 0
    theta_stars_idx[N] = K - 1

    curr_k_idx = 0
    s_idx = np.repeat(np.arange(N, dtype=int), F)
    Vn = np.zeros(N + 1, dtype=float)
    for n_split in range(1, N):
        while f_uniq_vals_w_cumsum[curr_k_idx] < w_cumsum_N_front[n_split]:
            curr_k_idx += 1
            if curr_k_idx >= K:
                raise ValueError(f'curr_k_idx reached end of uniq_vals, curr_k_idx: {curr_k_idx}, K: {K}')
        theta_stars[n_split] = f_uniq_vals[curr_k_idx]
        theta_stars_idx[n_split] = curr_k_idx

        left_mask = s_idx < n_split
        left_term = np.maximum(f_1d[left_mask] - theta_stars[n_split], 0.0)
        right_term = np.maximum(theta_stars[n_split] - f_1d[~left_mask], 0.0)
        Vn[n_split] = float(np.dot(w_1d[left_mask], left_term) + np.dot(w_1d[~left_mask], right_term))

    Vn_diff_arr = Vn[:-1] - Vn[1:]
    numerator_N = wf_N[1:] + Vn_diff_arr
    phat = numerator_N / (w_sum_N[1:])

    if np.abs(np.sum(Vn_diff_arr)) > 1e-10:
        print(f'Warning: V0 - VN = {np.sum(Vn_diff_arr)} exceeds 1e-10')
    if np.min(phat[1:] - phat[:-1]) < -1e-12:
        print(f'Warning: phat is not ordered, min violation: {np.min(phat[1:] - phat[:-1])}')

    return phat, Vn[:-1] - Vn[1:]


def minimax_value_neg(alpha_list: np.ndarray, Vn_values: np.ndarray) -> float:
    N = alpha_list.shape[0]
    return np.sum(alpha_list * (Vn_values[0:N] - Vn_values[1:N + 1]))


def _solve_weighted_hinge_from_sorted(g_sorted, w_sorted, left_sorted, tol=1e-12, verbose=False):
    """Given globally sorted values/weights and a left/right split, return
    (theta_star, interval) minimizing the weighted hinge objective."""
    if g_sorted.size == 0:
        raise ValueError("Empty input to _solve_weighted_hinge_from_sorted")

    slope_minus_inf = -float(w_sorted[left_sorted].sum())

    uniq_vals: List[float] = []
    m_minus: List[float] = []
    m_plus: List[float] = []
    running_slope = slope_minus_inf

    M = g_sorted.size
    i = 0
    while i < M:
        v = g_sorted[i]
        j = i
        jump = 0.0
        while j < M and g_sorted[j] == v:
            jump += w_sorted[j]
            j += 1
        slope_after = running_slope + jump
        uniq_vals.append(float(v))
        m_minus.append(float(running_slope))
        m_plus.append(float(slope_after))
        running_slope = slope_after
        i = j

    theta_lo = np.inf
    theta_hi = -np.inf

    def _include_interval(lo: float, hi: float) -> None:
        nonlocal theta_lo, theta_hi
        theta_lo = min(theta_lo, lo)
        theta_hi = max(theta_hi, hi)

    if abs(slope_minus_inf) <= tol:
        _include_interval(-np.inf, uniq_vals[0])

    if verbose:
        print(f'tol: {tol}')
    K = len(uniq_vals)

    for k in range(K):
        if m_plus[k] >= 0:
            theta_lo = uniq_vals[k]
            theta_hi = uniq_vals[k]
            break

    if abs(m_plus[-1]) <= tol:
        _include_interval(uniq_vals[-1], np.inf)
        print(f'Flat segment to the far right: uniq_vals[-1]: {uniq_vals[-1]}, np.inf')

    if theta_lo > theta_hi:
        theta_lo = uniq_vals[0]
        theta_hi = uniq_vals[0]
        print(f'Fallback used: theta_lo: {theta_lo}, theta_hi: {theta_hi}')

    if np.isneginf(theta_lo) and np.isposinf(theta_hi):
        theta_star = 0.0
        print(f'Both theta_lo and theta_hi are -inf and inf, theta_star: {theta_star}')
    elif np.isneginf(theta_lo):
        theta_star = float(theta_hi)
        print(f'theta_lo is -inf, theta_star: {theta_star}')
    elif np.isposinf(theta_hi):
        theta_star = float(theta_lo)
        print(f'theta_hi is inf, theta_star: {theta_star}')
    else:
        theta_star = float(theta_lo)

    return float(theta_star), (float(theta_lo), float(theta_hi))


def solve_weighted_hinge_split_all_n(
    weights_NF: np.ndarray,
    forecasts_NF: np.ndarray,
    tol: float = 1e-12,
    n_list: List[int] = None,
    verbose: bool = False,
) -> dict:
    weights = np.asarray(weights_NF, dtype=np.float64)
    forecasts = np.asarray(forecasts_NF, dtype=np.float64)

    if weights.shape != forecasts.shape:
        raise ValueError(f"Shape mismatch: weights {weights.shape}, forecasts {forecasts.shape}")
    if weights.ndim != 2:
        raise ValueError(f"Expected 2D arrays of shape (N, F), got ndim={weights.ndim}")
    if np.any(weights < -tol):
        raise ValueError("weights_NF must be nonnegative")

    N, F = weights.shape
    total_w = float(weights.sum())
    if not np.isclose(total_w, 1.0, rtol=0.0, atol=tol):
        print(f"weights_NF must sum to 1.0, got {total_w}")

    g = forecasts.reshape(-1)
    w = weights.reshape(-1)
    if g.size == 0:
        raise ValueError("weights_NF and preds_NF must be non-empty")

    order = np.argsort(g, kind="mergesort")
    g_sorted = g[order]
    w_sorted = w[order]

    s_idx = np.repeat(np.arange(N, dtype=int), F)
    s_sorted = s_idx[order]

    theta_star_all = np.empty(N + 1, dtype=float)
    theta_interval_all = np.empty((N + 1, 2), dtype=float)
    minimum_all = np.empty(N + 1, dtype=float)

    theta_star_all[0] = -np.inf
    theta_star_all[N] = np.inf
    theta_interval_all[0, :] = -np.inf
    theta_interval_all[N, :] = np.inf
    minimum_all[0] = 0
    minimum_all[N] = 0

    for n_split in range(1, N):
        if verbose:
            if n_list is not None and n_split not in n_list:
                continue
            print(f'n_split: {n_split}')

        left_sorted = s_sorted < n_split
        theta_star, theta_interval = _solve_weighted_hinge_from_sorted(
            g_sorted=g_sorted, w_sorted=w_sorted, left_sorted=left_sorted, tol=tol, verbose=verbose,
        )

        left_mask = s_idx < n_split
        left_term = np.maximum(g[left_mask] - theta_star, 0.0)
        right_term = np.maximum(theta_star - g[~left_mask], 0.0)
        minimum = float(np.dot(w[left_mask], left_term) + np.dot(w[~left_mask], right_term))

        theta_star_all[n_split] = theta_star
        theta_interval_all[n_split, 0] = theta_interval[0]
        theta_interval_all[n_split, 1] = theta_interval[1]
        minimum_all[n_split] = minimum

    return {
        "theta_star": theta_star_all,
        "theta_interval": theta_interval_all,
        "minimum": minimum_all,
    }

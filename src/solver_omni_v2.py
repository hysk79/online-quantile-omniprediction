"""Minimax solver for the grid-based multi-quantile omniprediction step (v2).

Ported unchanged (aside from module layout) from ``multi_q_minimax_solver_v2.py``.
Given the current Hedge weights over (quantile level, theta, forecaster), it
solves the per-step minimax problem and returns the randomized prediction for
each quantile level. Numerically tuned; do not alter the tolerance logic.
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple


class VnComputer_v2:
    def __init__(self, weights_F: np.ndarray, thetas: np.ndarray,
                 weighted_indicators_NmF: np.ndarray, tol: float = 1e-10):
        self.weights_F = np.asarray(weights_F, dtype=np.float64)
        self.weights = self.weights_F.sum(axis=2)  # (N, m)
        self.thetas = np.asarray(thetas, dtype=np.float64)
        self.weighted_indicators_NmF = weighted_indicators_NmF
        self.tol = tol

        self.N = weights_F.shape[0]
        self.m = weights_F.shape[1]
        self.F = weights_F.shape[2]

        self._precompute_cumulative_sums()

    def _precompute_cumulative_sums(self):
        # Cumulative sums along the quantile-level axis (index 0 == level 1).
        weighted_indicators = (self.weighted_indicators_NmF).sum(axis=2)  # (N, m)
        self.past_cumsum = np.zeros((self.N + 1, self.m), dtype=np.float64)
        for n in range(self.N + 1):
            if n == 0:
                self.past_cumsum[0, :] = 0
            else:
                self.past_cumsum[n, :] = self.past_cumsum[n - 1, :] + weighted_indicators[n - 1, :]

        weighted_indicators_complement = self.weights - weighted_indicators
        self.future_cumsum = np.zeros((self.N + 1, self.m), dtype=np.float64)
        for n in range(self.N, 0, -1):
            if n == self.N:
                self.future_cumsum[self.N, :] = 0
            else:
                self.future_cumsum[n, :] = self.future_cumsum[n + 1, :] + weighted_indicators_complement[n, :]

    def compute_Vn(self, n: int, j_opt_min: int = 0) -> Tuple[float, int]:
        if n == 0:
            return 0.0, 0
        if n == self.N:
            return 0.0, self.m
        assert 0 < n < self.N

        j_min = j_opt_min
        past_row = self.past_cumsum[n]
        future_row = self.future_cumsum[n]

        c0 = past_row[j_min:].sum()
        if j_min > 0:
            c0 += future_row[:j_min].sum()

        if j_min < self.m:
            inc = -past_row[j_min:self.m] + future_row[j_min:self.m]
            tail = c0 + np.cumsum(inc)
            costs_segment = np.empty(1 + tail.shape[0], dtype=np.float64)
            costs_segment[0] = c0
            costs_segment[1:] = tail
        else:
            costs_segment = np.array([c0], dtype=np.float64)

        j_rel = int(np.argmin(costs_segment))
        j_opt = j_min + j_rel
        min_cost = float(costs_segment[j_rel])
        return min_cost, j_opt

    def compute_all_Vn(self) -> Tuple[np.ndarray, np.ndarray]:
        Vn_values = np.zeros(self.N + 1, dtype=np.float64)
        j_optimal = np.zeros(self.N + 1, dtype=np.int32)
        Vn_values[0] = 0.0
        j_optimal[0] = 0
        for n in range(1, self.N + 1):
            Vn, j_opt = self.compute_Vn(n=n, j_opt_min=j_optimal[n - 1])
            Vn_values[n] = Vn
            j_optimal[n] = j_opt
        return Vn_values, j_optimal


def j_opt_converter(j_opt: int, thetas: np.ndarray) -> float:
    thetas_gap = thetas[1] - thetas[0]
    return thetas[0] + (j_opt - 0.5) * thetas_gap


def single_q_minmax_solver2_v2(
    theta_weights_F: np.ndarray,        # (m, F)
    weighted_indicators_m: np.ndarray,  # (m,)
    thetas: np.ndarray,                 # (m,)
    eq_value: float = 0.0,
    j_opt_pre: int = 0,                 # minimum value of k_star, j_{n-1}^*
    j_opt_n: int = np.inf,              # maximum value of k_star, j_n^*
    tol: float = 1e-10,
) -> dict:
    """Solve the minmax problem for a single quantile level (randomized phat)."""
    m = len(thetas)

    j_opt_pre = max(j_opt_pre, 0)
    j_opt_n = min(j_opt_n, m)
    assert j_opt_pre <= j_opt_n
    assert theta_weights_F.shape[0] == weighted_indicators_m.shape[0]

    theta_weights = theta_weights_F.sum(axis=1)
    weighted_weights = np.sum(weighted_indicators_m)

    Bk_pre = np.concatenate([[-weighted_weights], np.cumsum(theta_weights) - weighted_weights])

    if np.isclose(Bk_pre[j_opt_pre], eq_value, rtol=0, atol=tol):
        return {
            "phat": j_opt_converter(j_opt_pre, thetas),
            "k_star": j_opt_converter(j_opt_pre, thetas),
            "k_star_prob": 1.0,
        }

    for j in range(j_opt_pre + 1, j_opt_n + 1):
        if Bk_pre[j] >= eq_value:
            k_star = j - 1
            if np.isclose(eq_value, Bk_pre[j - 1], rtol=0, atol=tol):
                k_star_prob = 0.0
            elif np.isclose(eq_value, Bk_pre[j], rtol=0, atol=tol):
                k_star_prob = 1.0
            else:
                num = eq_value - Bk_pre[j - 1]
                denom = Bk_pre[j] - Bk_pre[j - 1]
                scale = max(abs(num), abs(denom), 1.0)
                k_star_prob = (num / scale) / (denom / scale)
            if k_star_prob < 0.0 or k_star_prob > 1.0:
                print(k_star_prob, eq_value, Bk_pre[j - 1], Bk_pre[j])
                print('curr_j:', j, 'j_opt_pre:', j_opt_pre, 'j_opt_n:', j_opt_n)
            phat = np.random.choice(
                [j_opt_converter(k_star, thetas), j_opt_converter(k_star + 1, thetas)],
                p=[k_star_prob, 1.0 - k_star_prob],
            )
            return {
                "phat": phat,
                "k_star": j_opt_converter(k_star, thetas),
                "k_star_prob": k_star_prob,
            }
    if np.isclose(Bk_pre[j_opt_n], eq_value, rtol=0, atol=tol):
        return {
            "phat": j_opt_converter(j_opt_n, thetas),
            "k_star": j_opt_converter(j_opt_n, thetas),
            "k_star_prob": 1.0,
        }
    assert False, (
        "Single-q search for multi-q optimization (Numerical) ERROR. Try a smaller learning rate."
        + f' j_opt_pre: {j_opt_pre}, j_opt_n: {j_opt_n}, eq_value: {eq_value}, Bk_pre: {Bk_pre}'
    )


def multi_q_minmax_solver_v2(
    theta_weights_F: np.ndarray,        # (N, m, F)
    thetas: np.ndarray,                 # (m,)
    weighted_indicators_NmF: np.ndarray,  # (N, m, F)
):
    N = theta_weights_F.shape[0]
    m = len(thetas)
    assert theta_weights_F.shape[1] == m
    assert weighted_indicators_NmF.shape == theta_weights_F.shape
    assert np.isclose(np.sum(theta_weights_F), 1.0)

    Vn_computer = VnComputer_v2(
        weights_F=theta_weights_F,
        thetas=thetas,
        weighted_indicators_NmF=weighted_indicators_NmF,
        tol=np.min(weighted_indicators_NmF.sum(axis=2)) / 3,
    )
    Vn_values, j_optimal = Vn_computer.compute_all_Vn()

    phat_dict_list = []
    weighted_indicators_Nm = weighted_indicators_NmF.sum(axis=2)  # (N, m)
    for n in range(N):
        phat_dict = single_q_minmax_solver2_v2(
            theta_weights_F=theta_weights_F[n, :, :],
            weighted_indicators_m=weighted_indicators_Nm[n, :],
            thetas=thetas,
            eq_value=Vn_values[n] - Vn_values[n + 1],
            j_opt_pre=j_optimal[n],
            j_opt_n=j_optimal[n + 1],
            tol=np.min(theta_weights_F.sum(axis=2)) / 3,
        )
        phat_dict_list.append(phat_dict)

    return phat_dict_list, Vn_values


def minimax_value_neg(alpha_list: np.ndarray, Vn_values: np.ndarray) -> float:
    r"""Negated minimax value: :math:`\sum_n \tau_n (V_{n-1} - V_n)`."""
    N = alpha_list.shape[0]
    return np.sum(alpha_list * (Vn_values[0:N] - Vn_values[1:N + 1]))

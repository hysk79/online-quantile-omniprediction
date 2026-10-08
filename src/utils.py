"""Small shared helpers (experiment naming, plotting colors)."""
from __future__ import annotations

import numpy as np


def decimal_to_str(eta) -> str:
    """Filesystem-safe string for a float (``0.25`` -> ``'0,25'``)."""
    return str(np.round(eta, 4)).replace(".", ",")


def exp_name_omni_v2(w, geo, eta, suffix: str = "") -> str:
    return f"wk{w}_{geo}_eta{decimal_to_str(eta)}{suffix}"


def exp_name_ql_pb(w, geo, eta, round_Y_F: bool, suffix: str = "") -> str:
    return f"wk{w}_{geo}_eta{decimal_to_str(eta)}{'_orig' if not round_Y_F else ''}{suffix}"


def color_func(total_n, idx):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / (1.25 * (total_n - 1))) for i in range(2 * (total_n - 1))]
    return colors[idx]

"""Plot helpers and derived metrics for ``notebooks/paper_figures.ipynb``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

RC_PARAMS = {
    "figure.figsize": (3, 3),
    "figure.titlesize": 16,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 13,
    "legend.title_fontsize": 14,
    "xtick.labelsize": 11,
}

ENS_COLOR = "#3d3d3d"
ENS_STYLES = ["-.", "dotted", (0, (3, 1, 1, 1))]
# (panel / row index, ensemble index in summary["ens_model_names"], display name)
ENS_PLOTTED = [(0, 0, "COVIDhub-Ens"), (1, 2, "JHUAPL-SLPEns")]


def lr_c_str_format(eta, o_or_h):
    """Legend label ``c_O=10^x`` / ``c_H=10^x`` for a learning-rate constant."""
    e = np.log10(eta)
    e = int(e) if float(e).is_integer() else round(e, 2)
    return rf"$c_{{{o_or_h.upper()}}}=10^{{{e}}}$"


def log10_formatter(ax):
    """Label a log-scaled x-axis by log10 of the value, on every row of a sharex grid."""
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{np.log10(v):g}"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.tick_params(axis="x", labelbottom=True)


def frac_steps_crossing(q_TN, tol=1e-9):
    """Fraction of time steps whose quantile predictions are not monotone in the level."""
    return float((np.diff(q_TN, axis=1) < -tol).any(axis=1).mean())


def derived_metrics(s, n_omni_eta=None):
    """Per-step / cumulative quantile losses from the main summary.

    QL values are the level-averaged pinball loss in counts divided by ``Y_max``
    (pinball / m in the summary, times ``m / Y_max`` in grid units). Returns a
    dict of arrays indexed ``[week, geo, (eta,) t]``. ``n_omni_eta`` truncates
    the learning-rate sweep of our algorithm (the paper drops the largest one).
    """
    n_omni_eta = n_omni_eta or len(s["eta_omni"])
    T = s["Y"].shape[-1]
    t1 = np.arange(T) + 1
    unit_m = s["m"] / s["Y"].max(axis=-1)                  # (W, G)
    raw_unit_m = s["raw_m"] / s["raw_Y"].max(axis=-1)

    d = {}
    d["omni_score"] = s["omni_score"][:, :, :n_omni_eta]
    d["ql_score"] = s["ql_score"]
    d["bf_best_score"] = s["bf_best_score"]
    d["ens_score"] = s["ens_score"]

    d["omni_t_ql"] = s["omni_pb"][:, :, :n_omni_eta] * unit_m[..., None, None]
    d["ql_t_ql"] = s["ql_pb"] * unit_m[..., None, None]
    d["omni_ql"] = d["omni_t_ql"].cumsum(axis=-1) / t1
    d["ql_ql"] = d["ql_t_ql"].cumsum(axis=-1) / t1

    bf_t_ql = s["bf_pb"] * unit_m[..., None, None]         # (W, G, T, F)
    d["best_f_ql"] = bf_t_ql.cumsum(axis=2).min(axis=-1) / t1
    d["best_f_t_ql"] = bf_t_ql.min(axis=-1)
    d["avg_f_t_ql"] = bf_t_ql.mean(axis=-1)
    ens_t_ql = s["ens_pb"] * unit_m[..., None, None]       # (W, G, T, E)
    d["ens_t_ql"] = np.moveaxis(ens_t_ql, -1, 2)           # (W, G, E, T)
    d["ens_ql"] = d["ens_t_ql"].cumsum(axis=-1) / t1
    d["bf_raw_t_ql"] = s["bf_raw_pb"] * raw_unit_m[..., None, None]
    return d


def quantile_plot(ax, dates, q_preds_orig, y_orig, alpha_list, color="tab:blue"):
    """Nested central-interval bands of quantile predictions plus the target."""
    alpha_list = np.asarray(alpha_list)
    n = len(alpha_list)
    for i in range((n + 1) // 2):
        if alpha_list[i] + alpha_list[n - 1 - i] != 1:
            raise ValueError("alpha_list is not symmetric")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(5, 5))
    for ia, a in enumerate(alpha_list):
        if a > 0.5:
            break
        ax.fill_between(dates, y1=q_preds_orig[:, ia], y2=q_preds_orig[:, n - 1 - ia],
                        color=color, alpha=0.2 + a * 6 / 5, linewidth=0)
    ax.tick_params(axis="x", rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y%m"))
    ax.plot(dates, y_orig, color="black", linewidth=2, label="True Y")
    return ax


def standardized_ranks_from_losses(loss_df: pd.DataFrame, lower_is_better: bool = True) -> pd.DataFrame:
    """Per-row ranks rescaled to [0, 1]: best model -> 1, worst -> 0."""
    if loss_df.shape[1] < 2:
        raise ValueError("Need at least 2 models.")
    ranks = loss_df.rank(axis=1, method="average", ascending=lower_is_better)
    k = loss_df.notna().sum(axis=1)
    assert np.all(k >= 2)
    return (k.values[:, None] - ranks) / (k.values[:, None] - 1)


def ridgeline_rank_violin(rank_df: pd.DataFrame, model_order=None, bandwidth=None,
                          width=0.42, x_grid_size=400, quartile_colors=None,
                          figsize=(9, 7), title=None, xlabel="standardized rank",
                          ylabel="model", face_alpha=1.0, line_color="black",
                          line_width=1.0):
    """Horizontal ridgeline violins of standardized ranks, filled by quartile."""
    if quartile_colors is None:
        quartile_colors = ["#440154", "#31688e", "#35b779", "#fde725"]
    model_order = list(rank_df.columns) if model_order is None else list(model_order)
    rank_df = rank_df[model_order]

    fig, ax = plt.subplots(figsize=figsize)
    x_grid = np.linspace(0, 1, x_grid_size)
    y_positions = np.arange(len(model_order))[::-1]

    for y0, model in zip(y_positions, model_order):
        x = rank_df[model].dropna().to_numpy()
        if len(x) < 2:
            continue
        if np.std(x) < 1e-12:
            x = x + 1e-6 * np.random.randn(len(x))
        density = gaussian_kde(x, bw_method=bandwidth)(x_grid)
        density = density / density.max() * width
        q1, q2, q3 = np.quantile(x, [0.25, 0.50, 0.75])
        bounds = [0.0, q1, q2, q3, 1.0]
        for j in range(4):
            mask = (x_grid >= bounds[j]) & (x_grid <= bounds[j + 1])
            if mask.sum() < 2:
                continue
            ax.fill_between(x_grid[mask], y0 - density[mask], y0 + density[mask],
                            facecolor=quartile_colors[j], edgecolor="none",
                            alpha=face_alpha, zorder=2)
        ax.plot(x_grid, y0 + density, color=line_color, lw=line_width, zorder=3)
        ax.plot(x_grid, y0 - density, color=line_color, lw=line_width, zorder=3)
        ax.hlines(y0, 0, 1, color="black", lw=0.4, alpha=0.25, zorder=1)

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.5, len(model_order) - 0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_order)
    for label, model in zip(ax.get_yticklabels(), model_order):
        if "Our" in model:
            label.set_color("blue")
            label.set_fontweight("bold")
        if "Hedge-QL" in model:
            label.set_color("green")
            label.set_fontweight("bold")
        if "COVIDhub_Ens" in model or "JHUAPL-SLP" in model:
            label.set_fontweight("bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    handles = [Patch(facecolor=quartile_colors[i], edgecolor="black", label=str(i + 1)) for i in range(4)]
    ax.legend(handles=handles, title="Quartiles", loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=12)
    plt.tight_layout()
    return fig, ax

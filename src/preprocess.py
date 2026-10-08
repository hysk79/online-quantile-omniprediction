"""Data preprocessing pipeline.

Turns the raw hospitalization CSVs into the per-geo trimmed pickles that the
algorithms consume. The grid ``unit`` per geo and the outlier-clip table come
from ``config``.

Stages
------
1. :func:`build_raw_state_pickles` - slice the raw CSVs per (week, geo).
2. :func:`build_forecasts_dict`    - assemble daily + weekly forecast dicts.
3. :func:`build_trimmed`           - clip outliers, attach grid unit, split off
                                     ensemble models  -> ``data/trimmed/{geo}.pkl.gz``.
4. :func:`compute_base_forecaster_metrics` - precompute base-forecaster score
                                     traces -> ``results/base_forecasters/``.

The trimmed pickles (stages 1-3) are shipped in ``data/trimmed``; stage 4 and
everything downstream can be rerun from them.
"""
from __future__ import annotations

import gzip
import os
import pickle
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as cfg

# Raw CSVs span a slightly wider date range than the analysis window.
_RAW_START = "2020-12-29"
_RAW_END = "2023-06-09"

_GEO_LOWER = [g.lower() for g in cfg.GEO_FULL_LIST]


# ---------------------------------------------------------------------------
# Stage 1: raw CSV -> per (week, geo) pickle
# ---------------------------------------------------------------------------

def build_raw_state_pickles(weeks: Sequence[int] = (1, 2, 3, 4),
                            data_dir: Optional[str] = None) -> None:
    data_dir = data_dir or cfg.DATA_DIR
    for w in weeks:
        df_raw = pd.read_csv(os.path.join(data_dir, f"hospitalizations_{w}wk.csv"), index_col=0)
        df_raw = df_raw[(df_raw.target_end_date >= _RAW_START)
                        & (df_raw.target_end_date <= _RAW_END)
                        & df_raw.forecaster.isin(cfg.FORECASTER_LIST)]
        print(f"{w}wk data shape: {df_raw.shape}")
        out_dir = os.path.join(data_dir, f"{w}wk_states")
        os.makedirs(out_dir, exist_ok=True)
        for geo in _GEO_LOWER:
            df = df_raw[df_raw.geo_value == geo].drop(columns=["geo_value"])
            pickle.dump(df, open(os.path.join(out_dir, f"{geo.upper()}.pkl"), "wb"))


def _weekly_agg(series: pd.Series, dates_list_7) -> pd.Series:
    series = series[dates_list_7]
    group_index = np.arange(len(series)) // 7
    assert len(series) % 7 == 0
    series_w = series.groupby(group_index).sum(min_count=7)
    series_w.index = series.index[::7]
    return series_w


def _take_last_among_dup(df: pd.DataFrame) -> pd.DataFrame:
    assert all(c in df.columns for c in ["target_end_date", "forecast_date"])
    return df.loc[df.groupby("target_end_date")["forecast_date"].idxmax()]


# ---------------------------------------------------------------------------
# Stage 2: per (week, geo) pickle -> daily + weekly forecast dicts
# ---------------------------------------------------------------------------

def build_forecasts_dict(impute_na: bool = False, data_dir: Optional[str] = None) -> None:
    data_dir = data_dir or cfg.DATA_DIR
    dates_daily = pd.date_range(start=_RAW_START, end=_RAW_END).strftime("%Y-%m-%d")
    dates_list_7 = dates_daily[:(dates_daily.size // 7) * 7]
    dates_list_w = dates_list_7[::7]
    alpha_list = list(cfg.ALPHA_LIST)

    os.makedirs(cfg.STATES_DIR, exist_ok=True)
    os.makedirs(cfg.STATES_WEEKLY_DIR, exist_ok=True)

    for geo in tqdm(_GEO_LOWER):
        forecasts_dict = {}
        forecasts_dict_w = {}
        Y = Y_w = None
        for w in range(1, 5):
            forecasts_dict[w] = {}
            forecasts_dict_w[w] = {}
            df_w = pickle.load(open(os.path.join(data_dir, f"{w}wk_states/{geo.upper()}.pkl"), "rb"))

            if w == 1:
                g = df_w.groupby("target_end_date")["actual"]
                Y = g.first().sort_index()
                Y_w = _weekly_agg(Y, dates_list_7)
                assert Y.index.equals(dates_daily), f"{geo.upper()} Y index != dates_daily"

            for f_name in cfg.FORECASTER_LIST:
                forecasts_dict[w][f_name] = {}
                forecasts_dict_w[w][f_name] = {}
                df_here = df_w[df_w.forecaster == f_name]
                ud, rc = np.unique(df_here.target_end_date, return_counts=True)
                if np.sum(rc > 1) >= 1:
                    df_here = _take_last_among_dup(df_here)
                for alpha in alpha_list:
                    forecasts_dict[w][f_name][alpha] = (
                        df_here[[f"forecast_{alpha}", "target_end_date"]]
                        .set_index("target_end_date")[f"forecast_{alpha}"]
                        .reindex(dates_daily)
                    )

            if impute_na:
                for alpha in alpha_list:
                    avg_of_others = pd.concat(
                        [forecasts_dict[w][f][alpha] for f in cfg.FORECASTER_LIST
                         if f not in cfg.ENS_MODEL_NAMES], axis=1,
                    ).median(axis=1, skipna=True)
                    for f_name in cfg.FORECASTER_LIST:
                        forecasts_dict[w][f_name][alpha] = forecasts_dict[w][f_name][alpha].fillna(avg_of_others)

            for f_name in cfg.FORECASTER_LIST:
                for alpha in alpha_list:
                    forecasts_dict_w[w][f_name][alpha] = _weekly_agg(forecasts_dict[w][f_name][alpha], dates_list_7)

        suffix = "" if impute_na else "_raw"
        pickle.dump({
            "forecasts_dict": forecasts_dict, "alpha_list": alpha_list,
            "dates_list": dates_daily, "forecaster_list": cfg.FORECASTER_LIST, "Y": Y,
        }, open(os.path.join(cfg.STATES_DIR, f"{geo.upper()}{suffix}.pkl"), "wb"))
        pickle.dump({
            "forecasts_dict": forecasts_dict_w, "alpha_list": alpha_list,
            "dates_list": dates_list_w, "forecaster_list": cfg.FORECASTER_LIST, "Y": Y_w,
        }, open(os.path.join(cfg.STATES_WEEKLY_DIR, f"{geo.upper()}{suffix}.pkl"), "wb"))


# ---------------------------------------------------------------------------
# Stage 3: outlier clipping + grid unit + ensemble split -> trimmed pickle
# ---------------------------------------------------------------------------

def _smooth_outliers(forecasts_wo_ens_dict: dict, f_list_wo_ens: List[str],
                     alpha_list: list, alpha_cut: float, cuts: List[float]) -> None:
    """Clip the upper tail of every non-ensemble forecaster (in place).

    For the quantile level ``alpha_cut`` and every higher level, clip to the corresponding entry of ``cuts`` across
    all four horizons; the first entry also caps every level up to ``alpha_cut``.
    """
    alpha_cut_ix = int(np.where(np.array(alpha_list) == alpha_cut)[0][0])
    assert len(cuts) + alpha_cut_ix == len(alpha_list), (
        f"cuts of length {len(cuts)} don't line up with alpha_cut={alpha_cut}")
    for f_name in f_list_wo_ens:
        for ia in range(alpha_cut_ix, len(alpha_list)):
            ix = ia - alpha_cut_ix
            if forecasts_wo_ens_dict[4][f_name][alpha_list[ia]].max() > cuts[ix]:
                for w in range(1, 5):
                    if ix == 0:
                        for alpha in alpha_list[:(ia + 1)]:
                            forecasts_wo_ens_dict[w][f_name][alpha] = forecasts_wo_ens_dict[w][f_name][alpha].clip(upper=cuts[0])
                    else:
                        forecasts_wo_ens_dict[w][f_name][alpha_list[ia]] = forecasts_wo_ens_dict[w][f_name][alpha_list[ia]].clip(upper=cuts[ix])


def build_trimmed(geos: Optional[Sequence[str]] = None, impute_na: bool = True) -> None:
    geos = geos or cfg.GEO_LIST
    os.makedirs(cfg.TRIMMED_DIR, exist_ok=True)
    suffix = "" if impute_na else "_raw"

    for geo in geos:
        d = pickle.load(open(os.path.join(cfg.STATES_WEEKLY_DIR, f"{geo}{suffix}.pkl"), "rb"))
        alpha_list = list(d["alpha_list"])
        forecasts_wo_ens_dict = d["forecasts_dict"]

        # Split ensemble models out of the competing base-forecaster set.
        forecasts_ens_dict = {w: {} for w in range(1, 5)}
        f_list_wo_ens = [f for f in d["forecaster_list"] if f not in cfg.ENS_MODEL_NAMES]
        for w in range(1, 5):
            for ens_f in cfg.ENS_MODEL_NAMES:
                forecasts_ens_dict[w][ens_f] = forecasts_wo_ens_dict[w].pop(ens_f)

        for alpha_cut, cuts in cfg.OUTLIER_CUTS.get(geo, []):
            _smooth_outliers(forecasts_wo_ens_dict, f_list_wo_ens, alpha_list, alpha_cut, cuts)

        with gzip.open(os.path.join(cfg.TRIMMED_DIR, f"{geo}{suffix}.pkl.gz"), "wb") as fh:
            pickle.dump({
                "forecaster_list": f_list_wo_ens,
                "alpha_list": alpha_list,
                "dates_list": d["dates_list"],
                "Y": d["Y"],
                "unit": cfg.unit_for_geo(geo),
                "forecasts_dict": forecasts_wo_ens_dict,
                "forecasts_ens_dict": forecasts_ens_dict,
            }, fh)


# ---------------------------------------------------------------------------
# Stage 4: base-forecaster / ensemble score traces
# ---------------------------------------------------------------------------

def compute_base_forecaster_metrics(geo: str, impute_na: bool = True,
                                    round_Y_F: bool = True,
                                    out_dir: Optional[str] = None) -> None:
    """Precompute per-horizon base-forecaster and ensemble score traces.

    Uses the refactored :func:`data_io.load_geo_arrays` for array assembly and
    :mod:`metrics` for the score traces, replacing the bespoke code that used to
    live in a notebook. Writes
    ``wk{w}_{geo}[_raw]_baseforecasters.pkl`` to ``out_dir`` (default: the
    trimmed-data folder).
    """
    import data_io
    from metrics import (elementary_scores_grid_N_F, pinball_loss,
                         omni_error_from_scores, ql_error_from_pb_loss)

    out_dir = out_dir or cfg.BASE_FORECASTER_DIR
    os.makedirs(out_dir, exist_ok=True)
    suffix = "" if impute_na else "_raw"
    for w in range(1, 5):
        loaded = data_io.load_geo_arrays(geo, w, raw=not impute_na)
        alpha_list = loaded["alpha_list"]
        unit = loaded["unit"]
        T, N, F = loaded["T"], loaded["N"], loaded["F"]

        thetas_min = min(int(np.floor(np.nanmin(loaded["preds_TNF"][:, 0, :]) / unit)), 0)
        thetas_max = int(np.ceil(np.nanmax(loaded["preds_TNF"][:, -1, :]) / unit))
        m = thetas_max - thetas_min
        thetas = np.arange(thetas_min, thetas_max) + 0.5

        if round_Y_F:
            Y_scaled = np.round(loaded["y_arr"] / unit)
            preds = np.round(loaded["preds_TNF"] / unit)
            ens = np.round(loaded["ens_TNF"] / unit)
        else:
            Y_scaled = loaded["y_arr"] / unit
            preds = loaded["preds_TNF"] / unit
            ens = loaded["ens_TNF"] / unit

        F_ens = ens.shape[2]
        f_esc = np.zeros((T, N, m, F))
        f_pb = np.zeros((T, N, F))
        e_esc = np.zeros((T, N, m, F_ens))
        e_pb = np.zeros((T, N, F_ens))
        for t in range(T):
            f_esc[t] = elementary_scores_grid_N_F(preds[t], Y_scaled[t], thetas, alpha_list)
            f_pb[t] = pinball_loss(preds[t], Y_scaled[t], alpha_list[:, None]) / m
            e_esc[t] = elementary_scores_grid_N_F(ens[t], Y_scaled[t], thetas, alpha_list)
            e_pb[t] = pinball_loss(ens[t], Y_scaled[t], alpha_list[:, None]) / m

        f_score_trace = omni_error_from_scores(f_esc)
        f_pb_trace = ql_error_from_pb_loss(f_pb)
        out = {
            "geo": geo, "w": w, "round_Y_F": round_Y_F,
            "unit": unit, "T": T, "N": N, "m": m, "F": F,
            "dates_list": loaded["dates_list"], "alpha_list": alpha_list,
            "forecaster_names": loaded["forecaster_names"],
            "ens_model_names": loaded.get("ens_model_names"),
            "Y": pd.Series(Y_scaled, index=loaded["dates_list"]),
            "Y_orig": loaded["y_arr"],
            "forecasters_preds_history_orig": loaded["preds_TNF"],
            "ens_preds_history_orig": loaded["ens_TNF"],
            "forecasters_preds_history": preds,
            "forecasters_score_trace": f_score_trace,
            "best_forecaster_score_trace": f_score_trace.min(axis=1),
            "forecasters_pb_loss_history": f_pb,
            "forecasters_pb_loss_trace": f_pb_trace,
            "best_forecaster_pb_loss_trace": f_pb_trace.min(axis=1),
            "ens_preds_history": ens,
            "ens_score_trace": omni_error_from_scores(e_esc),
            "ens_pb_loss_history": e_pb,
            "ens_pb_loss_trace": ql_error_from_pb_loss(e_pb),
        }
        fname = f"wk{w}_{geo}{suffix}_baseforecasters{'' if round_Y_F else '_orig'}.pkl"
        pickle.dump(out, open(os.path.join(out_dir, fname), "wb"))

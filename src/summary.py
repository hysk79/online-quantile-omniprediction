"""Compact summaries of the experiment outputs used by the paper figures.

The full per-run result pickles (``results/omni``, ``results/hedge_ql``) and
the base-forecaster metric pickles (``results/base_forecasters``) are several
GB. The paper figures only need a few per-step traces of them, which these
builders collect into two small pickles under ``results/summary``:

- :func:`build_main_summary`        -> Figures 2, 3, 4, 6, 7, 9
- :func:`build_sensitivity_summary` -> Figures 5, 8

All arrays are indexed ``[week - 1, geo_index, ...]`` with ``geo_index`` into
``summary["geo_list"]`` (``config.GEO_LIST``: states first, US last).
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np

import config as cfg
import data_io
import utils

WEEKS = (1, 2, 3, 4)


def _load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _bf_path(bf_dir, w, geo, raw=False):
    return os.path.join(bf_dir, f"wk{w}_{geo}{'_raw' if raw else ''}_baseforecasters.pkl")


def build_main_summary(omni_dir: Optional[str] = None, ql_dir: Optional[str] = None,
                       bf_dir: Optional[str] = None, suffix: str = "",
                       out_path: Optional[str] = None, geos=None) -> dict:
    """Collect everything Figures 2, 3, 4, 6, 7, 9 need into one dict.

    Per (week, geo):
      ``Y`` (grid-scaled target), ``unit``, ``m``;
      base forecasters: best omni-error trace, per-step level-averaged pinball
      loss ``(T, F)`` for the imputed (``bf_pb``) and raw (``bf_raw_pb``) data;
      ensembles: omni-error trace and per-step pinball ``(T, E)``;
      our algorithm / Hedge-QL for every learning rate: omni-error trace
      (``omni_score``, ``ql_score``) and per-step pinball (``omni_pb``,
      ``ql_pb``), shape ``(n_eta, T)``.
    US only: quantile predictions of every run (``us_omni_preds``,
    ``us_ql_preds``; grid scale) and of the ensembles (``us_ens_preds_orig``).

    Pinball losses are divided by ``m`` (as in the result pickles).
    """
    omni_dir = omni_dir or cfg.SAVE_DIR_OMNI
    ql_dir = ql_dir or cfg.SAVE_DIR_QL_PB
    bf_dir = bf_dir or cfg.BASE_FORECASTER_DIR
    out_path = out_path or cfg.MAIN_SUMMARY_PATH
    geos = list(geos or cfg.GEO_LIST)
    eta_omni = np.asarray(cfg.ETA_LIST_OMNI)
    eta_ql = np.asarray(cfg.ETA_LIST_QL)
    W, G, T, N = len(WEEKS), len(geos), cfg.T, cfg.N_LEVELS

    bf0 = _load(_bf_path(bf_dir, 1, geos[0]))
    F, E = len(bf0["forecaster_names"]), bf0["ens_score_trace"].shape[1]
    raw0 = _load(_bf_path(bf_dir, 1, geos[0], raw=True))

    s = {
        "geo_list": geos, "weeks": list(WEEKS), "eta_omni": eta_omni, "eta_ql": eta_ql,
        "alpha_list": np.asarray(cfg.ALPHA_LIST), "dates": np.asarray(cfg.DATES_LIST),
        "forecaster_names": list(bf0["forecaster_names"]),
        "raw_forecaster_names": list(raw0["forecaster_names"]),
        "ens_model_names": list(bf0["ens_model_names"]),
        "Y": np.zeros((W, G, T)), "unit": np.zeros((W, G)), "m": np.zeros((W, G)),
        "bf_best_score": np.zeros((W, G, T)), "bf_pb": np.zeros((W, G, T, F)),
        "bf_raw_pb": np.zeros((W, G, T, F)), "raw_m": np.zeros((W, G)),
        "raw_Y": np.zeros((W, G, T)),
        "ens_score": np.zeros((W, G, T, E)), "ens_pb": np.zeros((W, G, T, E)),
        "omni_score": np.zeros((W, G, len(eta_omni), T)),
        "omni_pb": np.zeros((W, G, len(eta_omni), T)),
        "ql_score": np.zeros((W, G, len(eta_ql), T)),
        "ql_pb": np.zeros((W, G, len(eta_ql), T)),
    }
    if "US" in geos:
        s["us_omni_preds"] = np.zeros((W, len(eta_omni), T, N))
        s["us_ql_preds"] = np.zeros((W, len(eta_ql), T, N))
        s["us_ens_preds_orig"] = np.zeros((W, T, N, E))

    for wi, w in enumerate(WEEKS):
        print(f"[summary] week {w}")
        for gi, geo in enumerate(geos):
            bf = _load(_bf_path(bf_dir, w, geo))
            raw = _load(_bf_path(bf_dir, w, geo, raw=True))
            assert list(raw["forecaster_names"]) == s["raw_forecaster_names"]
            s["Y"][wi, gi] = np.asarray(bf["Y"])
            s["unit"][wi, gi] = bf["unit"]
            s["m"][wi, gi] = bf["m"]
            s["bf_best_score"][wi, gi] = bf["forecasters_score_trace"].min(axis=1)
            s["bf_pb"][wi, gi] = bf["forecasters_pb_loss_history"].mean(axis=1)
            s["bf_raw_pb"][wi, gi] = raw["forecasters_pb_loss_history"].mean(axis=1)
            s["raw_m"][wi, gi] = raw["m"]
            s["raw_Y"][wi, gi] = np.asarray(raw["Y"])
            s["ens_score"][wi, gi] = bf["ens_score_trace"]
            s["ens_pb"][wi, gi] = bf["ens_pb_loss_history"].mean(axis=1)

            for ei, eta in enumerate(eta_omni):
                r = _load(os.path.join(omni_dir, f"results_{utils.exp_name_omni_v2(w, geo, eta, suffix=suffix)}.pkl"))
                s["omni_score"][wi, gi, ei] = r["omni_score_trace"]
                s["omni_pb"][wi, gi, ei] = r["pb_loss_history"].mean(axis=1)
                if geo == "US":
                    s["us_omni_preds"][wi, ei] = r["phat_history"]
            for ei, eta in enumerate(eta_ql):
                r = _load(os.path.join(ql_dir, f"results_{utils.exp_name_ql_pb(w, geo, eta, True, suffix=suffix)}.pkl"))
                s["ql_score"][wi, gi, ei] = r["pinball_omni_score_trace"]
                s["ql_pb"][wi, gi, ei] = r["pinball_pb_loss_history"].mean(axis=1)
                if geo == "US":
                    s["us_ql_preds"][wi, ei] = r["pinball_preds_history"]
            if geo == "US":
                s["us_ens_preds_orig"][wi] = bf["ens_preds_history_orig"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(s, fh)
    print(f"[summary] wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return s


def build_sensitivity_summary(runtime_pkl: Optional[str] = None,
                              grid_all_weeks_pkl: Optional[str] = None,
                              out_path: Optional[str] = None) -> dict:
    """Collect what Figures 5 (runtime) and 8 (Total QL vs. grid unit) need.

    ``runtime_pkl`` is the dict ``{"grid", "quantile", "forecaster"}`` of rows
    from :mod:`sensitivity` (US, week 4, ``c_O = 1``); ``grid_all_weeks_pkl`` is
    ``{week: {"grid": rows, "grid_qlpb": rows}}`` from the grid-unit sweep over
    all four weeks and both learning-rate sweeps. Mean pinball losses are in
    original counts.
    """
    runtime_pkl = runtime_pkl or os.path.join(cfg.SENSITIVITY_DIR, "sensitivity_results.pkl")
    grid_all_weeks_pkl = grid_all_weeks_pkl or os.path.join(cfg.SENSITIVITY_DIR, "grid_unit_error_all_weeks.pkl")
    out_path = out_path or cfg.SENSITIVITY_SUMMARY_PATH

    d = _load(runtime_pkl)
    s = {
        "runtime_grid": [{k: r[k] for k in ("unit", "m", "runtime_sec")} for r in d["grid"]],
        "runtime_quantile": [{k: r[k] for k in ("n_quantiles", "m", "runtime_sec")} for r in d["quantile"]],
        "runtime_forecaster": [{k: r[k] for k in ("n_forecasters", "m", "runtime_sec")} for r in d["forecaster"]],
        "grid_ql": {},
    }

    all_weeks = _load(grid_all_weeks_pkl)
    for w, res in sorted(all_weeks.items()):
        g, q = res["grid"], res["grid_qlpb"]
        s["grid_ql"][w] = {
            "units": [r["unit"] for r in g],
            "m": [r["m"] for r in g],
            "eta_omni": list(g[0]["etas"]),
            "eta_ql": list(q[0]["etas"]),
            # (n_units, n_eta): time- and level-averaged expected pinball loss of our alg.
            "omni_mean_ql": np.array([[st["pb_loss_history_orig"].mean() for st in r["stats_by_eta"]] for r in g]),
            # (n_units, n_eta): same for Hedge-QL
            "ql_mean_ql": np.array([[st["pinball"]["mean_ql"]["objective_abs"] for st in r["stats_by_eta"]] for r in q]),
            "best_forecaster_mean_ql": g[0]["stats_by_eta"][0]["mean_ql"]["best_forecaster_mean_ql"],
            "y_max": float(data_io.load_geo_arrays(cfg.SENSITIVITY_GEO, w)["y_arr"].max()),
        }
        assert [r["unit"] for r in q] == s["grid_ql"][w]["units"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(s, fh)
    print(f"[summary] wrote {out_path} ({os.path.getsize(out_path) / 1e6:.2f} MB)")
    return s

"""
walk_forward.py
───────────────
Professional walk-forward (anchored expanding-window) validation engine
and its dedicated Plotly chart functions.

Walk-forward simulates real trading: the model is retrained from scratch
on each expanding window and tested only on unseen future data, giving a
truly unbiased performance estimate.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

from config import FEATURE_COLS, SEQ_LEN, apply_layout
from data import build_sequences
from models import (DL_BUILDERS, HAS_TF, get_classical_model,
                    honest_metrics, callbacks)


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def walk_forward_windows(df_feat: pd.DataFrame,
                         min_train_years: int = 2,
                         test_months: int = 12) -> list[dict]:
    """
    Build an anchored expanding-window schedule.

    Training always starts from the beginning of the dataset.
    Each window advances the training end by `test_months` months, then
    tests on the NEXT `test_months` months — data never seen during training.

    Example (min_train_years=2, test_months=12):
      Window 1 → Train: 2015-01–2016-12  |  Test: 2017-01–2017-12
      Window 2 → Train: 2015-01–2017-12  |  Test: 2018-01–2018-12
      ...
    """
    idx        = df_feat.index
    data_start = idx[0]
    data_end   = idx[-1]

    first_test_start = (data_start + pd.DateOffset(years=min_train_years)).replace(day=1)

    windows    = []
    test_start = first_test_start
    while True:
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        if test_end > data_end:
            break
        windows.append({
            "train_start": data_start,
            "train_end":   test_start - pd.Timedelta(days=1),
            "test_start":  test_start,
            "test_end":    test_end,
        })
        test_start = test_start + pd.DateOffset(months=test_months)

    return windows


def _wf_slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Return rows whose index falls in [start, end] inclusive."""
    return df.loc[(df.index >= start) & (df.index <= end)].copy()


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_walk_forward(df_feat: pd.DataFrame,
                     model_name: str,
                     min_train_years: int = 2,
                     test_months: int = 12,
                     progress_cb=None) -> dict:
    """
    Execute walk-forward validation and return stitched out-of-sample results.

    Returns dict with keys:
        all_y_true  — actual prices across all test windows
        all_y_pred  — predicted prices across all test windows
        all_ya_pct  — actual % returns
        all_yp_pct  — predicted % returns
        all_idx     — DatetimeIndex aligned to predictions
        per_window  — list of per-window metric dicts
        agg_metrics — aggregate metrics + consistency stats
        windows     — window schedule used
    """
    is_dl   = model_name in DL_BUILDERS
    windows = walk_forward_windows(df_feat, min_train_years, test_months)

    if not windows:
        raise ValueError("Not enough data for walk-forward. Use a longer date range (≥ 3 years).")

    all_yt, all_yp, all_yp_pct, all_ya_pct, all_idx = [], [], [], [], []
    per_window = []

    for i, w in enumerate(windows):
        if progress_cb:
            progress_cb(
                i / len(windows),
                f"Window {i+1}/{len(windows)} — "
                f"Train {w['train_start'].year}–{w['train_end'].year} "
                f"→ Test {w['test_start'].year}",
            )

        df_train = _wf_slice(df_feat, w["train_start"], w["train_end"])
        df_test  = _wf_slice(df_feat, w["test_start"],  w["test_end"])

        if len(df_train) < 100 or len(df_test) < 10:
            continue

        X_tr = df_train[FEATURE_COLS].values.astype(np.float32)
        y_tr = df_train["Target_pct"].values.astype(np.float32)
        X_te = df_test[FEATURE_COLS].values.astype(np.float32)
        y_te = df_test["Target_pct"].values.astype(np.float32)

        if is_dl:
            scaler  = MinMaxScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)
            Xtr_seq, ytr_seq = build_sequences(X_tr_sc, y_tr, SEQ_LEN)
            Xte_seq, yte_seq = build_sequences(X_te_sc, y_te, SEQ_LEN)
            if len(Xtr_seq) < 32 or len(Xte_seq) < 5:
                continue
            mdl = DL_BUILDERS[model_name](len(FEATURE_COLS), SEQ_LEN)
            cb  = [callbacks.EarlyStopping(monitor="loss", patience=5,
                                           restore_best_weights=True)]
            mdl.fit(Xtr_seq, ytr_seq, epochs=20, batch_size=32,
                    callbacks=cb, verbose=0)
            y_pred_pct = mdl.predict(Xte_seq, verbose=0).flatten()
            y_te_al    = yte_seq
            te_idx     = df_test.index[SEQ_LEN:]
            prev_c     = df_test["Close"].squeeze().values[SEQ_LEN - 1:-1]
        else:
            mdl        = get_classical_model(model_name)
            mdl.fit(X_tr, y_tr)
            y_pred_pct = mdl.predict(X_te).astype(np.float32)
            y_te_al    = y_te
            te_idx     = df_test.index
            prev_c     = df_test["Close"].squeeze().values[:-1]
            y_pred_pct = y_pred_pct[:-1] if len(prev_c) < len(y_pred_pct) else y_pred_pct
            y_te_al    = y_te_al[:len(prev_c)]
            te_idx     = te_idx[:len(prev_c)]

        n = min(len(prev_c), len(y_pred_pct), len(y_te_al))
        if n < 5:
            continue

        prev_c, y_pred_pct, y_te_al, te_idx = (
            prev_c[:n], y_pred_pct[:n], y_te_al[:n], te_idx[:n])

        y_true_p = prev_c * (1 + y_te_al)
        y_pred_p = prev_c * (1 + y_pred_pct)

        wm = honest_metrics(y_te_al, y_pred_pct, prev_c)
        wm.update({
            "Window":      f"{w['test_start'].strftime('%Y-%m')} – {w['test_end'].strftime('%Y-%m')}",
            "Train Years": f"{w['train_start'].year} – {w['train_end'].year}",
            "N Test":      n,
        })
        per_window.append(wm)

        all_yt.extend(y_true_p);     all_yp.extend(y_pred_p)
        all_ya_pct.extend(y_te_al);  all_yp_pct.extend(y_pred_pct)
        all_idx.extend(te_idx)

    if not all_yt:
        raise ValueError("All windows were too small. Increase date range or reduce min_train_years.")

    all_yt     = np.array(all_yt);     all_yp     = np.array(all_yp)
    all_ya_pct = np.array(all_ya_pct); all_yp_pct = np.array(all_yp_pct)
    all_idx    = pd.DatetimeIndex(all_idx)

    agg = honest_metrics(all_ya_pct, all_yp_pct,
                         all_yt / (1 + all_ya_pct + 1e-10))
    agg["N Windows"]   = len(per_window)
    agg["N Total"]     = len(all_yt)
    da_vals            = [w["Dir Acc"] for w in per_window]
    agg["Dir Acc Std"] = round(float(np.std(da_vals)),  4)
    agg["Dir Acc Min"] = round(float(np.min(da_vals)),  4)
    agg["Dir Acc Max"] = round(float(np.max(da_vals)),  4)

    if progress_cb:
        progress_cb(1.0, "Walk-forward complete ✓")

    return {
        "all_y_true": all_yt,   "all_y_pred":  all_yp,
        "all_ya_pct": all_ya_pct, "all_yp_pct": all_yp_pct,
        "all_idx":    all_idx,  "per_window":  per_window,
        "agg_metrics": agg,     "windows":     windows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def chart_wf_overview(wf: dict) -> go.Figure:
    """Actual vs Predicted across all windows with alternating shaded bands."""
    fig     = go.Figure()
    palette = ["rgba(204,0,0,0.06)", "rgba(79,195,247,0.04)"]
    for j, w in enumerate(wf["windows"]):
        ts, te = w["test_start"], w["test_end"]
        if ts > wf["all_idx"][-1]: break
        fig.add_vrect(x0=ts, x1=min(te, wf["all_idx"][-1]),
                      fillcolor=palette[j % 2], line_width=0, layer="below")
        fig.add_annotation(x=ts, y=1.0, yref="paper", text=f"W{j+1}",
                           showarrow=False, font=dict(size=9, color="#7D8590"),
                           xanchor="left")
    fig.add_trace(go.Scatter(x=wf["all_idx"], y=wf["all_y_true"],
                             name="Actual",    line=dict(color="#4FC3F7", width=1.6)))
    fig.add_trace(go.Scatter(x=wf["all_idx"], y=wf["all_y_pred"],
                             name="Predicted", line=dict(color="#CC0000", width=1.4, dash="dash")))
    apply_layout(fig, height=420,
                 title="Walk-Forward: Actual vs Predicted across all test windows")
    fig.update_yaxes(title_text="Close Price")
    return fig


def chart_wf_dir_acc(wf: dict) -> go.Figure:
    """Per-window Direction Accuracy bars with 50% random baseline."""
    pw     = wf["per_window"]
    lbs    = [p["Window"] for p in pw]
    das    = [p["Dir Acc"] * 100 for p in pw]
    colors = ["#00C896" if d > 50 else "#CC0000" for d in das]
    fig    = go.Figure(go.Bar(x=lbs, y=das, marker_color=colors,
                              text=[f"{d:.1f}%" for d in das],
                              textposition="outside"))
    fig.add_hline(y=50, line_dash="dot", line_color="#C9A84C",
                  annotation_text="50% = random baseline",
                  annotation_font_color="#C9A84C")
    apply_layout(fig, height=340,
                 title="Direction Accuracy per Window (consistency is key)")
    fig.update_yaxes(range=[40, 70], ticksuffix="%")
    fig.update_xaxes(tickangle=-30)
    return fig


def chart_wf_equity(wf: dict, tx_cost: float = 0.001):
    """
    Stitched out-of-sample equity curve.
    Returns (fig, bt_dict) where bt_dict contains summary performance stats.
    """
    actual_pct = wf["all_ya_pct"]; pred_pct = wf["all_yp_pct"]
    n          = min(len(actual_pct), len(pred_pct))
    act        = actual_pct[:n];   prd = pred_pct[:n]
    pos        = np.sign(prd)
    pos_lag    = np.concatenate([[0], pos[:-1]])
    cost       = np.abs(np.diff(np.concatenate([[0], pos_lag]))) * tx_cost
    strat_r    = pos_lag * act - cost
    strat_cum  = np.cumprod(1 + strat_r)
    bh_cum     = np.cumprod(1 + act)
    peak       = np.maximum.accumulate(strat_cum)
    dd         = (strat_cum - peak) / peak * 100
    bt = {
        "strat_cum": strat_cum, "bh_cum":    bh_cum,
        "sharpe":    (strat_r.mean()/(strat_r.std()+1e-10))*np.sqrt(252),
        "max_dd":    float(dd.min()),
        "total_ret": float(strat_cum[-1]-1)*100,
        "bh_ret":    float(bh_cum[-1]-1)*100,
        "win_rate":  float((strat_r>0).mean()*100),
    }
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wf["all_idx"][:len(strat_cum)], y=strat_cum,
                             name="WF Strategy", line=dict(color="#CC0000", width=2)))
    fig.add_trace(go.Scatter(x=wf["all_idx"][:len(bh_cum)], y=bh_cum,
                             name="Buy & Hold",  line=dict(color="#4FC3F7", width=1.5, dash="dot")))
    apply_layout(fig, height=360,
                 title="Walk-Forward Equity Curve (truly out-of-sample)")
    fig.update_yaxes(title_text="Growth of $1")
    return fig, bt


def chart_wf_metric_trend(wf: dict) -> go.Figure:
    """Dir Acc and RMSE stability line charts across successive windows."""
    pw   = wf["per_window"]
    lbs  = [p["Window"]   for p in pw]
    das  = [p["Dir Acc"] * 100 for p in pw]
    rmse = [p["RMSE ret%"] for p in pw]
    fig  = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         vertical_spacing=0.08,
                         subplot_titles=("Direction Accuracy % per window",
                                         "Return RMSE % per window"))
    fig.add_trace(go.Scatter(x=lbs, y=das, mode="lines+markers",
                             line=dict(color="#CC0000", width=2),
                             marker=dict(size=7), name="Dir Acc %"), row=1, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#C9A84C", row=1, col=1)
    fig.add_trace(go.Scatter(x=lbs, y=rmse, mode="lines+markers",
                             line=dict(color="#4FC3F7", width=2),
                             marker=dict(size=7), name="RMSE ret%"), row=2, col=1)
    apply_layout(fig, height=460,
                 title="Walk-Forward Metric Stability across test windows")
    fig.update_xaxes(tickangle=-30)
    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    return fig

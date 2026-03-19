"""
charts.py
─────────
All Plotly chart builders, the sentiment signal engine,
and the multi-day recursive forecast engine.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import FEATURE_COLS, SEQ_LEN, apply_layout
from models import DL_BUILDERS


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def generate_sentiment(df_feat: pd.DataFrame) -> list[dict]:
    """
    Rule-based market sentiment from the latest indicator snapshot.
    Returns a list of {text, type, score} dicts.
    type: 'bull' | 'bear' | 'neut'   |   score: –1.0 to +1.0
    """
    last = df_feat.iloc[-1]
    prev = df_feat.iloc[-2] if len(df_feat) > 1 else last
    sigs = []

    # ── Moving average trend ──────────────────────────────────────────────────
    ma20r = float(last["MA20_r"]); ma50r = float(last["MA50_r"])
    if ma20r < 1 and ma50r < 1:
        sigs.append({"text": "📈 Price above MA20 & MA50 — confirmed uptrend",   "type": "bull", "score":  0.8})
    elif ma20r > 1 and ma50r > 1:
        sigs.append({"text": "📉 Price below MA20 & MA50 — confirmed downtrend", "type": "bear", "score": -0.8})
    elif ma20r < 1:
        sigs.append({"text": "⚡ Price above MA20 but below MA50 — mixed",        "type": "neut", "score":  0.2})
    else:
        sigs.append({"text": "⚠️ Price below MA20, above MA50 — weakening",      "type": "neut", "score": -0.2})

    # ── RSI ───────────────────────────────────────────────────────────────────
    rsi = float(last["RSI"])
    if   rsi > 75: sigs.append({"text": f"🔴 RSI {rsi:.1f} — severely overbought",        "type": "bear", "score": -0.9})
    elif rsi > 65: sigs.append({"text": f"🟠 RSI {rsi:.1f} — overbought zone",             "type": "bear", "score": -0.5})
    elif rsi < 25: sigs.append({"text": f"🟢 RSI {rsi:.1f} — severely oversold",           "type": "bull", "score":  0.9})
    elif rsi < 35: sigs.append({"text": f"🟡 RSI {rsi:.1f} — oversold, watch for bounce",  "type": "bull", "score":  0.5})
    else:          sigs.append({"text": f"⚖️ RSI {rsi:.1f} — neutral territory",           "type": "neut", "score":  0.0})

    # ── MACD crossover ────────────────────────────────────────────────────────
    mn = float(last["MACD_n"]); sn = float(last["MACD_sig_n"])
    pm = float(prev["MACD_n"]); ps = float(prev["MACD_sig_n"])
    if   pm < ps and mn > sn: sigs.append({"text": "⭐ MACD bullish crossover just triggered", "type": "bull", "score":  0.85})
    elif pm > ps and mn < sn: sigs.append({"text": "💀 MACD bearish crossover just triggered", "type": "bear", "score": -0.85})
    elif mn > sn:             sigs.append({"text": "✅ MACD above signal — bullish momentum",  "type": "bull", "score":  0.4})
    else:                     sigs.append({"text": "❌ MACD below signal — bearish momentum",  "type": "bear", "score": -0.4})

    # ── Bollinger Bands position ──────────────────────────────────────────────
    bb = float(last["BB_pos"])
    if   bb > 0.95: sigs.append({"text": "📊 Price at upper BB — potential overextension",    "type": "bear", "score": -0.6})
    elif bb < 0.05: sigs.append({"text": "📊 Price at lower BB — potential bounce zone",      "type": "bull", "score":  0.6})
    else:           sigs.append({"text": f"📊 Price within BB (position: {bb:.0%})",          "type": "neut", "score":  0.0})

    # ── Volume signal ─────────────────────────────────────────────────────────
    vr   = float(last["Vol_ratio"]) if not pd.isna(last["Vol_ratio"]) else 1.0
    ret1 = float(last["Ret1"])      if not pd.isna(last["Ret1"])      else 0.0
    if   vr > 1.5 and ret1 > 0: sigs.append({"text": f"📣 High vol ({vr:.1f}×) on up day — buying signal",    "type": "bull", "score":  0.7})
    elif vr > 1.5 and ret1 < 0: sigs.append({"text": f"📣 High vol ({vr:.1f}×) on down day — distribution",  "type": "bear", "score": -0.7})
    else:                        sigs.append({"text": f"📦 Volume normal ({vr:.1f}× 20d avg)",                "type": "neut", "score":  0.0})

    return sigs


# ─────────────────────────────────────────────────────────────────────────────
# RECURSIVE MULTI-DAY FORECAST ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def multi_day_forecast(model_obj, df_feat: pd.DataFrame,
                       active_model: str, horizon: int,
                       close_now: float) -> list[dict]:
    """
    Recursive multi-day forecast with de-biasing, volatility calibration,
    and mean-reversion correction.

    Each step:
      1. Get raw model prediction (% return).
      2. Extract direction signal (de-biased against recent model mean).
      3. Calibrate magnitude to historical daily volatility.
      4. Apply gentle mean-reversion toward 20-day rolling mean.
      5. Add small calibrated noise from historical return distribution.
      6. Advance exactly ONE business day (BDay(1)) each step.

    Returns list of dicts: [{ day, date, date_ts, price, pct_change, direction }]
    date_ts is a pd.Timestamp so the chart can share a datetime x-axis
    with the historical price line.
    """
    np.random.seed(42)

    hist      = df_feat.copy()
    close_ser = hist["Close"].squeeze()
    last_date = hist.index[-1]

    # Historical calibration anchors
    hist_returns = close_ser.pct_change().dropna()
    hist_vol     = float(hist_returns.std())
    hist_mean    = float(hist_returns.mean())
    rolling_mean = float(close_ser.rolling(20).mean().iloc[-1])

    # De-bias: average model output over last 60 rows
    recent_feats = hist[FEATURE_COLS].iloc[-60:].values.astype(np.float32)
    if active_model in DL_BUILDERS:
        keras_m, scaler_ = model_obj
        sc_feats = scaler_.transform(recent_feats)
        if len(sc_feats) >= SEQ_LEN:
            seqs = np.array([sc_feats[j:j+SEQ_LEN]
                             for j in range(len(sc_feats) - SEQ_LEN + 1)])
            bias = float(keras_m.predict(seqs, verbose=0).mean())
        else:
            bias = hist_mean
    else:
        bias = float(model_obj.predict(recent_feats).mean())

    results = []

    for i in range(horizon):
        last_feat  = hist[FEATURE_COLS].iloc[[-1]].values.astype(np.float32)
        last_close = float(hist["Close"].squeeze().iloc[-1])

        # Raw prediction
        if active_model in DL_BUILDERS:
            keras_m, scaler_ = model_obj
            feat_sc  = scaler_.transform(hist[FEATURE_COLS].values.astype(np.float32))
            seq      = feat_sc[-SEQ_LEN:].reshape(1, SEQ_LEN, len(FEATURE_COLS))
            raw_pred = float(keras_m.predict(seq, verbose=0).flatten()[0])
        else:
            raw_pred = float(model_obj.predict(last_feat)[0])

        # Direction signal (de-biased)
        direction_signal = np.sign(raw_pred - bias)
        if direction_signal == 0:
            direction_signal = np.sign(raw_pred) if raw_pred != 0 else 1.0

        # Calibrated return + mean-reversion + noise
        calibrated_return = direction_signal * hist_vol * 0.6
        deviation  = (last_close - rolling_mean) / (rolling_mean + 1e-9)
        mean_rev   = -deviation * 0.15
        noise      = np.random.normal(0, hist_vol * 0.3)
        final_ret  = np.clip(calibrated_return + mean_rev + noise,
                             -hist_vol * 3, hist_vol * 3)

        next_price = last_close * (1 + final_ret)
        next_date  = last_date + pd.tseries.offsets.BDay(1)   # exactly +1 BDay

        results.append({
            "day":        i + 1,
            "date":       next_date.strftime("%d %b %Y"),
            "date_ts":    next_date,
            "price":      next_price,
            "pct_change": final_ret * 100,
            "direction":  "▲" if final_ret >= 0 else "▼",
        })

        rolling_mean = rolling_mean * (19/20) + next_price * (1/20)

        new_row          = hist.iloc[[-1]].copy()
        new_row.index    = [next_date]
        new_row["Close"] = next_price
        for lag in [10, 5, 3, 2, 1]:
            if f"Ret{lag}" in new_row.columns and f"Ret{lag-1}" in new_row.columns and lag > 1:
                new_row[f"Ret{lag}"] = hist[f"Ret{lag-1}"].iloc[-1]
        new_row["Ret1"] = final_ret
        hist      = pd.concat([hist, new_row])
        last_date = next_date

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PRICE CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def chart_price(df: pd.DataFrame, df_feat: pd.DataFrame) -> go.Figure:
    """Full candlestick chart with MA overlays, volume, and RSI."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.22, 0.23], vertical_spacing=0.025,
                        subplot_titles=("", "Volume", "RSI (14)"))

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"].squeeze(), high=df["High"].squeeze(),
        low=df["Low"].squeeze(), close=df["Close"].squeeze(),
        increasing_line_color="#00C896", decreasing_line_color="#CC0000",
        name="OHLC", showlegend=False,
    ), row=1, col=1)

    for col_name, label, color in [("_MA20","MA20","#C9A84C"),("_MA50","MA50","#4FC3F7")]:
        if col_name in df_feat.columns:
            fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat[col_name].squeeze(),
                                     name=label, line=dict(color=color, width=1.3),
                                     opacity=0.8), row=1, col=1)

    if "_BB_upper" in df_feat.columns:
        fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["_BB_upper"].squeeze(),
                                 line=dict(color="rgba(150,150,255,0.4)", width=1, dash="dot"),
                                 showlegend=False, name="BB Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["_BB_lower"].squeeze(),
                                 line=dict(color="rgba(150,150,255,0.4)", width=1, dash="dot"),
                                 fill="tonexty", fillcolor="rgba(100,100,255,0.04)",
                                 showlegend=False, name="BB Lower"), row=1, col=1)

    if "Volume" in df.columns:
        vol_c = ["#CC0000" if float(o) > float(c) else "#00C896"
                 for o, c in zip(df["Open"].squeeze(), df["Close"].squeeze())]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"].squeeze(),
                             marker_color=vol_c, showlegend=False), row=2, col=1)

    if "RSI" in df_feat.columns:
        fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["RSI"].squeeze(),
                                 line=dict(color="#CE93D8", width=1.5),
                                 showlegend=False), row=3, col=1)
        for lv, col in [(70, "#CC0000"), (30, "#00C896")]:
            fig.add_hline(y=lv, line_dash="dot", line_color=col, line_width=1, row=3, col=1)

    apply_layout(fig, height=680, xaxis_rangeslider_visible=False)
    return fig


def chart_avp(test_idx, y_true, y_pred, model_name: str) -> go.Figure:
    """Actual vs Predicted price chart on the test set."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_idx, y=y_true, name="Actual",
                              line=dict(color="#4FC3F7", width=1.8)))
    fig.add_trace(go.Scatter(x=test_idx, y=y_pred, name="Predicted",
                              line=dict(color="#CC0000", width=1.8, dash="dash")))
    apply_layout(fig, height=380, title=f"Actual vs Predicted · {model_name}")
    fig.update_yaxes(title_text="Close Price")
    return fig


def chart_fi(fi: pd.DataFrame, model_name: str) -> go.Figure:
    """Horizontal bar chart showing feature importances."""
    top = fi.head(12).sort_values("Importance")
    fig = go.Figure(go.Bar(
        x=top["Importance"], y=top["Feature"], orientation="h",
        marker=dict(color=top["Importance"],
                    colorscale=[[0, "#1C2128"], [1, "#CC0000"]], showscale=False),
    ))
    apply_layout(fig, height=370, title=f"Feature Importance · {model_name}")
    fig.update_xaxes(title_text="Importance")
    return fig


def chart_forecast(results: list[dict], df_feat: pd.DataFrame,
                   close_now: float, ccy_sym: str, ticker: str) -> go.Figure:
    """
    Combined historical (last 90 days) + recursive forecast chart.
    Uses pd.Timestamp x-axis throughout so both traces share one timeline.
    Green dots = predicted up day, red dots = predicted down day.
    """
    forecast_ts     = [r["date_ts"] for r in results]
    forecast_prices = [r["price"]   for r in results]
    dot_colors      = ["#00C896" if r["pct_change"] >= 0 else "#CC0000"
                       for r in results]
    horizon         = len(results)

    hist_slice  = df_feat["Close"].squeeze().iloc[-90:]
    hist_dates  = hist_slice.index
    hist_prices = hist_slice.values

    fig = go.Figure()

    # Shaded forecast window
    fig.add_vrect(x0=forecast_ts[0], x1=forecast_ts[-1],
                  fillcolor="rgba(79,195,247,0.06)", line_width=0, layer="below",
                  annotation_text=f"  {horizon}-Day Forecast",
                  annotation_position="top left",
                  annotation_font_color="#4FC3F7", annotation_font_size=11)

    # Historical line
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_prices, mode="lines", name="Historical",
        line=dict(color="#E6EDF3", width=2),
        hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{ccy_sym}%{{y:.2f}}<extra>Historical</extra>",
    ))

    # Bridge connector
    fig.add_trace(go.Scatter(
        x=[hist_dates[-1], forecast_ts[0]],
        y=[float(hist_prices[-1]), forecast_prices[0]],
        mode="lines", line=dict(color="#7D8590", width=1.5, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))

    # Forecast line + coloured markers
    fig.add_trace(go.Scatter(
        x=forecast_ts, y=forecast_prices, mode="lines+markers", name="Forecast",
        line=dict(color="#4FC3F7", width=2, dash="dot"),
        marker=dict(color=dot_colors, size=9, line=dict(color="#0D1117", width=1.5)),
        hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>Forecast: {ccy_sym}%{{y:.2f}}<extra></extra>",
    ))

    # Last close reference
    fig.add_hline(y=close_now,
                  line_dash="dot", line_color="rgba(125,133,144,0.5)", line_width=1,
                  annotation_text=f"Last Close: {ccy_sym}{close_now:.2f}",
                  annotation_font_color="#7D8590", annotation_position="bottom right")

    apply_layout(fig, height=460,
                 title=f"{ticker} — Historical Prices + {horizon}-Day Recursive Forecast")
    fig.update_yaxes(title_text=f"Price ({ccy_sym})")
    fig.update_xaxes(title_text="Date", type="date")
    return fig

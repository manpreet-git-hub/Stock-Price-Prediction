"""
app.py
──────
Streamlit UI — the only file that imports and calls Streamlit directly.
All business logic lives in the sibling modules:

    config.py        — constants, stock registry, Plotly layout
    data.py          — data loading (yfinance) + feature engineering
    models.py        — ML / DL model builders, training, metrics
    walk_forward.py  — walk-forward validation engine + charts
    charts.py        — price charts, forecast engine, sentiment

Run:
    streamlit run app.py
"""

from __future__ import annotations
import warnings, logging, os
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from datetime import date

# ── Project modules ────────────────────────────────────────────────────────────
from config import STOCKS, FEATURE_COLS, SEQ_LEN, apply_layout
from data import load_data, add_features, compute_rsi
from models import (
    DL_BUILDERS, HAS_XGB, HAS_LGB, HAS_TF,
    available_models, train_classical, train_deep, honest_metrics,
)
from walk_forward import (
    run_walk_forward,
    chart_wf_overview, chart_wf_dir_acc,
    chart_wf_equity,   chart_wf_metric_trend,
)
from charts import (
    generate_sentiment, multi_day_forecast,
    chart_price, chart_avp, chart_fi, chart_forecast,
)

# Alias so compare tab can call apply_layout directly
_pl = apply_layout

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');
:root {
    --red:#CC0000; --red-dim:#7A0000; --gold:#C9A84C;
    --green:#00C896; --bg:#080B0F; --bg1:#0D1117;
    --bg2:#161B22; --bg3:#1C2128; --border:#30363D;
    --text:#E6EDF3; --muted:#7D8590;
}
html,body,[class*="css"]{background:var(--bg)!important;color:var(--text)!important;font-family:'Inter',sans-serif!important;}
[data-testid="stSidebar"]{background:var(--bg1)!important;border-right:1px solid var(--border);}
.hero-title{font-family:'Bebas Neue',sans-serif;font-size:3rem;letter-spacing:0.14em;color:var(--text);line-height:1;}
.hero-sub{font-size:0.7rem;letter-spacing:0.32em;color:var(--red);text-transform:uppercase;font-weight:500;}
.hero-divider{height:2px;background:linear-gradient(90deg,var(--red) 0%,transparent 75%);margin:0.7rem 0 1.3rem 0;}
.sec-title{font-family:'Bebas Neue',sans-serif;font-size:1.25rem;letter-spacing:0.12em;color:var(--text);
           border-left:3px solid var(--red);padding-left:0.6rem;margin:1.6rem 0 0.7rem 0;}
.kpi{background:var(--bg2);border:1px solid var(--border);border-top:2px solid var(--red);border-radius:4px;padding:0.7rem 0.9rem;}
.kpi.green{border-top-color:var(--green);}.kpi.gold{border-top-color:var(--gold);}
.kpi-label{font-size:0.62rem;letter-spacing:0.18em;color:var(--muted);text-transform:uppercase;margin-bottom:2px;}
.kpi-value{font-family:'IBM Plex Mono',monospace;font-size:1.35rem;font-weight:600;color:var(--text);}
.kpi-sub{font-size:0.68rem;color:var(--muted);margin-top:1px;}
.kpi-sub.up{color:var(--green);}.kpi-sub.down{color:var(--red);}
.model-badge{display:inline-block;background:var(--red-dim);color:var(--text);font-size:0.65rem;
             font-weight:600;letter-spacing:0.1em;padding:2px 8px;border-radius:3px;
             text-transform:uppercase;margin-bottom:0.4rem;}
.dl-badge{background:linear-gradient(90deg,#1a0044,#44007a);border:1px solid #7C3AED;}
.sent-card{background:var(--bg2);border:1px solid var(--border);border-left:3px solid var(--gold);
           border-radius:4px;padding:0.6rem 0.85rem;margin-bottom:0.4rem;font-size:0.8rem;}
.sent-card.bull{border-left-color:var(--green);}.sent-card.bear{border-left-color:var(--red);}
.stButton>button{background:var(--red)!important;color:white!important;border:none!important;
                 border-radius:3px!important;font-weight:600!important;letter-spacing:0.06em!important;}
.stButton>button:hover{background:var(--red-dim)!important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPER
# ─────────────────────────────────────────────────────────────────────────────
def kpi_html(label, value, sub="", cls=""):
    sub_cls = "up" if "▲" in sub else ("down" if "▼" in sub else "")
    return (f"<div class='kpi {cls}'>"
            f"<div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div>"
            f"<div class='kpi-sub {sub_cls}'>{sub}</div></div>")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:0.8rem 0 0.3rem'>
        <div style='font-size:0.62rem;letter-spacing:0.3em;color:#7D8590;
                    text-transform:uppercase;'>AI Stock Predictor</div>
    </div>
    <hr style='border-color:#30363D;margin:0.3rem 0'>
    """, unsafe_allow_html=True)

    st.markdown("**📌 STOCK**")
    stock_label    = st.selectbox("Select Stock", list(STOCKS.keys()), index=0)
    stock_info     = STOCKS[stock_label]
    active_ticker  = stock_info["ticker"]
    active_name    = stock_info["name"]
    active_exchange= stock_info["exchange"]
    active_currency= stock_info["currency"]
    active_emoji   = stock_info["emoji"]
    active_color   = stock_info["color"]

    st.markdown(
        f"<div style='background:#161B22;border:1px solid #30363D;border-radius:4px;"
        f"padding:0.45rem 0.8rem;margin-bottom:0.3rem;'>"
        f"<span style='font-size:1.1rem;'>{active_emoji}</span>"
        f"<span style='font-family:IBM Plex Mono;font-size:0.85rem;color:#E6EDF3;"
        f"margin-left:0.5rem;font-weight:600;'>{active_ticker}</span>"
        f"<span style='font-size:0.65rem;color:#7D8590;margin-left:0.5rem;'>"
        f"{active_exchange} · {active_currency}</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown("**📅 DATA**")
    start_date = st.date_input("Start", value=date(2015, 1, 1))
    end_date   = st.date_input("End",   value=date.today())
    interval   = st.selectbox("Interval", ["1d", "1wk"], index=0)

    st.markdown("<hr style='border-color:#30363D'>", unsafe_allow_html=True)
    st.markdown("**🤖 MODEL**")
    model_choices = available_models()
    model_name    = st.selectbox("Algorithm", model_choices)
    is_dl         = model_name in DL_BUILDERS
    epochs        = st.slider("DL Epochs", 10, 100, 40, 5) if is_dl else 40
    test_pct      = st.slider("Test Size %", 10, 30, 20, 5) / 100

    st.markdown("<hr style='border-color:#30363D'>", unsafe_allow_html=True)
    st.markdown("**🔄 WALK-FORWARD**")
    wf_min_train = st.slider("Min Train Years", 1, 4, 2, 1,
                             help="Minimum years of data before first test window")
    wf_test_mo   = st.selectbox("Test Window", [3, 6, 12], index=2,
                                format_func=lambda x: f"{x} months",
                                help="How many months each test period covers")

    st.markdown("<hr style='border-color:#30363D'>", unsafe_allow_html=True)
    load_btn    = st.button("⬇️  Load & Train",    use_container_width=True)
    wf_btn      = st.button("🔄  Walk-Forward",    use_container_width=True)
    compare_btn = st.button("📊  Compare All",     use_container_width=True)

    if not HAS_TF:
        st.warning("TensorFlow not found.\n`pip install tensorflow`\nLSTM/TCN/Transformer unavailable.")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER  (single render — session state with sidebar fallback)
# ─────────────────────────────────────────────────────────────────────────────
_h_emoji    = st.session_state.get("stock_emoji",    stock_info["emoji"])
_h_ticker   = st.session_state.get("stock_ticker",   stock_info["ticker"])
_h_name     = st.session_state.get("stock_name",     stock_info["name"])
_h_exchange = st.session_state.get("stock_exchange", stock_info["exchange"])

st.markdown(f"""
<div style='padding:1rem 0 0.2rem'>
    <div class='hero-title'>{_h_emoji} {_h_ticker}</div>
    <div class='hero-sub'>
        {_h_name} &nbsp;·&nbsp; {_h_exchange} &nbsp;·&nbsp;
        AI Stock Price Prediction &nbsp;·&nbsp; Deep Learning + Classical ML
    </div>
</div>
<div class='hero-divider'></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in dict(
    df_raw=None, df_feat=None, model=None, metrics=None,
    y_true=None, y_pred_p=None, y_pred_pct_=None, test_idx=None,
    fi=None, active_model=None, wf_result=None, wf_model=None,
    stock_ticker="RACE", stock_name="Ferrari N.V.",
    stock_exchange="NASDAQ", stock_currency="USD",
    stock_emoji="🏎️", stock_color="#CC0000",
).items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & TRAIN
# ─────────────────────────────────────────────────────────────────────────────
if load_btn:
    with st.spinner(f"📡 Fetching {active_ticker} data from Yahoo Finance…"):
        _df_raw = load_data(str(start_date), str(end_date), interval, active_ticker)

    if _df_raw.empty:
        st.error("❌ No data. Check date range / internet."); st.stop()

    with st.spinner("⚙️ Engineering features…"):
        _df_feat = add_features(_df_raw)

    if is_dl and not HAS_TF:
        st.error("TensorFlow required for deep learning models."); st.stop()

    msg = f"🧠 Training {model_name} — deep learning (30–90 s)…" if is_dl else f"🔧 Training {model_name}…"
    with st.spinner(msg):
        if is_dl:
            _m, _met, _yt, _yp, _yp_pct, _tidx, _fi = train_deep(_df_feat, model_name, test_pct, epochs)
        else:
            _m, _met, _yt, _yp, _yp_pct, _tidx, _fi = train_classical(_df_feat, model_name, test_pct)

    st.session_state.update(dict(
        df_raw=_df_raw, df_feat=_df_feat, model=_m, metrics=_met,
        y_true=_yt, y_pred_p=_yp, y_pred_pct_=_yp_pct,
        test_idx=_tidx, fi=_fi, active_model=model_name,
        stock_ticker=active_ticker,    stock_name=active_name,
        stock_exchange=active_exchange, stock_currency=active_currency,
        stock_emoji=active_emoji,      stock_color=active_color,
    ))
    st.success(f"✅ {active_emoji} {active_ticker} · {model_name} ready · "
               f"Dir Accuracy: {_met['Dir Acc']*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD TRIGGER
# ─────────────────────────────────────────────────────────────────────────────
if wf_btn:
    if st.session_state["df_feat"] is None:
        st.warning("⚠️  Load data first (click **Load & Train**).")
    elif is_dl and not HAS_TF:
        st.error("TensorFlow required for deep learning walk-forward.")
    else:
        _df_feat = st.session_state["df_feat"]
        _wf_prog = st.progress(0, text="Initialising walk-forward…")
        try:
            def _prog_cb(frac, txt):
                _wf_prog.progress(min(frac, 0.99), text=txt)

            with st.spinner(f"🔄 Walk-forward — {model_name} "
                            f"({wf_min_train}yr min, {wf_test_mo}mo windows)…"):
                _wf = run_walk_forward(_df_feat, model_name,
                                       min_train_years=wf_min_train,
                                       test_months=wf_test_mo,
                                       progress_cb=_prog_cb)
            _wf_prog.empty()
            st.session_state["wf_result"] = _wf
            st.session_state["wf_model"]  = model_name
            agg = _wf["agg_metrics"]
            st.success(
                f"✅ Walk-forward complete · {agg['N Windows']} windows · "
                f"{agg['N Total']} predictions · "
                f"Agg Dir Acc: **{agg['Dir Acc']*100:.1f}%** "
                f"(range {agg['Dir Acc Min']*100:.1f}%–{agg['Dir Acc Max']*100:.1f}%)"
            )
        except ValueError as e:
            _wf_prog.empty(); st.error(f"Walk-forward error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# GUARD — nothing loaded yet
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state["df_raw"] is None:
    st.info("👈  Set parameters in the sidebar and click **Load & Train**.")
    st.stop()

# Aliases
df_raw       = st.session_state["df_raw"]
df_feat      = st.session_state["df_feat"]
metrics      = st.session_state["metrics"]
y_true       = st.session_state["y_true"]
y_pred_p     = st.session_state["y_pred_p"]
y_pred_pct_  = st.session_state["y_pred_pct_"]
test_idx     = st.session_state["test_idx"]
fi           = st.session_state["fi"]
active_model = st.session_state["active_model"]
model_obj    = st.session_state["model"]

s_ticker   = st.session_state["stock_ticker"]
s_name     = st.session_state["stock_name"]
s_exchange = st.session_state["stock_exchange"]
s_currency = st.session_state["stock_currency"]
s_emoji    = st.session_state["stock_emoji"]
s_color    = st.session_state["stock_color"]
sym_label  = f"{s_emoji}  {s_ticker}"
ccy_sym    = "₹" if s_currency == "INR" else "$"

close_now  = float(df_raw["Close"].squeeze().iloc[-1])
close_prev = float(df_raw["Close"].squeeze().iloc[-2])
chg        = close_now - close_prev
chg_pct    = chg / close_prev * 100


# ─────────────────────────────────────────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────────────────────────────────────────
is_dl_active = active_model in DL_BUILDERS
bcls = "model-badge dl-badge" if is_dl_active else "model-badge"
icon = "🧠 Deep Learning" if is_dl_active else "🔧 Classical ML"
st.markdown(
    f"<span class='{bcls}'>{icon} · {active_model}</span>"
    f"<span style='margin-left:0.6rem;font-size:0.72rem;color:#7D8590;'>"
    f"{sym_label} · {s_name} · {s_exchange}</span>",
    unsafe_allow_html=True,
)

ks   = st.columns(5)
sign = "▲" if chg >= 0 else "▼"
ks[0].markdown(kpi_html("Last Close",  f"{ccy_sym}{close_now:.2f}",
               f"{sign} {ccy_sym}{abs(chg):.2f} ({chg_pct:+.1f}%)"), unsafe_allow_html=True)
da     = metrics["Dir Acc"]
da_lbl = "Strong ✅" if da > 0.55 else ("Decent" if da > 0.52 else "Near random ⚠️")
ks[1].markdown(kpi_html("Dir Accuracy", f"{da*100:.1f}%", da_lbl,
               "green" if da > 0.52 else ""), unsafe_allow_html=True)
ks[2].markdown(kpi_html("R² Score",   f"{metrics['R² Score']:.4f}",
               "Price accuracy (1.0 = perfect)"), unsafe_allow_html=True)
ks[3].markdown(kpi_html("RMSE ret%",  f"{metrics['RMSE ret%']:.3f}%", "Daily error"),
               unsafe_allow_html=True)
ks[4].markdown(kpi_html("RMSE $",     f"{ccy_sym}{metrics['RMSE $']:.2f}", "Display context"),
               unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
t_price, t_pred, t_fi, t_sent, t_wf, t_cmp, t_multi = st.tabs([
    "📈 Live Price", "🤖 AI Prediction", "📊 Feature Importance",
    "📰 Sentiment",  "🔄 Walk-Forward",  "📋 Compare",
    "🌐 Multi-Stock",
])


# ══════════ 📈 LIVE PRICE ════════════════════════════════════════════════════
with t_price:
    st.markdown("<div class='sec-title'>Live Price Chart</div>", unsafe_allow_html=True)
    st.plotly_chart(chart_price(df_raw, df_feat), use_container_width=True)

    st.markdown("<div class='sec-title'>Latest Indicator Snapshot</div>", unsafe_allow_html=True)
    snap_cols = [("RSI","RSI"),("MA10_r","MA10 ratio"),("MA20_r","MA20 ratio"),
                 ("MA50_r","MA50 ratio"),("BB_pos","BB Position"),("ATR_pct","ATR %"),
                 ("Vol_ratio","Vol ratio"),("MACD_n","MACD norm")]
    sc = st.columns(len(snap_cols))
    for col, (feat, label) in zip(sc, snap_cols):
        if feat in df_feat.columns:
            col.metric(label, f"{float(df_feat[feat].iloc[-1]):.3f}")


# ══════════ 🤖 AI PREDICTION ══════════════════════════════════════════════════
with t_pred:
    st.markdown("<div class='sec-title'>Test Set: Actual vs Predicted</div>",
                unsafe_allow_html=True)
    st.plotly_chart(chart_avp(test_idx, y_true, y_pred_p, active_model),
                    use_container_width=True)

    st.info("**Direction Accuracy** = primary metric. 50% = random. >55% = useful signal. "
            "**R² Score** = how closely predicted prices track actual prices. "
            "1.0 = perfect, 0.0 = no better than predicting the average, negative = worse than average.")

    # ── Next-Day Prediction ───────────────────────────────────────────────────
    st.markdown("<div class='sec-title'>🔮 Next-Day Prediction</div>", unsafe_allow_html=True)
    st.caption("Predict tomorrow's closing price using the latest available features.")

    if st.button("▶  Predict Tomorrow", key="next_day_btn"):
        try:
            last_feat = df_feat[FEATURE_COLS].iloc[[-1]].values.astype(np.float32)
            if active_model in DL_BUILDERS:
                keras_m, scaler_ = model_obj
                feat_sc  = scaler_.transform(df_feat[FEATURE_COLS].values.astype(np.float32))
                seq      = feat_sc[-SEQ_LEN:].reshape(1, SEQ_LEN, len(FEATURE_COLS))
                pred_pct = float(keras_m.predict(seq, verbose=0).flatten()[0])
            else:
                pred_pct = float(model_obj.predict(last_feat)[0])

            pred_price = close_now * (1 + pred_pct)
            direction  = "▲ UP" if pred_pct > 0 else "▼ DOWN"
            next_bday  = (df_feat.index[-1] + pd.tseries.offsets.BDay(1)).strftime("%d %b %Y")

            nd1, nd2, nd3 = st.columns(3)
            with nd1:
                st.markdown(kpi_html("Today's Close", f"{ccy_sym}{close_now:.2f}",
                                     "Last known price"), unsafe_allow_html=True)
            with nd2:
                st.markdown(kpi_html(f"Predicted Close ({next_bday})",
                                     f"{ccy_sym}{pred_price:.2f}",
                                     f"{direction}  ·  {pred_pct*100:+.3f}%",
                                     pred_pct > 0), unsafe_allow_html=True)
            with nd3:
                st.markdown(kpi_html("Signal", direction, active_model,
                                     pred_pct > 0), unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:0.68rem;color:#7D8590;margin-top:0.4rem;'>"
                f"{active_model} · {s_name} · NOT FINANCIAL ADVICE</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("<hr style='border-color:#30363D;margin:1.5rem 0'>", unsafe_allow_html=True)

    # ── Multi-Day Forecast ────────────────────────────────────────────────────
    st.markdown("<div class='sec-title'>📅 Multi-Day Forecast</div>", unsafe_allow_html=True)
    st.caption("Each predicted price feeds back into the feature window for the next step — "
               "no future data is used.")

    horizon_choice = st.radio("Forecast Horizon", [7, 14, 30],
                               format_func=lambda x: f"{x} Days", horizontal=True)

    if st.button("🔮  Run Forecast"):
        try:
            with st.spinner(f"Running {horizon_choice}-day forecast…"):
                forecast_results = multi_day_forecast(
                    model_obj, df_feat, active_model, horizon_choice, close_now)

            final_price  = forecast_results[-1]["price"]
            total_change = (final_price - close_now) / close_now * 100

            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.markdown(kpi_html("Current Price", f"{ccy_sym}{close_now:.2f}",
                                     "Today's close"), unsafe_allow_html=True)
            with fc2:
                st.markdown(kpi_html(f"Day {horizon_choice} Forecast",
                                     f"{ccy_sym}{final_price:.2f}",
                                     f"{'▲' if total_change >= 0 else '▼'} {abs(total_change):.2f}% total",
                                     total_change >= 0), unsafe_allow_html=True)
            with fc3:
                up_days   = sum(1 for r in forecast_results if r["pct_change"] >= 0)
                down_days = horizon_choice - up_days
                st.markdown(kpi_html("Up / Down Days", f"{up_days} ▲ / {down_days} ▼",
                                     "across forecast period"), unsafe_allow_html=True)

            st.plotly_chart(
                chart_forecast(forecast_results, df_feat, close_now, ccy_sym, s_ticker),
                use_container_width=True,
            )

            rows_to_show = forecast_results if horizon_choice <= 14 else forecast_results[::2]
            table_html = """
            <table style='width:100%;border-collapse:collapse;font-family:IBM Plex Mono;
                          font-size:0.8rem;margin-top:0.5rem;'>
              <thead><tr>
                <th style='color:#7D8590;font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;
                           border-bottom:1px solid #30363D;padding:0.4rem 0.8rem;text-align:left;'>Day</th>
                <th style='color:#7D8590;font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;
                           border-bottom:1px solid #30363D;padding:0.4rem 0.8rem;text-align:left;'>Date</th>
                <th style='color:#7D8590;font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;
                           border-bottom:1px solid #30363D;padding:0.4rem 0.8rem;text-align:right;'>Forecast Price</th>
                <th style='color:#7D8590;font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;
                           border-bottom:1px solid #30363D;padding:0.4rem 0.8rem;text-align:right;'>Day Change</th>
              </tr></thead><tbody>"""
            for r in rows_to_show:
                clr = "#00C896" if r["pct_change"] >= 0 else "#CC0000"
                table_html += (
                    f"<tr>"
                    f"<td style='padding:0.4rem 0.8rem;border-bottom:1px solid #1C2128;color:#7D8590;'>{r['day']}</td>"
                    f"<td style='padding:0.4rem 0.8rem;border-bottom:1px solid #1C2128;color:#E6EDF3;'>{r['date']}</td>"
                    f"<td style='padding:0.4rem 0.8rem;border-bottom:1px solid #1C2128;color:#E6EDF3;text-align:right;'>{ccy_sym}{r['price']:.2f}</td>"
                    f"<td style='padding:0.4rem 0.8rem;border-bottom:1px solid #1C2128;color:{clr};text-align:right;'>"
                    f"{r['direction']} {abs(r['pct_change']):.2f}%</td></tr>"
                )
            table_html += "</tbody></table>"

            if horizon_choice >= 30:
                st.caption("Showing every other day for readability (all 30 used in calculation).")
            st.markdown(table_html, unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:0.68rem;color:#7D8590;margin-top:0.6rem;'>"
                f"{active_model} · {s_name} · NOT FINANCIAL ADVICE</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Forecast failed: {e}")


# ══════════ 📊 FEATURE IMPORTANCE ════════════════════════════════════════════
with t_fi:
    st.markdown("<div class='sec-title'>Feature Importance</div>", unsafe_allow_html=True)
    if fi is not None:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(chart_fi(fi, active_model), use_container_width=True)
        with c2:
            st.markdown("**Top Predictors**")
            total = fi["Importance"].sum()
            for _, row in fi.head(8).iterrows():
                pct = row["Importance"] / total * 100
                st.markdown(
                    f"<div style='background:#161B22;border-left:3px solid #CC0000;"
                    f"padding:5px 9px;border-radius:3px;margin-bottom:4px;'>"
                    f"<span style='font-size:0.8rem;color:#E6EDF3;'>{row['Feature']}</span><br>"
                    f"<span style='font-family:IBM Plex Mono;font-size:0.68rem;color:#7D8590;'>"
                    f"{pct:.1f}%</span></div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info(f"Feature importance unavailable for **{active_model}**. "
                f"Switch to a classical model (Linear Regression, Random Forest, etc.).")


# ══════════ 📰 SENTIMENT ══════════════════════════════════════════════════════
with t_sent:
    st.markdown("<div class='sec-title'>AI Market Sentiment</div>", unsafe_allow_html=True)
    sigs  = generate_sentiment(df_feat)
    avg_s = np.mean([s["score"] for s in sigs])
    s_lbl = "BULLISH" if avg_s > 0.2 else ("BEARISH" if avg_s < -0.2 else "NEUTRAL")
    s_col = "#00C896" if avg_s > 0.2 else ("#CC0000" if avg_s < -0.2 else "#C9A84C")
    st.markdown(
        f"<div style='background:#161B22;border:1px solid #30363D;border-radius:5px;"
        f"padding:0.9rem 1.4rem;margin-bottom:0.9rem;'>"
        f"<span style='font-size:0.62rem;letter-spacing:0.2em;color:#7D8590;"
        f"text-transform:uppercase;'>Overall Signal</span><br>"
        f"<span style='font-family:IBM Plex Mono;font-size:1.9rem;font-weight:700;"
        f"color:{s_col};'>{s_lbl}</span>"
        f"<span style='font-size:0.72rem;color:#7D8590;margin-left:0.8rem;'>"
        f"composite: {avg_s:+.2f}</span></div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    for i, sig in enumerate(sigs):
        (c1 if i % 2 == 0 else c2).markdown(
            f"<div class='sent-card {sig['type']}'>{sig['text']}</div>",
            unsafe_allow_html=True,
        )


# ══════════ 🔄 WALK-FORWARD ══════════════════════════════════════════════════
with t_wf:
    st.markdown("<div class='sec-title'>Walk-Forward Validation</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#161B22;border:1px solid #30363D;border-radius:5px;
                padding:0.85rem 1.2rem;margin-bottom:1rem;font-size:0.82rem;
                line-height:1.7;color:#7D8590;'>
    <b style='color:#E6EDF3;'>Why Walk-Forward?</b><br>
    A single train/test split lets you accidentally tune on the test set (look-ahead bias).
    Walk-forward validation retrains the model on each expanding window and tests only on
    unseen future data — giving a truly unbiased performance estimate.<br><br>
    <b style='color:#E6EDF3;'>How it works:</b><br>
    &nbsp;Window 1 → Train: start → year N &nbsp;|&nbsp; Test: year N+1<br>
    &nbsp;Window 2 → Train: start → year N+1 &nbsp;|&nbsp; Test: year N+2 &nbsp;...<br>
    All test predictions are stitched into one continuous out-of-sample series.
    </div>
    """, unsafe_allow_html=True)

    wf            = st.session_state.get("wf_result")
    wf_model_used = st.session_state.get("wf_model", "")

    if wf is None:
        st.info("👈  Click **🔄 Walk-Forward** in the sidebar.\nLoad data first (**Load & Train**).")
    else:
        agg = wf["agg_metrics"]; pw = wf["per_window"]

        st.markdown(f"<span class='model-badge'>{wf_model_used} · Walk-Forward</span>",
                    unsafe_allow_html=True)
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        da_agg = agg["Dir Acc"]
        dal    = "Strong ✅" if da_agg > 0.55 else ("Decent" if da_agg > 0.52 else "Near random ⚠️")
        k1.markdown(kpi_html("Agg Dir Acc",   f"{da_agg*100:.1f}%", dal,
                             "green" if da_agg > 0.52 else ""), unsafe_allow_html=True)
        k2.markdown(kpi_html("Dir Acc Range",
                             f"{agg['Dir Acc Min']*100:.1f}–{agg['Dir Acc Max']*100:.1f}%",
                             "min–max across windows"), unsafe_allow_html=True)
        k3.markdown(kpi_html("Dir Acc Std",   f"{agg['Dir Acc Std']*100:.2f}%",
                             "stability (lower = more consistent)"), unsafe_allow_html=True)
        k4.markdown(kpi_html("R² Score",      f"{agg['R² Score']:.4f}",
                             "Price accuracy (1.0 = perfect)"), unsafe_allow_html=True)
        k5.markdown(kpi_html("RMSE ret%",     f"{agg['RMSE ret%']:.3f}%",
                             "Daily prediction error"), unsafe_allow_html=True)
        k6.markdown(kpi_html("N Windows",     str(agg["N Windows"]),
                             f"{agg['N Total']} total predictions"), unsafe_allow_html=True)

        st.info("**Aggregate Direction Accuracy** is across all stitched test predictions. "
                "**Std** = consistency — 55% in every window beats 60%/45% swinging.")
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<div class='sec-title'>Stitched Out-of-Sample Predictions</div>",
                    unsafe_allow_html=True)
        st.caption("Coloured bands = individual test windows. No window was used in its own training.")
        st.plotly_chart(chart_wf_overview(wf), use_container_width=True)

        col_l, col_r = st.columns([1.5, 1])
        with col_l:
            st.markdown("<div class='sec-title'>Direction Accuracy per Window</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(chart_wf_dir_acc(wf), use_container_width=True)
        with col_r:
            st.markdown("<div class='sec-title'>Per-Window Breakdown</div>",
                        unsafe_allow_html=True)
            pw_rows = []
            for p in pw:
                rating = "✅" if p["Dir Acc"] > 0.55 else ("~" if p["Dir Acc"] > 0.50 else "⚠️")
                pw_rows.append({
                    "Window":   p["Window"],
                    "Dir Acc":  f"{p['Dir Acc']*100:.1f}%  {rating}",
                    "R² Score": f"{p['R² Score']:.4f}",
                    "RMSE%":    f"{p['RMSE ret%']:.3f}%",
                    "N":        p["N Test"],
                })
            st.dataframe(pd.DataFrame(pw_rows), hide_index=True, use_container_width=True)

        st.markdown("<div class='sec-title'>Metric Stability across Windows</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(chart_wf_metric_trend(wf), use_container_width=True)

        st.markdown("<div class='sec-title'>Truly Out-of-Sample Equity Curve</div>",
                    unsafe_allow_html=True)
        st.caption("Built only from walk-forward test predictions — "
                   "each prediction was made by a model that had NEVER seen that date.")
        wf_fig, wf_bt = chart_wf_equity(wf)
        st.plotly_chart(wf_fig, use_container_width=True)

        b1, b2, b3, b4 = st.columns(4)
        b1.markdown(kpi_html("WF Strategy Return", f"{wf_bt['total_ret']:+.1f}%",
                             "walk-forward test set"), unsafe_allow_html=True)
        b2.markdown(kpi_html("Buy & Hold Return",  f"{wf_bt['bh_ret']:+.1f}%",
                             "same period"),         unsafe_allow_html=True)
        b3.markdown(kpi_html("WF Sharpe",          f"{wf_bt['sharpe']:.3f}",
                             "annualised"),          unsafe_allow_html=True)
        b4.markdown(kpi_html("WF Max Drawdown",    f"{wf_bt['max_dd']:.1f}%",
                             "peak-to-trough"),      unsafe_allow_html=True)

        alpha = wf_bt["total_ret"] - wf_bt["bh_ret"]
        dc    = "#00C896" if alpha > 0 else "#CC0000"
        st.markdown(
            f"<div style='background:#161B22;border:1px solid #30363D;border-radius:4px;"
            f"padding:0.75rem 1.1rem;font-family:IBM Plex Mono;font-size:0.82rem;"
            f"margin-top:0.5rem;'>Walk-Forward Alpha vs Buy-Hold: "
            f"<span style='color:{dc};font-size:1.05rem;'>{alpha:+.1f}%</span>"
            f"&nbsp;·&nbsp; Win Rate: {wf_bt['win_rate']:.1f}%"
            f"&nbsp;·&nbsp; {agg['N Windows']} independent test windows</div>",
            unsafe_allow_html=True,
        )

        if st.session_state["metrics"] is not None:
            st.markdown("<div class='sec-title'>Single-Split vs Walk-Forward Comparison</div>",
                        unsafe_allow_html=True)
            ss = st.session_state["metrics"]
            comp_data = {
                "Method":    ["Single Split (standard)", "Walk-Forward (honest)"],
                "Dir Acc":   [f"{ss['Dir Acc']*100:.1f}%", f"{agg['Dir Acc']*100:.1f}%"],
                "R² Score":  [f"{ss['R² Score']:.4f}",     f"{agg['R² Score']:.4f}"],
                "RMSE ret%": [f"{ss['RMSE ret%']:.3f}%",    f"{agg['RMSE ret%']:.3f}%"],
                "Bias Risk": ["High (only 1 split)",        "Low (multiple windows)"],
            }
            st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)
            st.caption("Walk-forward metrics are typically lower than single-split — "
                       "that is expected and correct.")


# ══════════ 📋 COMPARE ════════════════════════════════════════════════════════
with t_cmp:
    st.markdown("<div class='sec-title'>Model Comparison</div>", unsafe_allow_html=True)
    st.caption("Trains all classical models on the same split with honest financial metrics.")

    cmp_mode = st.radio(
        "Comparison Mode",
        ["Single Split (fast)", "Walk-Forward (professional)"],
        horizontal=True,
        help="Walk-Forward trains each model on multiple expanding windows for unbiased comparison.",
    )

    run_cmp = compare_btn or st.button("⚖️  Run Comparison", key="cmp_btn")
    if run_cmp:
        c_names = ["Linear Regression", "Random Forest"]
        if HAS_XGB: c_names.append("XGBoost")
        if HAS_LGB: c_names.append("LightGBM")

        results = []
        prog    = st.progress(0, text="Comparing…")

        if cmp_mode == "Single Split (fast)":
            for i, mname in enumerate(c_names):
                with st.spinner(f"Training {mname}…"):
                    _, m, _, _, _, _, _ = train_classical(df_feat, mname, test_pct)
                    results.append({"Model": mname, "Method": "Single Split", **m})
                prog.progress((i + 1) / len(c_names))
        else:
            total_steps = len(c_names)
            for i, mname in enumerate(c_names):
                with st.spinner(f"Walk-forward: {mname}…"):
                    def _pcb(frac, txt, _i=i):
                        prog.progress((_i + frac) / total_steps, text=txt)
                    try:
                        wf_c = run_walk_forward(df_feat, mname, wf_min_train, wf_test_mo, _pcb)
                        m    = wf_c["agg_metrics"]
                        results.append({
                            "Model": mname, "Method": "Walk-Forward",
                            "Dir Acc": m["Dir Acc"], "R² Score": m["R² Score"],
                            "RMSE ret%": m["RMSE ret%"], "RMSE $": m["RMSE $"],
                            "Dir Acc Std": m["Dir Acc Std"], "N Windows": m["N Windows"],
                        })
                    except Exception as ex:
                        st.warning(f"{mname} failed: {ex}")

        prog.empty()

        if not results:
            st.error("No models completed successfully.")
        else:
            comp_df = (pd.DataFrame(results)
                       .sort_values("Dir Acc", ascending=False)
                       .reset_index(drop=True))
            comp_df.index += 1

            hl_max = ["Dir Acc", "R² Score"]
            hl_min = ["RMSE ret%", "RMSE $"] + (["Dir Acc Std"] if "Dir Acc Std" in comp_df.columns else [])

            st.dataframe(
                comp_df.style
                    .highlight_max(subset=hl_max, color="#1a3d2b")
                    .highlight_min(subset=hl_min,  color="#1a3d2b"),
                use_container_width=True,
            )

            fig_cmp = go.Figure(go.Bar(
                x=comp_df["Model"], y=comp_df["Dir Acc"] * 100,
                text=comp_df["Dir Acc"].map(lambda v: f"{v*100:.1f}%"),
                textposition="outside",
                marker=dict(color=comp_df["Dir Acc"],
                            colorscale=[[0,"#1C2128"],[1,"#CC0000"]], showscale=False),
            ))
            if "Dir Acc Std" in comp_df.columns:
                fig_cmp.update_traces(
                    error_y=dict(type="data",
                                 array=(comp_df["Dir Acc Std"]*100).tolist(),
                                 visible=True, color="#C9A84C"))
            fig_cmp.add_hline(y=50, line_dash="dot", line_color="#C9A84C",
                              annotation_text="50% = random",
                              annotation_font_color="#C9A84C")
            apply_layout(fig_cmp, height=360,
                         title=f"Direction Accuracy by Model ({cmp_mode})")
            fig_cmp.update_yaxes(range=[40, 70], ticksuffix="%")
            st.plotly_chart(fig_cmp, use_container_width=True)

            best   = comp_df.iloc[0]
            da_p   = best["Dir Acc"] * 100
            rating = "Strong ✅" if da_p > 55 else ("Decent" if da_p > 52 else "Near random ⚠️")
            st.success(f"Best: **{best['Model']}** · Dir Accuracy {da_p:.1f}% ({rating})")
    else:
        st.info("Click **Compare All** in the sidebar or the button above.")


# ══════════ 🌐 MULTI-STOCK ════════════════════════════════════════════════════
with t_multi:
    st.markdown("<div class='sec-title'>Multi-Stock Overview</div>", unsafe_allow_html=True)
    st.caption("Live snapshot of all four stocks. Select a stock in the sidebar and click "
               "**Load & Train** to run full AI prediction on it.")

    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_multi_snapshot(tickers: tuple, period: str = "3mo") -> dict:
        out = {}
        for tkr in tickers:
            try:
                df = yf.download(tkr, period=period, interval="1d",
                                 auto_adjust=True, progress=False)
                if df.empty: continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index)
                c = df["Close"].squeeze()
                out[tkr] = {
                    "df":    df,
                    "close": float(c.iloc[-1]),
                    "prev":  float(c.iloc[-2]) if len(c) > 1 else float(c.iloc[-1]),
                    "rsi":   float(compute_rsi(c).iloc[-1]),
                    "mom30": float((c.iloc[-1]/c.iloc[-30]-1)*100) if len(c) >= 30 else 0.0,
                }
            except Exception:
                pass
        return out

    all_tickers = tuple(v["ticker"] for v in STOCKS.values())
    with st.spinner("Fetching live data for all stocks…"):
        msnap = fetch_multi_snapshot(all_tickers)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    cols = st.columns(len(STOCKS))
    for col, (label, info) in zip(cols, STOCKS.items()):
        tkr  = info["ticker"]
        data = msnap.get(tkr)
        with col:
            if data is None:
                st.markdown(
                    f"<div class='kpi'><div class='kpi-label'>{info['emoji']} {tkr}</div>"
                    f"<div class='kpi-value' style='font-size:0.9rem;color:#7D8590;'>Unavailable</div></div>",
                    unsafe_allow_html=True)
                continue
            chg_  = data["close"] - data["prev"]
            chgp_ = chg_ / data["prev"] * 100
            ccy_  = "₹" if info["currency"] == "INR" else "$"
            clr_  = "#00C896" if chg_ >= 0 else "#CC0000"
            rsi_  = data["rsi"]
            rsi_c = "#CC0000" if rsi_ > 70 else ("#00C896" if rsi_ < 30 else "#C9A84C")
            mom_  = data["mom30"]
            mom_c = "#00C896" if mom_ >= 0 else "#CC0000"
            st.markdown(f"""
            <div style='background:#161B22;border:1px solid #30363D;border-top:3px solid {info["color"]};
                        border-radius:5px;padding:1rem;text-align:center;margin-bottom:0.5rem;'>
              <div style='font-size:1.4rem;'>{info["emoji"]}</div>
              <div style='font-family:IBM Plex Mono;font-size:1rem;font-weight:700;color:#E6EDF3;'>{tkr}</div>
              <div style='font-size:0.62rem;color:#7D8590;margin-bottom:0.5rem;'>{info["name"]}</div>
              <div style='font-family:IBM Plex Mono;font-size:1.5rem;color:#E6EDF3;'>{ccy_}{data["close"]:,.2f}</div>
              <div style='font-family:IBM Plex Mono;font-size:0.82rem;color:{clr_};margin-top:0.2rem;'>
                  {"▲" if chg_ >= 0 else "▼"} {ccy_}{abs(chg_):.2f} ({chgp_:+.2f}%)</div>
              <div style='display:flex;justify-content:center;gap:1.2rem;margin-top:0.6rem;
                          font-family:IBM Plex Mono;font-size:0.72rem;'>
                <span>RSI <span style='color:{rsi_c};'>{rsi_:.1f}</span></span>
                <span>30d <span style='color:{mom_c};'>{mom_:+.1f}%</span></span>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Normalised chart ──────────────────────────────────────────────────────
    st.markdown("<div class='sec-title'>Normalised Price Performance (base = 100)</div>",
                unsafe_allow_html=True)
    st.caption("All prices rebased to 100 — shows relative return, not absolute price.")
    fig_multi = go.Figure()
    for label, info in STOCKS.items():
        data = msnap.get(info["ticker"])
        if data is None: continue
        c    = data["df"]["Close"].squeeze()
        base = float(c.iloc[0])
        if base == 0: continue
        fig_multi.add_trace(go.Scatter(x=c.index, y=c/base*100,
                                        name=f"{info['emoji']} {info['ticker']}",
                                        line=dict(color=info["color"], width=2)))
    apply_layout(fig_multi, height=420,
                 title="Relative Price Performance — RACE · AAPL · TSLA · RELIANCE.NS")
    fig_multi.update_yaxes(title_text="Indexed (base=100)")
    st.plotly_chart(fig_multi, use_container_width=True)

    # ── RSI comparison ────────────────────────────────────────────────────────
    st.markdown("<div class='sec-title'>Current RSI Comparison</div>", unsafe_allow_html=True)
    rsi_data = {info["ticker"]: msnap[info["ticker"]]["rsi"]
                for _, info in STOCKS.items() if info["ticker"] in msnap}
    if rsi_data:
        rsi_cols = ["#CC0000" if v > 70 else ("#00C896" if v < 30 else "#C9A84C")
                    for v in rsi_data.values()]
        fig_rsi = go.Figure(go.Bar(x=list(rsi_data.keys()), y=list(rsi_data.values()),
                                    marker_color=rsi_cols,
                                    text=[f"{v:.1f}" for v in rsi_data.values()],
                                    textposition="outside"))
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="#CC0000",
                          annotation_text="Overbought 70", annotation_font_color="#CC0000")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="#00C896",
                          annotation_text="Oversold 30",   annotation_font_color="#00C896")
        apply_layout(fig_rsi, height=300, title="RSI (14) — Current Reading")
        fig_rsi.update_yaxes(range=[0, 100])
        st.plotly_chart(fig_rsi, use_container_width=True)

    # ── Leaderboard ───────────────────────────────────────────────────────────
    st.markdown("<div class='sec-title'>30-Day Momentum Leaderboard</div>",
                unsafe_allow_html=True)
    mom_rows = []
    for _, info in STOCKS.items():
        data = msnap.get(info["ticker"])
        if data is None: continue
        ccy_ = "₹" if info["currency"] == "INR" else "$"
        mom_rows.append({
            "Stock":      f"{info['emoji']}  {info['ticker']}",
            "Name":       info["name"],
            "Last Close": f"{ccy_}{data['close']:,.2f}",
            "Day Change": f"{(data['close']-data['prev'])/data['prev']*100:+.2f}%",
            "30D Return": f"{data['mom30']:+.2f}%",
            "RSI (14)":   f"{data['rsi']:.1f}",
            "Exchange":   info["exchange"],
        })
    if mom_rows:
        mom_df = (pd.DataFrame(mom_rows)
                  .sort_values("30D Return",
                               key=lambda s: s.str.replace("%","").astype(float),
                               ascending=False))
        st.dataframe(mom_df, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#30363D;margin-top:2rem;'>
<div style='text-align:center;color:#7D8590;font-size:0.65rem;
            letter-spacing:0.12em;padding:0.4rem 0 1.2rem;'>
    AI STOCK PRICE PREDICTION · RACE · AAPL · TSLA · RELIANCE.NS &nbsp;|&nbsp;
    Streamlit · yfinance · TensorFlow/Keras · scikit-learn · XGBoost · LightGBM · Plotly
    &nbsp;|&nbsp; <span style='color:#CC0000;'>NOT FINANCIAL ADVICE</span>
</div>
""", unsafe_allow_html=True)

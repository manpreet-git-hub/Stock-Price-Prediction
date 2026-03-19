"""
config.py
─────────
Global constants, stock registry, and Plotly layout helpers.
Imported by every other module — no Streamlit dependency here.
"""

# ─────────────────────────────────────────────────────────────────────────────
# STOCK REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
STOCKS = {
    "Ferrari  (RACE)": {
        "ticker":   "RACE",
        "name":     "Ferrari N.V.",
        "exchange": "NASDAQ",
        "currency": "USD",
        "emoji":    "🏎️",
        "color":    "#CC0000",
    },
    "Apple  (AAPL)": {
        "ticker":   "AAPL",
        "name":     "Apple Inc.",
        "exchange": "NASDAQ",
        "currency": "USD",
        "emoji":    "🍎",
        "color":    "#A2AAAD",
    },
    "Tesla  (TSLA)": {
        "ticker":   "TSLA",
        "name":     "Tesla Inc.",
        "exchange": "NASDAQ",
        "currency": "USD",
        "emoji":    "⚡",
        "color":    "#E31937",
    },
    "Reliance  (RELIANCE.NS)": {
        "ticker":   "RELIANCE.NS",
        "name":     "Reliance Industries",
        "exchange": "NSE",
        "currency": "INR",
        "emoji":    "🇮🇳",
        "color":    "#1E90FF",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL / SEQUENCE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LEN = 30          # lookback window for deep-learning sequences

FEATURE_COLS = [
    "MA10_r", "MA20_r", "MA50_r",
    "EMA10_r", "EMA20_r",
    "RSI",
    "MACD_n", "MACD_sig_n", "MACD_hist_n",
    "BB_width_pct", "BB_pos",
    "ATR_pct",
    "Vol_ratio",
    "Ret1", "Ret2", "Ret3", "Ret5", "Ret10",
]

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY LAYOUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(8,11,15,0.95)",
    font=dict(family="Inter", color="#7D8590", size=11),
    legend=dict(bgcolor="rgba(13,17,23,0.9)", bordercolor="#30363D", borderwidth=1),
    margin=dict(l=50, r=20, t=36, b=40),
    hoverlabel=dict(bgcolor="#161B22", bordercolor="#30363D", font_color="#E6EDF3"),
)
_GRID = dict(gridcolor="#1C2128", zerolinecolor="#1C2128")


def apply_layout(fig, **kw):
    """Apply standard dark theme + grid to any Plotly figure."""
    fig.update_layout(**_PL, **kw)
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig

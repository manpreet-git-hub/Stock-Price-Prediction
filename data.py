"""
data.py
───────
Data loading (yfinance) and feature engineering.
All features are scale-invariant ratios / return-based — no raw prices.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from config import FEATURE_COLS


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_data(start: str, end: str, interval: str, ticker: str = "RACE") -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance and return a clean DataFrame."""
    df = yf.download(ticker, start=start, end=end, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATOR HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def compute_rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    d = s.diff()
    g = d.clip(lower=0).ewm(com=n - 1, min_periods=n).mean()
    l = (-d.clip(upper=0)).ewm(com=n - 1, min_periods=n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def compute_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range (ATR)."""
    hi, lo, cl = df["High"].squeeze(), df["Low"].squeeze(), df["Close"].squeeze()
    tr = pd.concat([(hi - lo), (hi - cl.shift()).abs(),
                    (lo - cl.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(com=n - 1, min_periods=n).mean()


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators and return an enriched DataFrame.

    Design principles:
    - ALL ML features are ratios or % returns (scale-invariant across time).
    - Chart-only columns are prefixed with '_' (e.g. _MA20, _BB_upper).
    - Target_pct = next-day % return (what the model predicts).
    """
    df  = df.copy()
    c   = df["Close"].squeeze()
    vol = df["Volume"].squeeze() if "Volume" in df.columns else pd.Series(1.0, index=df.index)

    # ── Moving average ratios (price / MA — ratio > 1 means price above MA) ──
    for n, col in [(10, "MA10_r"), (20, "MA20_r"), (50, "MA50_r")]:
        df[col] = c / c.rolling(n).mean()
    for n, col in [(10, "EMA10_r"), (20, "EMA20_r")]:
        df[col] = c / c.ewm(span=n, adjust=False).mean()

    # Chart-only raw MA values (prefixed with _)
    df["_MA20"] = c.rolling(20).mean()
    df["_MA50"] = c.rolling(50).mean()

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["RSI"] = compute_rsi(c)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    df["MACD_n"]      = macd / c          # normalised by price
    df["MACD_sig_n"]  = sig  / c
    df["MACD_hist_n"] = (macd - sig) / c

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_lo  = bb_mid - 2 * bb_std
    df["BB_width_pct"] = (bb_up - bb_lo) / c   # band width as % of price
    df["BB_pos"]       = (c - bb_lo) / (bb_up - bb_lo + 1e-9)   # 0–1 position
    df["_BB_upper"]    = bb_up     # chart only
    df["_BB_lower"]    = bb_lo     # chart only

    # ── Volatility & Volume ───────────────────────────────────────────────────
    df["ATR_pct"]   = compute_atr(df) / c
    df["Vol_ratio"] = vol / vol.rolling(20).mean().replace(0, np.nan)

    # ── Return-based lag features (no raw price — prevents data leakage) ──────
    pct = c.pct_change()
    for lag in [1, 2, 3, 5, 10]:
        df[f"Ret{lag}"] = pct.shift(lag)

    # ── Target: next-day % return ─────────────────────────────────────────────
    df["Target_pct"] = c.pct_change().shift(-1)

    return df.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE BUILDER  (for deep learning models)
# ─────────────────────────────────────────────────────────────────────────────
def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """
    Sliding window: converts a flat 2-D feature array into
    (samples, seq_len, features) format required by LSTM / TCN / Transformer.
    """
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

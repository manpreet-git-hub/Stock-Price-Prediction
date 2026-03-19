"""
models.py
─────────
ML models (Linear Regression, Random Forest, XGBoost, LightGBM),
Deep Learning models (LSTM, Temporal CNN, Transformer),
evaluation metrics, and training functions.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    tf.get_logger().setLevel("ERROR")
    HAS_TF = True
except ImportError:
    HAS_TF = False
    keras = layers = callbacks = tf = None

from config import FEATURE_COLS, SEQ_LEN
from data import build_sequences


# ─────────────────────────────────────────────────────────────────────────────
# DEEP LEARNING MODEL BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def build_lstm(n_feat: int, seq_len: int):
    """
    Stacked LSTM — captures sequential momentum and mean-reversion patterns.
    Layer 1 returns the full sequence; Layer 2 summarises into a hidden state.
    Dropout prevents memorising training returns.
    """
    inp = keras.Input(shape=(seq_len, n_feat))
    x   = layers.LSTM(64, return_sequences=True, dropout=0.2)(inp)
    x   = layers.LSTM(32, dropout=0.2)(x)
    x   = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1)(x)
    m   = keras.Model(inp, out, name="LSTM")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def build_tcn(n_feat: int, seq_len: int):
    """
    Temporal CNN with dilated causal convolutions.
    Dilation rates [1, 2, 4, 8] give a receptive field covering all 30 timesteps.
    Faster to train than LSTM; strong on periodic / cyclical signals.
    """
    inp = keras.Input(shape=(seq_len, n_feat))
    x   = inp
    for d in [1, 2, 4, 8]:
        x = layers.Conv1D(32, kernel_size=3, padding="causal",
                          dilation_rate=d, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1)(x)
    m   = keras.Model(inp, out, name="TemporalCNN")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def build_transformer(n_feat: int, seq_len: int):
    """
    Transformer encoder — multi-head self-attention lets each timestep attend
    to ALL others. Ideal for detecting structural regime shifts in the series.
    """
    d_model = 32
    inp     = keras.Input(shape=(seq_len, n_feat))
    x       = layers.Dense(d_model)(inp)

    if HAS_TF:
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_emb   = layers.Embedding(seq_len, d_model)(positions)
        x = x + pos_emb

    attn = layers.MultiHeadAttention(num_heads=4, key_dim=d_model // 4)(x, x)
    x    = layers.LayerNormalization()(x + attn)
    ff   = layers.Dense(64, activation="relu")(x)
    ff   = layers.Dense(d_model)(ff)
    x    = layers.LayerNormalization()(x + ff)
    x    = layers.GlobalAveragePooling1D()(x)
    x    = layers.Dense(16, activation="relu")(x)
    out  = layers.Dense(1)(x)
    m    = keras.Model(inp, out, name="Transformer")
    m.compile(optimizer=keras.optimizers.Adam(5e-4), loss="mse")
    return m


DL_BUILDERS: dict = {}
if HAS_TF:
    DL_BUILDERS = {
        "LSTM":         build_lstm,
        "Temporal CNN": build_tcn,
        "Transformer":  build_transformer,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLASSICAL ML MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def get_classical_model(name: str):
    """Return a sklearn Pipeline for the given model name."""
    if name == "Linear Regression":
        return Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
    if name == "Random Forest":
        return Pipeline([("m", RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            max_features=0.7, random_state=42, n_jobs=-1))])
    if name == "XGBoost" and HAS_XGB:
        return Pipeline([("m", XGBRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            random_state=42, verbosity=0))])
    if name == "LightGBM" and HAS_LGB:
        return Pipeline([("m", LGBMRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=4,
            num_leaves=15, min_child_samples=10,
            random_state=42, verbose=-1, n_jobs=-1))])
    raise ValueError(f"Unknown model: {name}")


def available_models() -> list[str]:
    """Return list of all available model names (depends on installed packages)."""
    base = ["Linear Regression", "Random Forest"]
    if HAS_XGB: base.append("XGBoost")
    if HAS_LGB: base.append("LightGBM")
    if HAS_TF:  base += list(DL_BUILDERS.keys())
    return base


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
def honest_metrics(y_true_pct, y_pred_pct, prev_close) -> dict:
    """
    Finance-honest model evaluation.

    Direction Accuracy — primary metric (50% = coin flip, >55% = useful signal).
    R² Score           — computed on reconstructed prices (intuitive 0–1 range).
    RMSE ret%          — prediction error in daily return % units.
    RMSE $             — price-space RMSE for display context.
    """
    dir_acc    = float(((y_true_pct > 0) == (y_pred_pct > 0)).mean())
    rmse_ret   = float(np.sqrt(mean_squared_error(y_true_pct, y_pred_pct)))
    y_tp       = prev_close * (1 + y_true_pct)
    y_pp       = prev_close * (1 + y_pred_pct)
    r2_price   = float(r2_score(y_tp, y_pp))
    rmse_price = float(np.sqrt(mean_squared_error(y_tp, y_pp)))
    return {
        "Dir Acc":   round(dir_acc,         4),
        "R² Score":  round(r2_price,        4),
        "RMSE ret%": round(rmse_ret * 100,  4),
        "RMSE $":    round(rmse_price,      2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING — CLASSICAL ML
# ─────────────────────────────────────────────────────────────────────────────
def train_classical(df_feat: pd.DataFrame, model_name: str,
                    test_size: float = 0.20):
    """
    Train a classical ML model on a strict time-series split.
    Returns: (model, metrics, y_true_price, y_pred_price, y_pred_pct, test_idx, fi)
    """
    X     = df_feat[FEATURE_COLS].values.astype(np.float32)
    y_pct = df_feat["Target_pct"].values.astype(np.float32)
    close = df_feat["Close"].squeeze().values.astype(np.float32)
    idx   = df_feat.index

    split      = int(len(X) * (1 - test_size))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y_pct[:split], y_pct[split:]
    prev_cl_te = close[split - 1:-1]
    test_idx   = idx[split:]

    model  = get_classical_model(model_name)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te).astype(np.float32)

    metrics  = honest_metrics(y_te, y_pred, prev_cl_te)
    y_pred_p = prev_cl_te * (1 + y_pred)
    y_true_p = prev_cl_te * (1 + y_te)

    # Feature importance
    m_inner = model.named_steps.get("m", model.steps[-1][1])
    if hasattr(m_inner, "feature_importances_"):
        fi = pd.DataFrame({"Feature": FEATURE_COLS,
                           "Importance": m_inner.feature_importances_}
                         ).sort_values("Importance", ascending=False)
    elif hasattr(m_inner, "coef_"):
        fi = pd.DataFrame({"Feature": FEATURE_COLS,
                           "Importance": np.abs(m_inner.coef_)}
                         ).sort_values("Importance", ascending=False)
    else:
        fi = None

    return model, metrics, y_true_p, y_pred_p, y_pred, test_idx, fi


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING — DEEP LEARNING
# ─────────────────────────────────────────────────────────────────────────────
def train_deep(df_feat: pd.DataFrame, model_name: str,
               test_size: float = 0.20, epochs: int = 40, batch: int = 32):
    """
    Train an LSTM / TCN / Transformer on scaled sequences.
    Returns: ((keras_model, scaler), metrics, y_true_price, y_pred_price, y_pred_pct, test_idx, None)
    """
    scaler = MinMaxScaler()
    X_sc   = scaler.fit_transform(df_feat[FEATURE_COLS].values.astype(np.float32))
    y_pct  = df_feat["Target_pct"].values.astype(np.float32)
    close  = df_feat["Close"].squeeze().values.astype(np.float32)
    idx    = df_feat.index

    split        = int(len(X_sc) * (1 - test_size))
    Xtr_r, Xte_r = X_sc[:split], X_sc[split:]
    ytr_r, yte_r = y_pct[:split], y_pct[split:]

    Xtr, ytr = build_sequences(Xtr_r, ytr_r, SEQ_LEN)
    Xte, yte = build_sequences(Xte_r, yte_r, SEQ_LEN)

    builder = DL_BUILDERS[model_name]
    model   = builder(len(FEATURE_COLS), SEQ_LEN)

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=4, min_lr=1e-5),
    ]
    model.fit(Xtr, ytr, validation_split=0.15, epochs=epochs,
              batch_size=batch, callbacks=cb, verbose=0)

    y_pred     = model.predict(Xte, verbose=0).flatten()
    prev_cl_te = close[split + SEQ_LEN - 1:-1]
    test_idx   = idx[split + SEQ_LEN:]

    metrics  = honest_metrics(yte, y_pred, prev_cl_te)
    y_pred_p = prev_cl_te * (1 + y_pred)
    y_true_p = prev_cl_te * (1 + yte)

    return (model, scaler), metrics, y_true_p, y_pred_p, y_pred, test_idx, None

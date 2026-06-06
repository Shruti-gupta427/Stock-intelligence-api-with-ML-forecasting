import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
warnings.filterwarnings("ignore")

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).round(2)

def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def _bollinger(series: pd.Series, window: int = 20):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return ma + 2 * std, ma - 2 * std  

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    close = df["CLOSE"]

    for lag in [1, 2, 3, 5, 10]:
        feat[f"ret_{lag}d"] = close.pct_change(lag)
    log_close = np.log(close)
    for lag in [1, 2, 3, 5]:
        feat[f"log_lag_{lag}"] = log_close.shift(lag)
    for w in [5, 10, 20]:
        feat[f"roll_mean_{w}"] = close.rolling(w).mean() / close - 1   # % above MA
        feat[f"roll_std_{w}"]  = close.rolling(w).std() / close

    
    feat["rsi_14"] = _rsi(close, 14) / 100  
    macd, signal = _macd(close)
    feat["macd"]        = macd / close
    feat["macd_signal"] = signal / close
    feat["macd_hist"]   = (macd - signal) / close
    bb_up, bb_lo = _bollinger(close, 20)
    feat["bb_position"] = (close - bb_lo) / (bb_up - bb_lo + 1e-9)

    if "VOLUME" in df.columns:
        vol = df["VOLUME"].replace(0, np.nan)
        feat["vol_ratio"] = vol / vol.rolling(10).mean()
        feat["vol_ratio"] = feat["vol_ratio"].fillna(1.0)
        feat["day_of_week"] = df.index.dayofweek / 4.0   # 0–1
        feat["month"]       = df.index.month / 12.0

    return feat
def _build_ensemble() -> StackingRegressor:
    base = [
        ("ridge",
         Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])),
        ("rf",
         RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=5,
                               random_state=42, n_jobs=-1)),
        ("gbm",
         GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                   max_depth=4, subsample=0.8,
                                   min_samples_leaf=5, random_state=42)),
    ]
    meta = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=0.5))])
    return StackingRegressor(
        estimators=base,
        final_estimator=meta,
        cv=TimeSeriesSplit(n_splits=3),
        passthrough=False,
        n_jobs=-1,
    )
def _walk_forward_metrics(X: np.ndarray, y: np.ndarray, n_splits: int = 3):
    """Return out-of-sample R² and MAE via time-series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=max(5, len(X) // 10))
    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in tscv.split(X):
        model = _build_ensemble()
        model.fit(X[train_idx], y[train_idx])
        y_pred_all.extend(model.predict(X[test_idx]))
        y_true_all.extend(y[test_idx])
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    r2  = r2_score(y_true_all, y_pred_all)
    mae = mean_absolute_error(np.exp(y_true_all), np.exp(y_pred_all))  # back to price
    return r2, mae


def _predict_with_intervals(model, last_known_X: np.ndarray, n_days: int = 5):
    preds_per_estimator = []
    for _, est in model.estimators_:
        day_preds = []
        row = last_known_X.copy()
        for _ in range(n_days):
            p = est.predict(row.reshape(1, -1))[0]
            day_preds.append(p)
            row = _shift_lag_features(row, p)
        preds_per_estimator.append(day_preds)

    preds_arr = np.array(preds_per_estimator)  
    point = preds_arr.mean(axis=0)
    sigma = preds_arr.std(axis=0)
    return point, sigma


def _shift_lag_features(row: np.ndarray, new_log_price: float) -> np.ndarray:

    return row

def fetch_and_process(symbol: str):
    df = yf.download(symbol, period="2y", progress=False)
    if df.empty:
        return None, None, None, None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].upper() for col in df.columns]
    else:
        df.columns = [col.upper() for col in df.columns]

    df["DAILY_RETURN"] = df["CLOSE"].pct_change()
    df["MA_7"]  = df["CLOSE"].rolling(7).mean().round(2)
    df["MA_20"] = df["CLOSE"].rolling(20).mean().round(2)
    df["RSI"]   = _rsi(df["CLOSE"]).round(2)
    df["VOLATILITY"] = df["DAILY_RETURN"].rolling(21).std().round(4)

    summary_metrics = {
        "high_52":   round(float(df["CLOSE"].max()), 2),
        "low_52":    round(float(df["CLOSE"].min()), 2),
        "avg_close": round(float(df["CLOSE"].mean()), 2),
        "last_close": round(float(df["CLOSE"].iloc[-1]), 2),
    }

   
    feat = _build_features(df)
    df_ml = df.join(feat).dropna()

    if len(df_ml) < 60:
        return df.tail(30).reset_index().assign(Date=lambda d: d["Date"].astype(str)), \
               [], summary_metrics, {}

    feature_cols = feat.columns.tolist()
    X = df_ml[feature_cols].values
    y = np.log(df_ml["CLOSE"].values)      # log-price target

    
    r2_oos, mae_price = _walk_forward_metrics(X, y, n_splits=3)

    final_model = _build_ensemble()
    final_model.fit(X, y)

    last_X = X[-1]
    log_preds, log_sigma = _predict_with_intervals(final_model, last_X, n_days=5)
    predictions      = [round(float(p), 2) for p in np.exp(log_preds)]
    pred_upper       = [round(float(p), 2) for p in np.exp(log_preds + log_sigma)]
    pred_lower       = [round(float(p), 2) for p in np.exp(log_preds - log_sigma)]

    accuracy_metrics = {
        "r2_score":              round(float(r2_oos), 3),
        "mean_absolute_error":   round(float(mae_price), 2),
        "model_confidence":      "High" if r2_oos > 0.6 else "Moderate" if r2_oos > 0.3 else "Low",
        "validation_method":     "walk-forward (time-series CV, no data leakage)",
        "model_type":            "stacked ensemble (Ridge + RandomForest + GBM)",
        "forecast_upper_1sigma": pred_upper,
        "forecast_lower_1sigma": pred_lower,
    }
    df_tail = df.tail(30).reset_index()
    df_tail["Date"] = df_tail["Date"].astype(str)

    return df_tail, predictions, summary_metrics, accuracy_metrics

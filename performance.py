import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

def fetch_and_process(symbol: str):
    #  data Collection
    df = yf.download(symbol, period="1y")
    if df.empty:
        return None, None, None, None
    
   
    df.columns = [col.upper() for col in df.columns]

    df['DAILY_RETURN'] = df['CLOSE'].pct_change()
    df['MA_7'] = df['CLOSE'].rolling(window=7).mean().round(2)
    df['MA_20'] = df['CLOSE'].rolling(window=20).mean().round(2)
    
    # technical indicators (RSI)
    delta = df['CLOSE'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = (100 - (100 / (1 + rs))).round(2)

    # volatility Score
    df['VOLATILITY'] = df['DAILY_RETURN'].rolling(window=21).std().round(4)
    
    # summary
    summary_metrics = {
        "high_52": round(float(df['CLOSE'].max()), 2),
        "low_52": round(float(df['CLOSE'].min()), 2),
        "avg_close": round(float(df['CLOSE'].mean()), 2),
        "last_close": round(float(df['CLOSE'].iloc[-1]), 2)
    }

    # ml predictor
    df_ml = df.dropna()
    if not df_ml.empty:
        X = np.arange(len(df_ml)).reshape(-1, 1)
        y = df_ml['CLOSE'].values
        
        model = LinearRegression().fit(X, y)
        y_pred_historical = model.predict(X)
        r2 = r2_score(y, y_pred_historical)
        mae = mean_absolute_error(y, y_pred_historical)
        future_index = np.array([len(df_ml) + i for i in range(1, 6)]).reshape(-1, 1)
        predictions = model.predict(future_index).flatten().tolist()
        predictions = [round(p, 2) for p in predictions] # Round for cleanliness
        
        accuracy_metrics = {
            "r2_score": round(float(r2), 3),
            "mean_absolute_error": round(float(mae), 2),
            "model_confidence": "High" if r2 > 0.8 else "Moderate" if r2 > 0.5 else "Low"
        }
    else:
        predictions, accuracy_metrics = [], {}
    return df.tail(30).reset_index(), predictions, summary_metrics, accuracy_metrics
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.concurrency import run_in_threadpool
from performance import fetch_and_process
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import io
import logging
import asyncio
import threading
import redis
import pickle
import os



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Intelligence API")


COMPANY_MAP = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation",
    "TSLA": "Tesla, Inc.", "GOOGL": "Alphabet Inc. (Google)",
    "TCS.NS": "Tata Consultancy Services Limited",
    "INFY.NS": "Infosys Limited", "RELIANCE.NS": "Reliance Industries Limited"
}

cache = {}
CACHE_EXPIRY = 3600 


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (float, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)  
    return obj

_cache_lock = threading.Lock()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_EXPIRY = 3600

try:
    r = redis.from_url(REDIS_URL, decode_responses=False)
    r.ping()
    logger.info("Redis connected.")
except Exception as e:
    logger.warning(f"Redis unavailable, falling back to in-memory cache: {e}")
    r = None

_memory_cache = {}
_cache_lock = threading.Lock()

def get_data_with_cache(symbol: str):
    symbol = symbol.upper()
    cache_key = f"stock:{symbol}"

    # try redis 
    if r:
        try:
            cached = r.get(cache_key)
            if cached:
                logger.info(f"Redis HIT: {symbol}")
                return pickle.loads(cached)
        except Exception as e:
            logger.warning(f"Redis read failed: {e}")

    # if miss
    else:
        with _cache_lock:
            if cache_key in _memory_cache:
                ts, res = _memory_cache[cache_key]
                if time.time() - ts < CACHE_EXPIRY:
                    logger.info(f"Memory HIT: {symbol}")
                    return res

    # fresh data
    try:
        res = fetch_and_process(symbol)
        if res and len(res) >= 4 and res[0] is not None:

            # store in Redis
            if r:
                try:
                    r.setex(cache_key, CACHE_EXPIRY, pickle.dumps(res))
                    logger.info(f"Redis SET: {symbol}")
                except Exception as e:
                    logger.warning(f"Redis write failed: {e}")

            # store in memory fallback
            else:
                with _cache_lock:
                    _memory_cache[cache_key] = (time.time(), res)

            return res
    except Exception as e:
        logger.error(f"Fetch failed for {symbol}: {e}")

    return (None, None, None, None)
    

@app.get("/companies", tags=["General"])
async def get_companies():
   
    return {
        "supported_symbols": list(COMPANY_MAP.keys()),
        "company_map": COMPANY_MAP
    }

@app.get("/data/{symbol}")
async def get_stock_data(symbol: str):
    # Offload to threadpool to prevent freezing
    result = await run_in_threadpool(get_data_with_cache, symbol)
    
    
    if result[0] is None:
        raise HTTPException(status_code=404, detail="Symbol not found or data error.")
    df, pred, _, acc = result

    # convert DF to list and scrub all NaN/Inf values
    raw_data = df.to_dict(orient="records")
    safe_data = clean_for_json(raw_data)
    
    return {
        "symbol": symbol.upper(),
        "company": COMPANY_MAP.get(symbol.upper(), "Unknown"),
        "historical_data": safe_data,
        "ml_forecast": clean_for_json(pred),
        "accuracy": clean_for_json(acc)
    }

@app.get("/export/{symbol}")
async def export_to_csv(symbol: str):
    result = await run_in_threadpool(get_data_with_cache, symbol)
    if result[0] is None:
        raise HTTPException(status_code=404, detail="Data not available.")
    
    stream = io.StringIO()
    result[0].to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(stream, media_type="text/csv")

@app.get("/compare", tags=["Analytics"])
async def compare_stocks(symbol1: str, symbol2: str):
    s1, s2 = symbol1.upper(), symbol2.upper()

    # Concurrent fetch
    res1, res2 = await asyncio.gather(
        run_in_threadpool(get_data_with_cache, s1),
        run_in_threadpool(get_data_with_cache, s2)
    )

    df1, df2 = res1[0], res2[0]
    if df1 is None or df2 is None:
        missing = [s for s, d in [(s1, df1), (s2, df2)] if d is None]
        raise HTTPException(status_code=404, detail=f"Symbol(s) not found: {', '.join(missing)}")

    combined = pd.merge(
        df1[['Date', 'CLOSE']],
        df2[['Date', 'CLOSE']],
        on='Date',
        suffixes=(f'_{s1}', f'_{s2}')
    )

    if combined.empty:
        return {"error": "No overlapping trading dates found."}

    correlation = combined.iloc[:, 1].corr(combined.iloc[:, 2])

    if pd.isna(correlation):
        return {
            "comparison": f"{COMPANY_MAP.get(s1, s1)} vs {COMPANY_MAP.get(s2, s2)}",
            "correlation_coefficient": None,
            "insight": "Insufficient overlapping data to compute correlation."
        }

    corr_val = round(float(correlation), 3)
    return {
        "comparison": f"{COMPANY_MAP.get(s1, s1)} vs {COMPANY_MAP.get(s2, s2)}",
        "correlation_coefficient": corr_val,
        "insight": "Highly Correlated" if corr_val > 0.7 else "Moderately Correlated" if corr_val > 0.4 else "Low Correlation"
    }


@app.get("/summary/{symbol}", tags=["Stock Data"])
async def get_summary(symbol: str):
    symbol_up = symbol.upper()
    result = await run_in_threadpool(get_data_with_cache, symbol_up)

    if not result or len(result) < 4:
        raise HTTPException(status_code=503, detail="Data service unavailable.")

    _, _, summary, _ = result
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol_up} not found.")

    return {
        "symbol": symbol_up,
        "company_name": COMPANY_MAP.get(symbol_up, "Unknown Company"),
        "metrics": clean_for_json(summary)
    }

@app.get("/health", tags=["System"])
async def health_check():
    redis_status = "unavailable"
    if r:
        try:
            r.ping()
            redis_status = "connected"
        except:
            redis_status = "error"

    return {
        "status": "online",
        "redis": redis_status,
        "timestamp": time.time()
    }
# data visulaization without any frontend
@app.get("/chart/{symbol}", response_class=HTMLResponse, tags=["Visualization"])
async def get_chart(symbol: str):
    result = await run_in_threadpool(get_data_with_cache, symbol.upper())
    if result[0] is None:
        raise HTTPException(status_code=404, detail="Symbol not found.")

    df = result[0]
    company = COMPANY_MAP.get(symbol.upper(), symbol.upper())

    fig = go.Figure()


    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['OPEN'],
        high=df['HIGH'],
        low=df['LOW'],
        close=df['CLOSE'],
        name='Price'
    ))
    # MA7
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['MA_7'],
        mode='lines', name='MA 7',
        line=dict(color='orange', width=1.5)
    ))
    # MA20
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['MA_20'],
        mode='lines', name='MA 20',
        line=dict(color='blue', width=1.5)
    ))
    fig.update_layout(
        title=f"{company} — Last 30 Days",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False
    )
    return fig.to_html(full_html=True, include_plotlyjs='cdn')

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"GLOBAL CRASH: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong inside the server.", "error": str(exc)}
    )
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from performance import fetch_and_process
import pandas as pd
import time
import io
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=" Stock Intelligence API",
    description="Advanced Financial Data API with ML Forecasting and Technical Indicators",
)

# caching
cache = {}
CACHE_EXPIRY = 3600  # 1 Hour

def get_data_with_cache(symbol: str):
    symbol = symbol.upper()
    now = time.time()
    
    if symbol in cache:
        timestamp, cached_res = cache[symbol]
        if now - timestamp < CACHE_EXPIRY:
            logger.info(f"Cache Hit for {symbol}")
            return cached_res
    
    logger.info(f"Cache Miss/Fetch for {symbol}")
    res = fetch_and_process(symbol)
    
    # only cache if data was successfully fetched
    if res[0] is not None:
        cache[symbol] = (now, res)
    return res

# performance tracking middleware
@app.middleware("http")
async def log_execution_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"Path: {request.url.path} | Duration: {duration:.4f}s")
    return response

@app.get("/companies", tags=["General"])
async def get_companies():
   
    return ["AAPL", "MSFT", "TSLA", "GOOGL", "TCS.NS", "INFY.NS", "RELIANCE.NS"]

@app.get("/data/{symbol}", tags=["Stock Data"])
async def get_stock_data(symbol: str):
    df, predictions, _, accuracy = get_data_with_cache(symbol)
    
    if df is None:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found.")
    
    return {
        "symbol": symbol.upper(),
        "historical_data": df.to_dict(orient="records"),
        "ml_forecast_next_5_days": predictions,
        "model_performance": accuracy
    }

@app.get("/summary/{symbol}", tags=["Stock Data"])
async def get_summary(symbol: str):
    _, _, summary, _ = get_data_with_cache(symbol)
    
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found.")
    
    return summary

@app.get("/compare", tags=["Analytics"])
async def compare_stocks(symbol1: str, symbol2: str):
    df1, _, _, _ = get_data_with_cache(symbol1)
    df2, _, _, _ = get_data_with_cache(symbol2)
    
    if df1 is None or df2 is None:
        raise HTTPException(status_code=404, detail="One or both symbols not found.")
    
    # Align dates and calculate correlation
    combined = pd.merge(
        df1[['Date', 'CLOSE']], 
        df2[['Date', 'CLOSE']], 
        on='Date', 
        suffixes=(f'_{symbol1}', f'_{symbol2}')
    )
    
    correlation = combined.iloc[:, 1].corr(combined.iloc[:, 2])
    
    return {
        "comparison": f"{symbol1.upper()} vs {symbol2.upper()}",
        "correlation_coefficient": round(float(correlation), 3),
        "insight": "Highly Correlated" if correlation > 0.7 else "Low Correlation"
    }

@app.get("/export/{symbol}", tags=["Utilities"])
async def export_to_csv(symbol: str):
    df, _, _, _ = get_data_with_cache(symbol)
    if df is None:
        raise HTTPException(status_code=404, detail="Symbol not found.")
    
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={symbol.upper()}_report.csv"}
    )

@app.get("/health", tags=["System"])
async def health_check():
    """System status and cache monitoring."""
    return {
        "status": "online",
        "cache_entries": len(cache),
        "timestamp": time.time()
    }
# 📈 Stock Intelligence API
 
A production-ready FastAPI backend for real-time stock data, technical indicators, ML-based price forecasting, interactive chart visualization, and Redis-powered caching.
 
---
 
## 🚀 Features
 
- Real-time stock data via `yfinance`
- Technical indicators — RSI, Moving Averages (MA7, MA20), Volatility
- ML price forecasting using Linear Regression
- **Interactive chart visualization** — candlestick, volume, RSI, and ML forecast in browser
- Redis caching with in-memory fallback
- CSV export endpoint
- Multi-stock correlation comparison
- Dockerized deployment with Docker Compose
  
- 🚧 This project is under active development. Collaborators and contributions are highly welcome!
---
 
## 🗂️ Project Structure
 
```
stock-intelligence-api/
├── main.py               # FastAPI app, routes, Redis cache
├── performance.py        # Data fetching, indicators, ML model
├── Dockerfile            # Container build instructions
├── docker-compose.yml    # Multi-container setup (API + Redis)
├── requirements.txt      # Python dependencies
├── .dockerignore         # Files excluded from Docker build
└── .gitignore            # Files excluded from Git
```
 
---
 
## 📦 Supported Symbols
 
| Symbol | Company |
|--------|---------|
| AAPL | Apple Inc. |
| MSFT | Microsoft Corporation |
| TSLA | Tesla, Inc. |
| GOOGL | Alphabet Inc. (Google) |
| TCS.NS | Tata Consultancy Services |
| INFY.NS | Infosys Limited |
| RELIANCE.NS | Reliance Industries Limited |
 
---
 
## 🛠️ Local Setup (Without Docker)
 
### 1. Clone the repository
```bash
git clone https://github.com/Shruti-gupta427/stock-intelligence-api.git
cd stock-intelligence-api
```
 
### 2. Create and activate virtual environment
```bash
python -m venv venv
 
# Windows
venv\Scripts\activate
 
# Mac/Linux
source venv/bin/activate
```
 
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
 
### 4. Run the server
```bash
uvicorn main:app --reload
```
 
API will be live at `http://localhost:8000/docs`
 
---
 
## 🐳 Docker Deployment
 
### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running
### Run with Docker Compose
```bash
docker compose up --build -d
```
 
This starts two containers:
- `api` — FastAPI app on port 8000
- `redis` — Redis cache on port 6379
### Useful Docker Commands
```bash
# View live logs
docker compose logs -f
 
# API logs only
docker compose logs -f api
 
# Check what's cached in Redis
docker compose exec redis redis-cli keys "*"
 
# Clear Redis cache
docker compose exec redis redis-cli flushall
 
# Stop containers
docker compose down
 
# Stop and delete Redis data
docker compose down -v
 
# Rebuild after code changes
docker compose up --build -d
```
 
---
 
## ⚠️ Network Note — College/Office Wi-Fi
 
If you are on a **college or office network**, Docker may fail to pull images with errors like:
 
```
tls: protocol version not supported
http: server gave HTTP response to HTTPS client
```
 
This happens because institutional networks use a **transparent proxy/firewall** that intercepts and blocks Docker Hub's SSL connections (`registry-1.docker.io`, `auth.docker.io`).
 
**Fix — use mobile hotspot for pulling images:**
 
1. Turn on hotspot on your phone
2. Connect your laptop to it
3. Pull the required image:
```bash
docker pull redis:7-alpine
```
4. Switch back to your college/office Wi-Fi
5. Run normally:
```bash
docker compose up --build -d
```
 
> **Note:** Docker images are cached locally after the first pull. You only need hotspot once per image — future runs use the cached version regardless of network.
 
---
 
## 🔗 API Endpoints
 
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/docs` | Swagger UI |
| GET | `/companies` | List all supported symbols |
| GET | `/data/{symbol}` | Historical data + ML forecast |
| GET | `/summary/{symbol}` | 52-week high/low, avg close |
| GET | `/compare?symbol1=X&symbol2=Y` | Correlation between two stocks |
| GET | `/export/{symbol}` | Download data as CSV |
| GET | `/chart/{symbol}` | Interactive chart in browser |
| GET | `/health` | API + Redis status |
 
---
 
## 📉 Interactive Chart — `/chart/{symbol}`
 
Open directly in any browser — no frontend needed.
 
```
http://localhost:8000/chart/GOOGL
http://localhost:8000/chart/AAPL
http://localhost:8000/chart/TCS.NS
```
 
The chart has three synced panels:
 
| Panel | Content |
|-------|---------|
| Top (60%) | Candlestick + MA7 + MA20 + ML forecast (dashed) |
| Middle (20%) | Volume bars — green if price up, red if down |
| Bottom (20%) | RSI with overbought (70) and oversold (30) lines |
 
**Features:**
- Dark themed with Plotly
- Zoom and pan across all panels simultaneously
- Unified hover — hover on any point shows all values at once
- ML forecast extends 5 trading days beyond the last data point
- No frontend or additional setup required — rendered as pure HTML
---
 
## 📊 Sample Response — `/data/GOOGL`
 
```json
{
  "symbol": "GOOGL",
  "company": "Alphabet Inc. (Google)",
  "historical_data": [
    {
      "Date": "2026-03-06",
      "CLOSE": 298.3099,
      "HIGH": 300.3185,
      "LOW": 294.9723,
      "OPEN": 295.8817,
      "VOLUME": 25576900,
      "DAILY_RETURN": -0.000784,
      "MA_7": 304.32,
      "MA_20": 308.87,
      "RSI": 42.31,
      "VOLATILITY": 0.0144
    }
  ],
  "ml_forecast": [295.12, 294.87, 294.63, 294.38, 294.14],
  "accuracy": {
    "r2_score": 0.843,
    "mean_absolute_error": 12.45,
    "model_confidence": "High"
  }
}
```
 
---
 
## ⚙️ Tech Stack
 
| Layer | Technology |
|-------|-----------|
| Framework | FastAPI |
| Data Source | yfinance |
| ML Model | scikit-learn (Linear Regression) |
| Visualization | Plotly |
| Cache | Redis 7 |
| Server | Uvicorn |
| Containerization | Docker + Docker Compose |
| Language | Python 3.11 |
 
---
 
## 📝 Environment Variables
 
| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
 
---
 
## 👩‍💻 Author
 
**Shruti Gupta** — [@shrutixcodes](https://github.com/Shruti-gupta427)
 

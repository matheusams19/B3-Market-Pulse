from pathlib import Path
import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

TICKERS = ["PETR4", "VALE3", "ITUB4", "BBDC4", "WEGE3", "ABEV3", "BBAS3", "B3SA3", "MGLU3"]
BASE = "https://brapi.dev/api/quote"

TOKEN = os.getenv("BRAPI_TOKEN")
HEADERS = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}

Path("data/bronze").mkdir(parents=True, exist_ok=True)

def fetch_history(ticker: str) -> pd.DataFrame:
    url = f"{BASE}/{ticker}?range=1y&interval=1d"
    r = requests.get(url, headers=HEADERS, timeout=30)
    data = r.json()

    if data.get("error"):
        raise RuntimeError(f"{ticker}: {data.get('message')} ({data.get('code')})")

    results = data.get("results") or []
    candles = results[0].get("historicalDataPrice") if results else None
    if not candles:
        raise RuntimeError(f"{ticker}: sem histórico")

    df = pd.DataFrame(candles)
    df["date"] = pd.to_datetime(df["date"], unit="s")
    df["ticker"] = ticker
    return df[["ticker", "date", "open", "high", "low", "close", "volume"]].sort_values("date")

ok, fail = 0, 0
for t in TICKERS:
    try:
        df = fetch_history(t)
        df.to_parquet(Path("data/bronze") / f"{t}.parquet", index=False)
        print(f"✅ Bronze: {t} ({len(df)} linhas)")
        ok += 1
    except Exception as e:
        print(f" Pulando {t}: {e}")
        fail += 1

print(f" Concluído: Bronze | ok={ok} fail={fail}")

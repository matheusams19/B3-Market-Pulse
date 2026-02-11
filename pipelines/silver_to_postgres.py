from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

ENGINE_URL = "postgresql://marketpulse:marketpulse@localhost:5432/marketpulse"
engine = create_engine(ENGINE_URL)

# limpa tabelas pra re-rodar sem duplicar
with engine.begin() as conn:
    conn.execute(text("DELETE FROM features_daily;"))
    conn.execute(text("DELETE FROM prices_daily;"))
    conn.execute(text("DELETE FROM risk_scores;"))

for file in Path("data/silver").glob("*.parquet"):
    df = pd.read_parquet(file)

    prices = df[["ticker", "date", "open", "high", "low", "close", "volume"]].copy()
    features = df[["ticker", "date", "rsi_14", "ma_20", "ma_50", "volatility_20"]].copy()

    prices.to_sql("prices_daily", engine, if_exists="append", index=False, method="multi")
    features.to_sql("features_daily", engine, if_exists="append", index=False, method="multi")

    print(f" Postgres: {file.name}")

print(" Concluído: Postgres")

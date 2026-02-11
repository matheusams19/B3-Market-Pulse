from pathlib import Path
import pandas as pd
import numpy as np

Path("data/silver").mkdir(parents=True, exist_ok=True)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

for file in Path("data/bronze").glob("*.parquet"):
    df = pd.read_parquet(file).sort_values("date")

    df["ret_1d"] = df["close"].pct_change()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_50"] = df["close"].rolling(50).mean()
    df["volatility_20"] = df["ret_1d"].rolling(20).std()
    df["rsi_14"] = rsi(df["close"], 14)

    out = Path("data/silver") / file.name
    df.to_parquet(out, index=False)
    print(f" Silver: {file.name}")

print(" Concluído: Silver")

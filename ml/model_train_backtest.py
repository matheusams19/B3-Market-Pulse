import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.linear_model import LogisticRegression

from ml.metrics import cumulative_return, max_drawdown, sharpe_ratio

ENGINE_URL = "postgresql://marketpulse:marketpulse@localhost:5432/marketpulse"
MODEL_NAME = "LR_TECH_SENT_V2"
THRESH = 0.55  # sinal comprado quando prob_up > 0.55

FEATURES = ["ma_20", "ma_50", "volatility_20", "avg_sentiment"]

def load_dataset(engine, ticker: str) -> pd.DataFrame:
    q = """
SELECT
  ticker,
  date,
  close,
  ma_20,
  ma_50,
  volatility_20,
  avg_sentiment
FROM gold_features
WHERE ticker = %(t)s
ORDER BY date
"""
    df = pd.read_sql(q, engine, params={"t": ticker})

    df["sentiment_3d"] = df["avg_sentiment"].rolling(3).mean().fillna(0)

    df["ret_1d"] = df["close"].pct_change()
    df["ret_5d_fwd"] = df["close"].pct_change(5).shift(-5)
    df["y_up_5d"] = (df["ret_5d_fwd"] > 0).astype(int)

    return df


def temporal_split(df: pd.DataFrame, train_ratio: float = 0.75):
    n = len(df)
    cut = int(n * train_ratio)
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    return train, test

def fit_predict(train: pd.DataFrame, test: pd.DataFrame):
    # limpa nulos
    train = train.dropna(subset=FEATURES + ["y_up_5d"])
    test = test.dropna(subset=FEATURES + ["y_up_5d", "ret_1d"])

    if len(train) < 120 or len(test) < 30:
        return None, None

    X_train = train[FEATURES].values
    y_train = train["y_up_5d"].values

    X_test = test[FEATURES].values

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    prob_up = clf.predict_proba(X_test)[:, 1]
    return test, prob_up

def backtest_from_probs(test: pd.DataFrame, prob_up: np.ndarray) -> pd.DataFrame:
    df = test.copy()
    df["prob_up"] = prob_up
    df["signal"] = (df["prob_up"] > THRESH).astype(int)

    # posição de hoje vale para o retorno de amanhã
    df["strategy_ret"] = df["signal"].shift(1).fillna(0) * df["ret_1d"].fillna(0)
    df["equity"] = (1 + df["strategy_ret"]).cumprod()

    # buy & hold no mesmo período (pra comparação justa)
    df["bh_ret"] = df["ret_1d"].fillna(0)
    df["bh_equity"] = (1 + df["bh_ret"]).cumprod()
    return df

def save_predictions(engine, ticker: str, df: pd.DataFrame):
    out = df[["date", "prob_up", "signal"]].copy()
    out["model_name"] = MODEL_NAME
    out["ticker"] = ticker
    out = out[["model_name", "ticker", "date", "prob_up", "signal"]]
    out = out.dropna().drop_duplicates(subset=["model_name", "ticker", "date"])

    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM model_predictions WHERE model_name=:m AND ticker=:t"),
            {"m": MODEL_NAME, "t": ticker},
        )

    out.to_sql("model_predictions", engine, if_exists="append", index=False, method="multi")

def save_equity(engine, ticker: str, df: pd.DataFrame, strategy_label: str, equity_col: str, ret_col: str):
    out = df[["date"]].copy()
    out["strategy"] = strategy_label
    out["ticker"] = ticker
    out["equity"] = df[equity_col]
    out["returns"] = df[ret_col]
    out = out.dropna(subset=["date", "equity"]).drop_duplicates(subset=["strategy", "ticker", "date"])

    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM backtest_equity WHERE strategy=:s AND ticker=:t"),
            {"s": strategy_label, "t": ticker},
        )

    out.to_sql("backtest_equity", engine, if_exists="append", index=False, method="multi")

def save_results(engine, ticker: str, metrics: dict):
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM model_results WHERE model_name=:m AND ticker=:t"),
            {"m": MODEL_NAME, "t": ticker},
        )

    ins = """
    INSERT INTO model_results
      (model_name, ticker, start_date, end_date, cumulative_return, sharpe, max_drawdown)
    VALUES
      (:model_name, :ticker, :start_date, :end_date, :cumulative_return, :sharpe, :max_drawdown)
    """
    with engine.begin() as conn:
        conn.execute(text(ins), {"model_name": MODEL_NAME, "ticker": ticker, **metrics})

def main():
    engine = create_engine(ENGINE_URL)

    tickers = pd.read_sql("SELECT DISTINCT ticker FROM prices_daily ORDER BY ticker", engine)["ticker"].tolist()
    if not tickers:
        raise RuntimeError("Sem tickers em prices_daily.")

    for t in tickers:
        df = load_dataset(engine, t)
        df = df.dropna(subset=["ret_1d"])  # garante retorno

        train, test = temporal_split(df)
        test_clean, prob_up = fit_predict(train, test)
        if test_clean is None:
            print(f" Pulando {t}: poucos dados após limpeza")
            continue

        bt = backtest_from_probs(test_clean, prob_up)

        metrics = {
            "start_date": bt["date"].min(),
            "end_date": bt["date"].max(),
            "cumulative_return": cumulative_return(bt["strategy_ret"]),
            "sharpe": sharpe_ratio(bt["strategy_ret"]),
            "max_drawdown": max_drawdown(bt["equity"]),
        }

        save_predictions(engine, t, bt)
        save_results(engine, t, metrics)

        # salvar curvas para comparar no Streamlit
        save_equity(engine, t, bt, f"{MODEL_NAME}_STRAT", "equity", "strategy_ret")
        save_equity(engine, t, bt, f"{MODEL_NAME}_BUY_HOLD", "bh_equity", "bh_ret")

        print(
            f" {t} | ML Retorno={metrics['cumulative_return']:.2%} | Sharpe={metrics['sharpe']:.2f} | DD={metrics['max_drawdown']:.2%}"
        )

    print("ML concluído. Resultados em model_results e model_predictions.")

if __name__ == "__main__":
    main()
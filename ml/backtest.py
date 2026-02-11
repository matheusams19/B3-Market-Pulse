import pandas as pd
from sqlalchemy import create_engine, text

from ml.metrics import cumulative_return, max_drawdown, sharpe_ratio
from ml.signals import ma_crossover_signal

ENGINE_URL = "postgresql://marketpulse:marketpulse@localhost:5432/marketpulse"

def load_data(engine, ticker: str) -> pd.DataFrame:
    # Junta preço + features no dia
    q = """
    SELECT
      p.ticker,
      p.date,
      p.close,
      f.ma_20,
      f.ma_50
    FROM prices_daily p
    JOIN features_daily f
      ON p.ticker = f.ticker AND p.date = f.date
    WHERE p.ticker = %(t)s
    ORDER BY p.date
    """
    return pd.read_sql(q, engine, params={"t": ticker})

def run_backtest(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["ret_1d"] = df["close"].pct_change()

    # sinal / posição (0 ou 1)
    df["position"] = ma_crossover_signal(df)

    # regra: posição de hoje decide retorno de amanhã (evita look-ahead)
    df["strategy_ret"] = df["position"].shift(1).fillna(0) * df["ret_1d"].fillna(0)

    # curva de patrimônio
    df["equity"] = (1 + df["strategy_ret"]).cumprod()

    result = {
        "start_date": df["date"].min(),
        "end_date": df["date"].max(),
        "cumulative_return": cumulative_return(df["strategy_ret"]),
        "sharpe": sharpe_ratio(df["strategy_ret"]),
        "max_drawdown": max_drawdown(df["equity"]),
    }
    return result

def save_result(engine, ticker: str, strategy: str, metrics: dict):
    ins = """
    INSERT INTO backtest_results
      (strategy, ticker, start_date, end_date, cumulative_return, sharpe, max_drawdown)
    VALUES
      (:strategy, :ticker, :start_date, :end_date, :cumulative_return, :sharpe, :max_drawdown)
    """
    with engine.begin() as conn:
        conn.execute(
            text(ins),
            {
                "strategy": strategy,
                "ticker": ticker,
                "start_date": metrics["start_date"],
                "end_date": metrics["end_date"],
                "cumulative_return": metrics["cumulative_return"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
            },
        )

def save_equity(engine, ticker: str, strategy: str, df: pd.DataFrame, col_equity: str, col_ret: str):
    out = df[["date"]].copy()
    out["strategy"] = strategy
    out["ticker"] = ticker
    out["equity"] = df[col_equity]
    out["returns"] = df[col_ret]

    out = out.dropna(subset=["date", "equity"])
    out = out.drop_duplicates(subset=["strategy", "ticker", "date"])

    out.to_sql(
        "backtest_equity",
        engine,
        if_exists="append",
        index=False,
        method="multi"
    )

def main():
    engine = create_engine(ENGINE_URL)

    tickers = pd.read_sql("SELECT DISTINCT ticker FROM prices_daily ORDER BY ticker", engine)["ticker"].tolist()
    if not tickers:
        raise RuntimeError("Sem tickers em prices_daily. Você carregou o Postgres?")

    strategy_name = "MA20_GT_MA50"

    # opcional: limpa resultados anteriores dessa estratégia (pra não duplicar)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM backtest_results WHERE strategy = :s"), {"s": strategy_name})

    for t in tickers:
        df = load_data(engine, t)
        if len(df) < 60:
            print(f"⚠️ Pulando {t}: poucos dados ({len(df)})")
            continue

        metrics = run_backtest(df)
        save_result(engine, t, strategy_name, metrics)

        print(
            f"✅ {t} | Retorno={metrics['cumulative_return']:.2%} | Sharpe={metrics['sharpe']:.2f} | DD={metrics['max_drawdown']:.2%}"
        )

    print(" Backtest concluído e salvo em backtest_results.")

if __name__ == "__main__":
    main()
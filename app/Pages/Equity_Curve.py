import datetime as dt
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Equity Curve", layout="wide")
st.title("📈 Equity Curve (Sem vs Com Sentimento)")

ENGINE_URL = "postgresql://marketpulse:marketpulse@localhost:5432/marketpulse"
engine = create_engine(ENGINE_URL)

BASE_MODEL = "LR_TECH_V1"
SENT_MODEL = "LR_TECH_SENT_V2"

@st.cache_data(ttl=60)
def load_tickers():
    q = "SELECT DISTINCT ticker FROM model_predictions ORDER BY ticker;"
    return pd.read_sql(text(q), engine)["ticker"].tolist()

@st.cache_data(ttl=60)
def load_prices(ticker: str) -> pd.DataFrame:
    q = """
    SELECT date, close
    FROM prices_daily
    WHERE ticker = :t
    ORDER BY date;
    """
    d = pd.read_sql(text(q), engine, params={"t": ticker})
    d["date"] = pd.to_datetime(d["date"])
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    return d.dropna(subset=["date", "close"])

@st.cache_data(ttl=60)
def load_preds(ticker: str, model_name: str) -> pd.DataFrame:
    q = """
    SELECT date, signal AS y_pred
    FROM model_predictions
    WHERE ticker = :t
      AND model_name = :m
    ORDER BY date;
    """
    d = pd.read_sql(text(q), engine, params={"t": ticker, "m": model_name})
    d["date"] = pd.to_datetime(d["date"])
    d["y_pred"] = pd.to_numeric(d["y_pred"], errors="coerce").fillna(0).astype(int)
    return d


def build_equity(prices: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    x = prices.merge(preds, on="date", how="left")
    x["y_pred"] = x["y_pred"].fillna(0).astype(int)

    x["ret"] = x["close"].pct_change().fillna(0)
    x["pos"] = x["y_pred"].clip(0, 1)          # 1 compra, 0 caixa
    x["strat_ret"] = x["pos"] * x["ret"]
    x["equity"] = (1 + x["strat_ret"]).cumprod()
    return x

tickers = load_tickers()
colA, colB, colC = st.columns([2,2,2])

with colA:
    ticker = st.selectbox("Ticker", tickers, index=0)

with colB:
    base_model = st.text_input("Modelo base", BASE_MODEL)

with colC:
    sent_model = st.text_input("Modelo com sentimento", SENT_MODEL)

start_date = st.date_input("Data inicial", value=dt.date(2025, 1, 1))

prices = load_prices(ticker)
pred_base = load_preds(ticker, base_model)
pred_sent = load_preds(ticker, sent_model)

eq_base = build_equity(prices, pred_base)[["date","equity"]].rename(columns={"equity":"Equity Base"})
eq_sent = build_equity(prices, pred_sent)[["date","equity"]].rename(columns={"equity":"Equity Sent"})

eq = eq_base.merge(eq_sent, on="date", how="inner")
eq = eq[eq["date"] >= pd.to_datetime(start_date)]

if eq.empty:
    st.warning("Sem dados suficientes para plotar. Verifique datas / modelos.")
else:
    fig = px.line(eq, x="date", y=["Equity Base", "Equity Sent"], title=f"{ticker} — Equity Curve")
    st.plotly_chart(fig, use_container_width=True)

    total_base = float(eq["Equity Base"].iloc[-1] - 1)
    total_sent = float(eq["Equity Sent"].iloc[-1] - 1)

    c1, c2 = st.columns(2)
    c1.metric("Retorno acumulado (Base)", f"{total_base*100:.2f}%")
    c2.metric("Retorno acumulado (Sent)", f"{total_sent*100:.2f}%")

    st.caption("Estratégia simples: compra quando y_pred=1, senão fica em caixa.")

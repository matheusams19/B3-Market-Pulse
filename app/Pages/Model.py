import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

st.set_page_config(page_title="Modelo ML", layout="wide")
st.title("🤖 Modelo ML — Resultados & Equity")

from app.db import get_engine
engine = get_engine()

@st.cache_data(ttl=60)
def load_models():
    q = """
    SELECT DISTINCT model_name
    FROM model_results
    ORDER BY model_name
    """
    return pd.read_sql(q, engine)["model_name"].tolist()

@st.cache_data(ttl=60)
def load_tickers(model_name: str):
    q = """
    SELECT DISTINCT ticker
    FROM model_results
    WHERE model_name = %(m)s
    ORDER BY ticker
    """
    return pd.read_sql(q, engine, params={"m": model_name})["ticker"].tolist()

@st.cache_data(ttl=60)
def load_results(model_name: str):
    q = """
    SELECT
      ticker,
      cumulative_return,
      sharpe,
      max_drawdown
    FROM model_results
    WHERE model_name = %(m)s
    ORDER BY sharpe DESC
    """
    return pd.read_sql(q, engine, params={"m": model_name})

@st.cache_data(ttl=60)
def load_equity(model_name: str, ticker: str):
    q = """
    SELECT
      date,
      strategy,
      equity
    FROM backtest_equity
    WHERE ticker = %(t)s
      AND strategy IN (%(m)s, %(bh)s)
    ORDER BY date
    """
    return pd.read_sql(
        q,
        engine,
        params={"t": ticker, "m": f"{model_name}_STRAT", "bh": f"{model_name}_BUY_HOLD"},
    )

@st.cache_data(ttl=60)
def load_predictions(model_name: str, ticker: str):
    q = """
    SELECT
      date,
      prob_up,
      signal
    FROM model_predictions
    WHERE model_name = %(m)s
      AND ticker = %(t)s
    ORDER BY date
    """
    return pd.read_sql(q, engine, params={"m": model_name, "t": ticker})

models = load_models()
if not models:
    st.warning("Nenhum modelo encontrado.")
    st.stop()

model = st.selectbox("Modelo", models)

df_rank = load_results(model)

# KPIs globais
c1, c2, c3 = st.columns(3)
c1.metric("Sharpe Médio", f"{df_rank['sharpe'].mean():.2f}")
c2.metric("Melhor Retorno", f"{df_rank['cumulative_return'].max():.2%}")
c3.metric("Pior Drawdown", f"{df_rank['max_drawdown'].min():.2%}")

st.subheader("📊 Ranking por Sharpe")
st.dataframe(
    df_rank.style.format(
        {
            "cumulative_return": "{:.2%}",
            "sharpe": "{:.2f}",
            "max_drawdown": "{:.2%}",
        }
    ),
    use_container_width=True,
)

ticker = st.selectbox("Ativo", load_tickers(model))

# Equity curve
df_eq = load_equity(model, ticker)
st.subheader("📈 Equity Curve — Modelo vs Buy & Hold")
fig_eq = px.line(
    df_eq,
    x="date",
    y="equity",
    color="strategy",
    title=f"{ticker} — {model}",
)
st.plotly_chart(fig_eq, use_container_width=True)

# Probabilidades
df_pred = load_predictions(model, ticker)
st.subheader("🧠 Probabilidade prevista (prob_up)")
fig_p = px.line(
    df_pred,
    x="date",
    y="prob_up",
    title=f"{ticker} — prob_up",
)
st.plotly_chart(fig_p, use_container_width=True)

with st.expander("📋 Predições"):
    st.dataframe(df_pred.tail(50), use_container_width=True)

st.markdown(
    """
    ---
    <div style="text-align: center;">
        <small>Desenvolvido por MarketPulse 🚀</small>
    </div>
    """,
    unsafe_allow_html=True,
)

import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

st.set_page_config(page_title="Backtest", layout="wide")
st.title("📈 Backtest — Estratégia MA20 > MA50")

engine = create_engine("postgresql://marketpulse:marketpulse@localhost:5432/marketpulse")

@st.cache_data(ttl=60)
def load_results():
    q = """
    SELECT
      strategy,
      ticker,
      start_date,
      end_date,
      cumulative_return,
      sharpe,
      max_drawdown
    FROM backtest_results
    ORDER BY cumulative_return DESC
    """
    return pd.read_sql(q, engine)

df = load_results()

if df.empty:
    st.warning("Nenhum backtest encontrado.")
    st.stop()

# KPIs gerais
c1, c2, c3 = st.columns(3)
c1.metric("Melhor Retorno", f"{df['cumulative_return'].max():.2%}")
c2.metric("Sharpe Médio", f"{df['sharpe'].mean():.2f}")
c3.metric("Pior Drawdown", f"{df['max_drawdown'].min():.2%}")

st.subheader("Ranking por Retorno")
st.dataframe(
    df.style.format(
        {
            "cumulative_return": "{:.2%}",
            "sharpe": "{:.2f}",
            "max_drawdown": "{:.2%}",
        }
    ),
    use_container_width=True,
)

# Gráfico
fig = px.bar(
    df,
    x="ticker",
    y="cumulative_return",
    color="sharpe",
    title="Retorno acumulado por ativo (cor = Sharpe)",
)
st.plotly_chart(fig, use_container_width=True)
st.markdown(
    """
    ---
    <div style="text-align: center;">
        <small>Desenvolvido por MarketPulse 🚀</small>
    </div>
    """,
    unsafe_allow_html=True,
)
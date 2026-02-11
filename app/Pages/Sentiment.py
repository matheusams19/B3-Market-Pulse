import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

st.set_page_config(page_title="Sentimento de Mercado", layout="wide")
st.title("🧠 Sentimento de Mercado (NLP)")

from app.db import get_engine
engine = get_engine()

@st.cache_data(ttl=60)
def load_tickers():
    return pd.read_sql(
        "SELECT DISTINCT ticker FROM sentiment_daily ORDER BY ticker",
        engine
    )["ticker"].tolist()

@st.cache_data(ttl=60)
def load_sentiment(ticker):
    q = """
    SELECT
      date,
      avg_sentiment,
      n_items,
      sample_titles
    FROM sentiment_daily
    WHERE ticker = %(t)s
    ORDER BY date
    """
    return pd.read_sql(q, engine, params={"t": ticker})

@st.cache_data(ttl=60)
def load_prices(ticker):
    q = """
    SELECT date, close
    FROM prices_daily
    WHERE ticker = %(t)s
    ORDER BY date
    """
    return pd.read_sql(q, engine, params={"t": ticker})

ticker = st.selectbox("Ativo", load_tickers())

df_sent = load_sentiment(ticker)
df_price = load_prices(ticker)

# --- GRÁFICO PREÇO ---
st.subheader("📈 Preço")
fig_p = px.line(df_price, x="date", y="close", title=f"{ticker} — Preço")
st.plotly_chart(fig_p, use_container_width=True)

# --- GRÁFICO SENTIMENTO ---
st.subheader("🧠 Sentimento Diário (com volume de notícias)")

# garante tipo correto
df_sent["avg_sentiment"] = pd.to_numeric(df_sent["avg_sentiment"], errors="coerce").fillna(0)
df_sent["n_items"] = pd.to_numeric(df_sent["n_items"], errors="coerce").fillna(0)

# média móvel pra ficar visível mesmo com muito "neutral"
df_sent["sentiment_3d"] = df_sent["avg_sentiment"].rolling(3).mean()

fig_scatter = px.scatter(
    df_sent,
    x="date",
    y="avg_sentiment",
    size="n_items",
    hover_data=["n_items"],
    title=f"{ticker} — Sentimento (pontos) | tamanho = nº notícias",
)
st.plotly_chart(fig_scatter, use_container_width=True)

fig_line = px.line(
    df_sent,
    x="date",
    y="sentiment_3d",
    title=f"{ticker} — Sentimento (média móvel 3 dias)",
)
st.plotly_chart(fig_line, use_container_width=True)

st.subheader("📋 Últimos registros de sentimento")
st.dataframe(df_sent.tail(30).sort_values("date", ascending=False), use_container_width=True)

# --- INTERPRETAÇÃO ---
st.subheader("📊 Interpretação")
st.markdown("""
- 🟢 **Positivo**: notícias favoráveis, otimismo
- 🟡 **Neutro**: mercado sem viés claro
- 🔴 **Negativo**: pessimismo, risco aumentado

O sentimento **não prevê preço**, mas ajuda a:
- filtrar entradas ruins  
- reduzir drawdown  
- entender o contexto do movimento
""")

# --- MANCHETES ---
st.subheader("📰 Manchetes Recentes")
df_last = df_sent.tail(5).sort_values("date", ascending=False)
for _, row in df_last.iterrows():
    st.markdown(f"**{row['date']}** — Sent: `{row['avg_sentiment']:.2f}`")
    st.caption(row["sample_titles"])
    st.divider()

st.markdown(
    """
    ---
    <div style="text-align: center;">
        <small>Desenvolvido por MarketPulse 🚀</small>
    </div>
    """,
    unsafe_allow_html=True,
)

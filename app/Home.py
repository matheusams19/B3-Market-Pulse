import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

st.set_page_config(page_title="B3 MarketPulse", layout="wide")
st.title("📊 B3 MarketPulse")

DATABASE_URL = st.secrets["DATABASE_URL"]

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

@st.cache_data(ttl=60)
def get_tickers():
    return pd.read_sql("SELECT DISTINCT ticker FROM prices_daily ORDER BY ticker", engine)["ticker"].tolist()

@st.cache_data(ttl=60)
def get_prices(ticker: str):
    return pd.read_sql(
        "SELECT date, close, volume FROM prices_daily WHERE ticker = %(t)s ORDER BY date",
        engine,
        params={"t": ticker},
    )

tickers = get_tickers()
ticker = st.selectbox("Ativo", tickers)

df = get_prices(ticker)

c1, c2, c3 = st.columns(3)
c1.metric("Registros", f"{len(df)}")
if len(df):
    c2.metric("Último close", f"{df['close'].iloc[-1]:.2f}")
    c3.metric("Volume (último dia)", f"{int(df['volume'].iloc[-1]):,}".replace(",", "."))

fig = px.line(df, x="date", y="close", title=f"{ticker} — Fechamento (1 ano)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Últimas 20 linhas")
st.dataframe(df.tail(20), use_container_width=True)

st.markdown(
    """
    ---
    <div style="text-align: center;">
        <small>Desenvolvido por MarketPulse 🚀</small>
    </div>
    """,
    unsafe_allow_html=True,
)

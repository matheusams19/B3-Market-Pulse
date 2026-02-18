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


# 1. Configuração da Barra Lateral (Índice)
st.sidebar.title("📌 Índice do Projeto")
st.sidebar.markdown("Selecione a página que deseja analisar:")

# Botões de Navegação - Devem ser os únicos a usar st.switch_page
if st.sidebar.button("🏠 Home"):
    st.switch_page("Home.py")

if st.sidebar.button("🗄️ Database"):
    st.switch_page("db.py")

st.sidebar.markdown("---")
st.sidebar.subheader("Análises (Pages)")

# Use os caminhos exatos que aparecem no seu repositório GitHub
if st.sidebar.button("📊 Backtest"):
    st.switch_page("pages/Backtest.py")

if st.sidebar.button("📈 Equity Curve"):
    st.switch_page("pages/Equity_Curve.py")

if st.sidebar.button("🤖 Model"):
    st.switch_page("pages/Model.py")

if st.sidebar.button("⚖️ Model Compare"):
    st.switch_page("pages/Model_Compare.py")

if st.sidebar.button("💬 Sentiment Analysis"):
    st.switch_page("pages/Sentiment.py")

# 2. Conteúdo EXCLUSIVO da Home
# Remova daqui qualquer código que pertença ao Backtest ou outros módulos
st.title("MarketPulse - Dashboard Financeiro 🚀")
st.write("Bem-vindo ao sistema. Utilize o menu lateral para acessar as métricas específicas.")

# Exemplo de métricas gerais (opcional para a Home)
col1, col2 = st.columns(2)
col1.metric("Status do Sistema", "Online")
col2.metric("Última Atualização", "2026-02-18")

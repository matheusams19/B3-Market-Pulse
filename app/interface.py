import streamlit as st

def criar_menu():
    # Configuração da Barra Lateral (Índice)
    st.sidebar.title("📌 Índice do Projeto")
    st.sidebar.markdown("Selecione a página:")

    # Botões de Navegação
    if st.sidebar.button("🏠 Home"):
        st.switch_page("Home.py")

    if st.sidebar.button("🗄️ Database"):
        st.switch_page("db.py")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Análises")

    # Links para a pasta pages
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

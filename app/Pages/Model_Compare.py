import os
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Comparação de Modelos", layout="wide")
st.title("📊 Comparação de Modelos (Sem vs Com Sentimento) — Camada GOLD")

def get_engine_url():
    # 1) Streamlit Cloud (secrets)
    try:
        url = st.secrets.get("DATABASE_URL")
        if url:
            return url
    except Exception:
        pass

    # 2) Variável de ambiente (opcional)
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    # 3) Local (docker)
    return "postgresql://marketpulse:marketpulse@localhost:5432/marketpulse"

ENGINE_URL = get_engine_url()
engine = create_engine(ENGINE_URL)

st.caption(
    "Página consumindo diretamente a camada GOLD (view gold_model_decision). "
    "Cores indicam impacto do sentimento: melhora risco-retorno (verde), "
    "melhora com aumento de risco (amarelo), piora (vermelho), neutro (cinza)."
)

@st.cache_data(ttl=60)
def load_gold_decisions() -> pd.DataFrame:
    q = """
    SELECT
        ticker,
        delta_sharpe,
        delta_return,
        delta_drawdown,
        decision_label
    FROM public.gold_model_decision
    ORDER BY delta_sharpe DESC NULLS LAST;
    """
    df = pd.read_sql(text(q), engine)

    for c in ["delta_sharpe", "delta_return", "delta_drawdown"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["decision_label"] = df["decision_label"].fillna("NEUTRO").astype(str)
    return df

def label_to_ui(label: str) -> str:
    return {
        "MELHOR_RISCO_RETORNO": "Melhor risco-retorno",
        "RETORNO_MAIOR_RISCO": "Retorno maior, mais risco",
        "PIOROU_MODELO": "Piorou o modelo",
        "NEUTRO": "Neutro",
    }.get(label, "Neutro")

def row_style(row):
    imp = row.get("Impacto", "Neutro")
    if imp == "Melhor risco-retorno":
        return ["background-color: #1f7a1f; color: white"] * len(row)
    if imp == "Retorno maior, mais risco":
        return ["background-color: #b59b00; color: black"] * len(row)
    if imp == "Piorou o modelo":
        return ["background-color: #7a1f1f; color: white"] * len(row)
    return ["background-color: #2b2b2b; color: white"] * len(row)

df = load_gold_decisions()
df["impact"] = df["decision_label"].apply(label_to_ui)

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ativos (linhas)", int(df["ticker"].notna().sum()))
c2.metric("Melhoraram (ΔSharpe>0)", int((df["delta_sharpe"] > 0).sum()))
c3.metric("Pioraram (ΔSharpe<0)", int((df["delta_sharpe"] < 0).sum()))
c4.metric("Δ Sharpe médio", f"{df['delta_sharpe'].mean():.2f}")

st.divider()

# Top / Worst
best = df.sort_values("delta_sharpe", ascending=False).head(3)
worst = df.sort_values("delta_sharpe", ascending=True).head(3)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🟢 Top melhorias (Δ Sharpe)")
    st.dataframe(best[["ticker","delta_sharpe","delta_return","delta_drawdown","impact"]], use_container_width=True)

with col2:
    st.markdown("### 🔴 Piores impactos (Δ Sharpe)")
    st.dataframe(worst[["ticker","delta_sharpe","delta_return","delta_drawdown","impact"]], use_container_width=True)

st.download_button(
    "⬇️ Baixar (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="gold_model_decision.csv",
    mime="text/csv",
)

st.info(
    "🧠 Interpretação:\n"
    "- Verde: Sharpe ↑ e Drawdown ↓ (melhor risco-retorno)\n"
    "- Amarelo: Sharpe ↑ mas drawdown piora\n"
    "- Vermelho: Sharpe ↓ (piorou)\n"
    "- Cinza: neutro"
)

st.subheader("📋 Ranking (com cores)")

df_show = df.copy()
df_show["delta_sharpe"] = df_show["delta_sharpe"].round(2)
df_show["delta_return"] = (df_show["delta_return"] * 100).round(2)
df_show["delta_drawdown"] = (df_show["delta_drawdown"] * 100).round(2)

df_show = df_show.rename(columns={
    "delta_sharpe": "Δ Sharpe",
    "delta_return": "Δ Retorno (%)",
    "delta_drawdown": "Δ Drawdown (%)",
    "impact": "Impacto",
})

df_show = df_show[["ticker","Δ Sharpe","Δ Retorno (%)","Δ Drawdown (%)","Impacto"]].sort_values("Δ Sharpe", ascending=False)

st.dataframe(df_show.style.apply(row_style, axis=1), use_container_width=True, height=380)

st.divider()

st.subheader("🧭 Mapa de Impacto (Δ Drawdown vs Δ Sharpe)")
fig = px.scatter(
    df,
    x="delta_drawdown",
    y="delta_sharpe",
    color="impact",
    text="ticker",
    title="Quadrantes: risco (x) vs performance (y)"
)
fig.add_hline(y=0, line_dash="dash", opacity=0.3)
fig.add_vline(x=0, line_dash="dash", opacity=0.3)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Leitura: Y>0 (ΔSharpe positivo) = melhora de retorno ajustado ao risco. "
    "X<0 (ΔDrawdown negativo) = redução de drawdown (melhor)."
)

st.markdown(
    """
    ---
    <div style="text-align: center;">
        <small>MarketPulse 🚀</small>
    </div>
    """,
    unsafe_allow_html=True,
)

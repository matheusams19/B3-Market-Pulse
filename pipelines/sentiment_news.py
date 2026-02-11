from urllib.parse import quote_plus
import feedparser
import pandas as pd
from datetime import datetime
from transformers import pipeline
from sqlalchemy import create_engine, text

ENGINE_URL = "postgresql://marketpulse:marketpulse@localhost:5432/marketpulse"
SOURCE = "news"

TICKER_MAP = {
    "PETR4": "Petrobras",
    "VALE3": "Vale",
    "ITUB4": "Itau",
    "BBDC4": "Bradesco",
    "BBAS3": "Banco do Brasil",
    "WEGE3": "WEG",
    "ABEV3": "Ambev",
    "B3SA3": "B3",
    "MGLU3": "Magazine Luiza",
}

def make_sentiment_pipeline():
    # Modelo robusto pra sentimento (3 classes)
    # Pode baixar na primeira execução
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    )

sentiment_model = make_sentiment_pipeline()

def fetch_news(query: str, limit: int = 20):
    # encode da query para evitar URL inválida (acentos/espaços)
    q = quote_plus(f"{query} stock")
    url = f"https://news.google.com/rss/search?q={q}&hl=pt-BR&gl=BR&ceid=BR:pt-419"

    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
    except Exception as e:
        print(f"⚠️ Falha ao buscar RSS para '{query}': {e}")
        return []

    items = []
    for e in getattr(feed, "entries", [])[:limit]:
        try:
            published = datetime(*e.published_parsed[:6]).date()
        except Exception:
            published = datetime.utcnow().date()

        title = getattr(e, "title", "").strip()
        if not title:
            continue

        items.append({"title": title, "published": published})

    return items

def scores_from_pipeline_output(out):
    """
    out pode vir como:
    - dict {"label": "...", "score": ...}
    - lista de dicts [{"label":..., "score":...}, ...]
    Queremos POS, NEG, NEU.
    """
    pos = neg = neu = 0.0

    if isinstance(out, dict):
        # caso venha só uma classe (top-1)
        label = out.get("label", "").upper()
        score = float(out.get("score", 0.0))
        if "POS" in label:
            pos = score
        elif "NEG" in label:
            neg = score
        else:
            neu = score
        return pos, neg, neu

    # caso venha lista com todas as classes
    if isinstance(out, list):
        d = {str(x.get("label", "")).upper(): float(x.get("score", 0.0)) for x in out}
        # alguns modelos podem usar LABEL_0 etc; fallback simples:
        pos = d.get("POSITIVE", d.get("LABEL_2", 0.0))
        neu = d.get("NEUTRAL", d.get("LABEL_1", 0.0))
        neg = d.get("NEGATIVE", d.get("LABEL_0", 0.0))
        return pos, neg, neu

    return pos, neg, neu

def score_texts(texts):
    """
    Retorna score contínuo por título: sent = pos - neg
    """
    scores = []
    for t in texts:
        # top_k=None = retorna todas as classes (compatível)
        res = sentiment_model(t[:512], top_k=None)

        # res pode vir como:
        # [ [ {label, score}, ... ] ] ou [ {label, score} ]
        # normalizamos:
        if isinstance(res, list) and len(res) > 0:
            first = res[0]
        else:
            first = res

        # Se first ainda é lista de dicts, ok. Se é dict, ok.
        pos, neg, neu = scores_from_pipeline_output(first)
        sent = pos - neg
        scores.append(sent)

    return scores

def main():
    engine = create_engine(ENGINE_URL)

    for ticker, name in TICKER_MAP.items():
        try:
            print(f"Buscando notícias para {ticker} ({name})")

            news = fetch_news(name, limit=20)
            if not news:
                print(f"Sem notícias para {ticker}")
                continue

            df = pd.DataFrame(news)

            # calcula sentimento por título
            df["sentiment"] = score_texts(df["title"].tolist())

            # agrega por dia
            daily = (
                df.groupby("published")
                .agg(
                    avg_sentiment=("sentiment", "mean"),
                    n_items=("sentiment", "count"),
                    sample_titles=("title", lambda x: " | ".join(list(x)[:3])),
                )
                .reset_index()
                .rename(columns={"published": "date"})
            )

            daily["ticker"] = ticker
            daily["source"] = SOURCE

            # remove dados antigos do ticker/fonte
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "DELETE FROM sentiment_daily "
                        "WHERE ticker = :t AND source = :s"
                    ),
                    {"t": ticker, "s": SOURCE},
                )

            # garante colunas certas
            daily = daily[
                ["ticker", "date", "source", "avg_sentiment", "n_items", "sample_titles"]
            ]

            # salva no banco
            daily.to_sql(
                "sentiment_daily",
                engine,
                if_exists="append",
                index=False,
                method="multi",
            )

            print(
                f"✅ {ticker} | dias={len(daily)} | "
                f"sent(min/mean/max)=("
                f"{daily['avg_sentiment'].min():.3f}/"
                f"{daily['avg_sentiment'].mean():.3f}/"
                f"{daily['avg_sentiment'].max():.3f})"
            )

        except Exception as e:
            print(f"Erro no ticker {ticker}: {e}")
            continue

    print("Pipeline de sentimento finalizado com sucesso.")

📈 B3 MarketPulse

Pipeline end-to-end de Data Science e Machine Learning aplicado ao mercado acionário brasileiro (B3), integrando preços históricos, análise de sentimento de notícias via NLP, backtesting de estratégias quantitativas e visualização interativa em Streamlit, com arquitetura em camadas Bronze / Silver / Gold e banco PostgreSQL local e em cloud (Neon).

🎯 Objetivo do Projeto

Construir um pipeline end-to-end de dados e Machine Learning aplicado ao mercado acionário brasileiro (B3), cobrindo desde ingestão, processamento, modelagem, backtesting e visualização, com foco em tomada de decisão quantitativa baseada em dados.

O projeto tem como objetivos centrais:

- Modelar estratégias quantitativas de trading a partir de dados históricos de preços e indicadores técnicos

- Avaliar desempenho financeiro realista por meio de backtesting, utilizando métricas como Sharpe Ratio, Cumulative Return e Max Drawdown

- Implementar uma arquitetura em camadas (Bronze / Silver / Gold), separando ingestão, transformação, modelagem e consumo analítico

- Persistir resultados em PostgreSQL, com views semânticas na camada GOLD para consumo direto por aplicações

- Disponibilizar os resultados em um dashboard interativo (Streamlit), permitindo análise por ativo, modelo e período

Como extensão analítica do projeto, é incorporada uma camada de análise de sentimento de notícias financeiras (NLP), utilizada como feature adicional nos modelos de Machine Learning, com o objetivo de:

- Avaliar se informações qualitativas (sentimento de notícias) agregam valor estatístico e financeiro às estratégias quantitativas

- Comparar, de forma controlada, modelos com e sem sentimento, medindo impacto real sobre risco e retorno

- Classificar o efeito do sentimento em categorias como melhor risco-retorno, maior retorno com mais risco, neutro ou negativo

O projeto não parte da premissa de que o sentimento melhora resultados, mas sim testa essa hipótese de forma mensurável, reproduzível e orientada a dados.

---

🧱 Arquitetura Geral (End-to-End)

```text
                 ┌──────────────────────────┐
                 │          Fontes          │
                 │───────────────────────── │
                 │ • Preços B3              │
                 │ • Notícias financeiras   │
                 └─────────────┬────────────┘
                               │
                    (Ingestão / ETL)
                               │
┌────────────────────────────────────────────────────┐
│                     BRONZE                         │
│────────────────────────────────────────────────────│
│ • prices_daily                                     │
│ • sentiment_raw (RSS / notícias)                   │
└─────────────┬──────────────────────────────────────┘
              │
       (Limpeza / Features)
              │
┌────────────────────────────────────────────────────┐
│                     SILVER                         │
│────────────────────────────────────────────────────│
│ • features_daily                                   │
│ • sentiment_daily                                  │
│ • risk_scores                                      │
└─────────────┬──────────────────────────────────────┘
              │
     (ML + Backtest + Agregações)
              │
┌────────────────────────────────────────────────────┐
│                      GOLD                          │
│────────────────────────────────────────────────────│
│ • model_results                                    │
│ • model_predictions                                │
│ • gold_features (VIEW)                             │
│ • gold_model_decision (VIEW)                       │
│ • gold_trading_decision (VIEW)                     │
└─────────────┬──────────────────────────────────────┘
              │
        (Consumo direto)
              │
┌────────────────────────────────────────────────────┐
│                STREAMLIT APP                       │
│────────────────────────────────────────────────────│
│ • Comparação de Modelos                            │
│ • Ranking por impacto                              │
│ • Equity Curve (com vs sem sentimento)             │
│ • Visualização risco-retorno                       │
└────────────────────────────────────────────────────┘
```
---

🧠 Modelagem e Machine Learning

Modelos:

- LR_TECH_V1 → modelo base (sem sentimento)
- LR_TECH_SENT_V2 → modelo com sentimento de notícias

Features principais:

- Retornos históricos
- Volatilidade
- Indicadores técnicos
- Score de sentmento (NLP)

 NLP: 
- Modelo RoBERTa pré-treinado para sentimento financeiro
- Agregação diária do sentimento por ativo
- Integração como feature explicativa no modelo

📊 Backtesting e Métricas

Para cada ticker e modelo:

- 📈 Equity Curve
- 📐 Sharpe Ratio
- 📉 Max Drawdown
- 💰 Cumulative Return

Comparação direta:

- SELECT * FROM gold_model_decision;

Classificação automática:

🟢 Melhor risco-retorno
🟡 Retorno maior, mais risco
🔴 Piorou o modelo
⚪ Neutro

---

🖥️ Streamlit App

Páginas principais:

- Comparação de Modelos (Camada GOLD)
- Ranking com cores por impacto
- Mapa de risco vs performance
- Equity Curve (sem vs com sentimento)

Destaques:

- Consome views GOLD diretamente
- Totalmente dinâmico (ticker, modelo, datas)
- Pronto para deploy em cloud

---

🗄️ Banco de Dados

Local:

- PostgreSQL via Docker
- Utilizado para desenvolvimento e testes

Cloud

- PostgreSQL serverless via Neon
- Ideal para integração com Streamlit Cloud
- Migração via pg_dump / pg_restore

---

🧰 Stack Tecnológica

Linguagens & Core
Python
SQL
Data & ML
Pandas
NumPy
Scikit-learn
Transformers (NLP)
PyTorch
Banco & Infra
PostgreSQL
Docker
Neon (Postgres Cloud)
Visualização
Streamlit
Plotly
Arquitetura
Data Lakehouse (Bronze / Silver / Gold)
Views SQL como camada semântica

---

📂 Estrutura do Repositório

O projeto segue uma separação clara entre ingestão, transformação, modelagem, persistência e consumo analítico, adotando princípios de arquitetura de dados utilizados em ambientes produtivos.

```text
b3-marketpulse/
│
├── app/                              # Camada de apresentação (Streamlit)
│   ├── Home.py                       # Entry point do Streamlit (menu principal)
│   └── Pages/
│       ├── Model_Compare.py          # Comparação de modelos (camada GOLD)
│       │                              # - Sem vs Com Sentimento
│       │                              # - Sharpe, Drawdown, Retorno
│       │                              # - Ranking e classificação de impacto
│       │
│       └── Equity_Curve.py           # Equity curve por ativo
│                                      # - Estratégia baseada em signal (0/1)
│                                      # - Comparação Base vs Sentimento
│                                      # - Retorno acumulado ao longo do tempo
│
├── pipelines/                        # Engenharia de Dados (ETL / Feature Engineering)
│   ├── prices_ingest.py              # Ingestão de preços históricos da B3
│   │                                  # - Fonte externa / API
│   │                                  # - Persistência em PostgreSQL (Bronze)
│   │
│   ├── sentiment_news.py             # Coleta e processamento de notícias financeiras
│   │                                  # - RSS / fontes públicas
│   │                                  # - Limpeza e normalização de texto
│   │                                  # - Persistência de sentimento bruto
│   │
│   ├── feature_engineering.py        # Criação de features quantitativas
│   │                                  # - Indicadores técnicos
│   │                                  # - Volatilidade, retornos, agregações
│   │                                  # - Consolidação Silver
│   │
│   └── silver_to_postgres.py          # Carga final da camada Silver
│                                      # - Escrita estruturada no banco
│                                      # - Padronização de schemas
│
├── ml/                               # Ciência de Dados e Machine Learning
│   ├── model_train_backtest.py       # Treino dos modelos e backtesting
│   │                                  # - Modelo base (sem sentimento)
│   │                                  # - Modelo com sentimento (NLP)
│   │                                  # - Geração de signals e métricas financeiras
│   │
│   ├── backtest_utils.py             # Funções auxiliares de backtest
│   │                                  # - Sharpe Ratio
│   │                                  # - Max Drawdown
│   │                                  # - Cumulative Return
│   │
│   └── feature_utils.py              # Funções reutilizáveis de features
│                                      # - Encapsula lógica de engenharia de dados
│
├── sql/                              # Camada semântica e regras de negócio (SQL)
│   ├── gold_model_decision.sql       # View GOLD de decisão por ativo
│   │                                  # - Calcula deltas entre modelos
│   │                                  # - Classifica impacto (risco-retorno)
│   │
│   ├── gold_trading_decision.sql     # View GOLD para consumo operacional
│   │                                  # - Une predictions, signals e decisão final
│   │
│   └── schema.sql                    # Criação inicial de tabelas (opcional)
│                                      # - Facilita reprodução do banco
│
├── docker/                           # Infra local (desenvolvimento)
│   └── docker-compose.yml            # PostgreSQL local via Docker
│                                      # - Usado apenas para desenvolvimento
│
├── requirements.txt                  # Dependências do projeto
│                                      # - Streamlit, ML, NLP, SQL, etc.
│
├── .gitignore                        # Arquivos ignorados pelo Git
│                                      # - .env, dumps, venv, dados sensíveis
│
├── README.md                         # Documentação principal do projeto
│                                      # - Objetivo, arquitetura, stack, uso
│
└── LICENSE (opcional)                # Licença do projeto
```

🚀 Como Executar Localmente (Resumo)

Subir banco

- docker-compose up -d

Rodar pipelines

- python pipelines/sentiment_news.py
- python pipelines/feature_engineering.py
- python ml/model_train_backtest.py

Abrir app

- streamlit run app/Home.py

🌍 Deploy (Cloud)

Banco: Neon
App: Streamlit Cloud

- Credenciais via st.secrets
- App consome apenas views GOLD

🔮 Próximos Passos

- Adicionar mais fontes de notícias
- Testar outros modelos (XGBoost, LSTM)
- Estratégias multi-ativos
- Alertas em tempo real
- Monitoramento de drift de sentimento

👤 Autor

Matheus Saraiva
Projeto desenvolvido para portfólio em Data Science, Machine Learning e Engenharia de Dados, com foco em aplicações reais no mercado financeiro.

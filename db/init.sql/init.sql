CREATE TABLE IF NOT EXISTS prices_daily (
    ticker TEXT,
    date DATE,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS features_daily (
    ticker TEXT,
    date DATE,
    ma_20 NUMERIC,
    ma_50 NUMERIC,
    volatility_20 NUMERIC,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS risk_scores (
    ticker TEXT,
    date DATE,
    risk_label TEXT,
    risk_score NUMERIC,
    PRIMARY KEY (ticker, date)
);

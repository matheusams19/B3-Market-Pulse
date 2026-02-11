import numpy as np
import pandas as pd

def cumulative_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    return float((1 + returns).prod() - 1)

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return float(drawdown.min())

def sharpe_ratio(
    returns: pd.Series,
    rf_daily: float = 0.0,
    periods_per_year: int = 252
) -> float:
    r = returns.dropna() - rf_daily
    if r.std() == 0:
        return 0.0
    return float((r.mean() / r.std()) * np.sqrt(periods_per_year))
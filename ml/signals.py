import pandas as pd

def ma_crossover_signal(df: pd.DataFrame) -> pd.Series:
    """
    df precisa ter colunas: date, ma_20, ma_50
    Retorna posição (0 ou 1) para cada data.
    """
    signal = (df["ma_20"] > df["ma_50"]).astype(int)

    # evita operar quando ainda não tem MAs suficientes
    signal = signal.where(df["ma_50"].notna(), 0)

    return signal

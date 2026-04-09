import pandas as pd


BASE_FEATURES = [
    "lag_1_order",
    "lag_7_order",
    "lag_30_order",
    "rolling_mean_7",
    "rolling_std_7",
    "holiday_flag",
    "promo_flag",
    "month",
    "day_of_week", 
    "quarter",
    "temperature",
    "rainfall_mm",
    "fuel_price_index",
    "economic_index",
]


def ensure_time_and_lags(df: pd.DataFrame, target_col: str = "order_volume") -> pd.DataFrame:
    """
    Si faltan columnas temporales/lags, las crea.
    Si ya existen en el CSV, las respeta.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("El DataFrame debe tener DatetimeIndex.")

    out = df.copy()

    # Calendario
    if "month" not in out.columns:
        out["month"] = out.index.month
    if "day_of_week" not in out.columns:
        out["day_of_week"] = out.index.dayofweek
    if "quarter" not in out.columns:
        out["quarter"] = out.index.quarter

    # Lags / rolling
    if target_col not in out.columns:
        raise ValueError(f"Target '{target_col}' no encontrado en el DataFrame.")

    if "lag_1_order" not in out.columns:
        out["lag_1_order"] = out[target_col].shift(1)
    if "lag_7_order" not in out.columns:
        out["lag_7_order"] = out[target_col].shift(7)
    if "lag_30_order" not in out.columns:
        out["lag_30_order"] = out[target_col].shift(30)

    if "rolling_mean_7" not in out.columns:
        out["rolling_mean_7"] = out[target_col].rolling(7).mean()
    if "rolling_std_7" not in out.columns:
        out["rolling_std_7"] = out[target_col].rolling(7).std()

    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Devuelve solo las features disponibles en el DataFrame.
    """
    feats = []
    for col in BASE_FEATURES:
        if col in df.columns:
            feats.append(col)
    return feats


def prepare_xy(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    dropna: bool = True,
):
    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' no encontrado.")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan features: {missing}")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    if dropna:
        mask = X.notna().all(axis=1) & y.notna()
        X = X.loc[mask]
        y = y.loc[mask]

    return X, y
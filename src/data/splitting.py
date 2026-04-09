import pandas as pd
from src.data.features import prepare_xy


def time_split_xy(
    df: pd.DataFrame,
    cutoff_date: str,
    target_col: str,
    feature_cols: list[str],
    dropna: bool = True,
):
    """
    Split temporal, con fecha formato YYYY-MM-DD:
    - train: fecha < cutoff_date
    - test:  fecha >= cutoff_date
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df debe tener DatetimeIndex")

    cutoff = pd.to_datetime(cutoff_date)
    train_df = df[df.index < cutoff]
    test_df = df[df.index >= cutoff]

    X_train, y_train = prepare_xy(train_df, target_col, feature_cols, dropna=dropna)
    X_test, y_test = prepare_xy(test_df, target_col, feature_cols, dropna=dropna)

    return X_train, X_test, y_train, y_test
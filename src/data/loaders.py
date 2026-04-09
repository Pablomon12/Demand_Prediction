from pathlib import Path
import pandas as pd

def load_logistics_dataset(csv_path:str) -> pd.DataFrame:
    """
    Carga el CSV, parsea `date`, ordena y deja el índice temporal.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo en: {csv_path}")
    df = pd.read_csv(csv_path)

    if "date" not in df.columns:
        raise ValueError("El CSV debe contener una columna 'date'.")

    df["date"] = pd.to_datetime(df["date"],errors="coerce")

    if df["date"].isna().any():
        raise ValueError("Hay fechas inválidas en la columna 'date'.")

    df = df.sort_values("date").set_index("date")
    df = df.dropna()
    df_completo = df.asfreq("D")

    if df.size != df_completo.size:
        raise ValueError("Hay fechas faltantes en el dataset.") 


    return df


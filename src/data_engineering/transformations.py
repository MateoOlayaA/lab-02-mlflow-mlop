# IMPORTACIONES
import pandas as pd
from src.data_engineering.extract_data import load_raw_data

from config.paths import (
    PROCESSED_DATA_PATH,
)

# ----------------------------------------------------------------------
# FUNCIÓN CLEAN
# ----------------------------------------------------------------------
def clean_column_names( df: pd.DataFrame ) -> pd.DataFrame:
    """
        Función encargada de transformar los nombres
        de las columnas que conserva el dataset
    """
    df = df.copy()
    
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return df

# ----------------------------------------------------------------------
# FUNCIÓN PREPROCESS
# ----------------------------------------------------------------------
def preprocess_data() -> pd.DataFrame:
    """
        Función que se encarga de cargar los datos
        y aplicar otras transformaciones al dataset
    """
    df = load_raw_data()
    df = clean_column_names(df)

    if "customerid" in df.columns:
        df = df.drop(columns=["customerid"])

    expected_columns = ["gender", "age", "annual_income_k$", "spending_score_1_100"]

    missing = [col for col in expected_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Faltan columnas esperadas: {missing}. Columnas encontradas: {list(df.columns)}"
        )

    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    if df["gender"].isnull().any():
        raise ValueError("La columna gender tiene valores diferentes a Male/Female.")

    # Guardando datos preprocesados en la carpeta (data/processed)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    return df
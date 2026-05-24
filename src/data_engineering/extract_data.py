# IMPORTACIONES
import pandas as pd

from config.paths import (
    RAW_DATA_PATH,
)

def load_raw_data() -> pd.DataFrame:
    """
        Función encargada de cargar el conjunto de datos
        desde la fuente data/raw
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo { RAW_DATA_PATH }. "
            "Coloca Mall_Customers.csv dentro de data/raw/"
        )

    return pd.read_csv( RAW_DATA_PATH )

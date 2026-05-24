# IMPORTACIONES
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "gender",
    "age",
    "annual_income_k$",
    "spending_score_1_100",
]

# ----------------------------------------------------------------------
# FUNCIÓN
# ----------------------------------------------------------------------
def get_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        Función encargada de seleccionar una lista de características
        que conserva el dataset
    """
    return df[FEATURE_COLUMNS].copy()

# ----------------------------------------------------------------------
# FUNCIÓN
# ----------------------------------------------------------------------
def scale_features(X: pd.DataFrame):
    """
        Función encargada de aplicar la técnica de
        estándarización a los datos del dataset
    """

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler
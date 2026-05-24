import mlflow
import mlflow.pyfunc
import pandas as pd

from config.paths import (
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
)

# ----------------------------------------------------------------------
# FUNCIÓN
# ----------------------------------------------------------------------
def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"

    return mlflow.pyfunc.load_model(model_uri)

# ----------------------------------------------------------------------
# FUNCIÓN
# ----------------------------------------------------------------------
def predict_customer():
    model = load_model()

    input_df = pd.DataFrame([{
        "gender": 1,
        "age": 35,
        "annual_income_k$": 120,
        "spending_score_1_100": 50,
    }])

    prediction = model.predict(input_df)

    print("Segmento predicho:", int(prediction[0]))
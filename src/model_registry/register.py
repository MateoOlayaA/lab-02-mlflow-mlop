# IMPORTACIONES
import mlflow
from mlflow.tracking import MlflowClient

from config.paths import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
)

# ----------------------------------------------------------------------
# FUNCIÓN
# ----------------------------------------------------------------------
def register_best_model():

    # Indicando a mlflow que se conecte al servidor local de seguimiento
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Creando cliente de administración para mflow
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Buscando experimento guardado en el servidor de seguimiento local
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError("No existe el experimento")

    # Organizando (mayor a menor) los modelos que tiene el experimento
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1 DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No hay runs disponibles")

    # Obteniendo el mejor modelo del experimento
    best_run = runs[0]
    run_id   = best_run.info.run_id

    # Obteniendo ID del experimento que hay en el servidor de seguimiento
    model_uri = f"runs:/{run_id}/model"

    # Registrando el mejor modelo del experimento en el "model registry"
    result = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    # Asignando alias al modelo registrado en el "model registry"
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=result.version
    )

    print("Modelo registrado correctamente")
    print("Run ID:", run_id)
    print("Modelo:", REGISTERED_MODEL_NAME)
    print("Versión:", result.version)
    print("Alias: champion")
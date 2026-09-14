from dataclasses import dataclass
from pathlib import Path


# Este registro es la única fuente de nombres de archivo de los modelos disponibles.
MODEL_REGISTRY = {
    "LightGBM": "LightGBM_Final_V2.pkl",
    "Random Forest": "RandomForest_Final_V2.pkl",
    "XGBoost": "XGBoost_Final_V2.pkl",
    "Decision Tree": "DecisionTree_Final_V2.pkl",
}


# Desde src/config/settings.py, dos niveles hacia arriba corresponde a la raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Rutas y parámetros globales inmutables de la aplicación."""

    app_name: str = "ML Model Deployment"
    model_path: Path = PROJECT_ROOT / "artifacts" / "model.joblib"
    # Los artefactos se mantienen dentro del repositorio, no en rutas absolutas del equipo.
    models_dir: Path = PROJECT_ROOT / "src" / "models"


settings = Settings()

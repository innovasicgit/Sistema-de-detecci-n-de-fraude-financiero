from dataclasses import dataclass
from pathlib import Path


MODEL_REGISTRY = {
    "LightGBM": "LightGBM_Final_V2.pkl",
    "Random Forest": "RandomForest_Final_V2.pkl",
    "XGBoost": "XGBoost_Final_V2.pkl",
    "Decision Tree": "DecisionTree_Final_V2.pkl",
}


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    app_name: str = "ML Model Deployment"
    model_path: Path = PROJECT_ROOT / "artifacts" / "model.joblib"
    models_dir: Path = PROJECT_ROOT / "src" / "models"


settings = Settings()

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    app_name: str = "ML Model Deployment"
    model_path: Path = PROJECT_ROOT / "artifacts" / "model.joblib"


settings = Settings()

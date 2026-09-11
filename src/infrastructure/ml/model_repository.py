from functools import lru_cache

from src.config.settings import MODEL_REGISTRY, settings
from src.infrastructure.ml.joblib_model_loader import JoblibModelLoader


class ModelRepository:
    """Resuelve y cachea los pipelines entrenados por nombre visible."""

    def load(self, model_name: str):
        return _load_model(model_name)


@lru_cache(maxsize=None)
def _load_model(model_name: str):
    filename = MODEL_REGISTRY.get(model_name)
    if filename is None:
        raise ValueError(f"Modelo no disponible: {model_name}")
    return JoblibModelLoader(settings.models_dir / filename).load()
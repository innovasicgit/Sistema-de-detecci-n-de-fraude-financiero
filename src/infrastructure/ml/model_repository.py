from functools import lru_cache

from src.config.settings import MODEL_REGISTRY, settings
from src.infrastructure.ml.joblib_model_loader import JoblibModelLoader


class ModelRepository:
    """Resuelve y cachea los pipelines entrenados por nombre visible."""

    def load(self, model_name: str):
        # El caché real está en la función global para sobrevivir a nuevos objetos
        # ModelRepository creados durante los reruns de Streamlit.
        return _load_model(model_name)


@lru_cache(maxsize=None)
def _load_model(model_name: str):
    # Primero se traduce el nombre mostrado al usuario a un archivo físico.
    filename = MODEL_REGISTRY.get(model_name)
    if filename is None:
        raise ValueError(f"Modelo no disponible: {model_name}")
    # lru_cache evita deserializar varias veces el mismo pipeline pesado.
    return JoblibModelLoader(settings.models_dir / filename).load()
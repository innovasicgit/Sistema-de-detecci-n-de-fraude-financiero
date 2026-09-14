from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.domain.entities.analysis import AnalysisResult, ModelEvaluation
from src.domain.ports.batch_model import BatchModelLoader
from src.application.fraud_analysis_service import DatasetValidation, FraudAnalysisService


@dataclass(frozen=True)
class AnalyzeDatasetInput:
    """Datos mínimos que necesita el caso de uso para procesar un archivo."""

    dataframe: pd.DataFrame
    model_name: str


class AnalyzeDatasetUseCase:
    """Orquesta loaders y servicios sin depender directamente de Streamlit."""

    def __init__(self, model_loader: BatchModelLoader, service: FraudAnalysisService | None = None) -> None:
        self._model_loader = model_loader
        self._service = service or FraudAnalysisService()

    def validate(self, dataframe: pd.DataFrame, model_name: str) -> DatasetValidation:
        # La validación se ejecuta al seleccionar/cambiar el modelo, antes de permitir analizar.
        return self._service.validate_columns(dataframe, self._model_loader.load(model_name))

    def execute(self, request: AnalyzeDatasetInput) -> AnalysisResult:
        # Este método es el flujo de predicción operativa para archivos sin necesidad de etiqueta.
        if request.dataframe.empty:
            raise ValueError("El archivo no contiene registros para analizar.")

        model = self._model_loader.load(request.model_name)
        if hasattr(model, "feature_names_in_"):
            # Los pipelines reales reciben un DataFrame con sus nombres y orden originales.
            return self._service.analyze(dataframe=request.dataframe, model=model, model_name=request.model_name)

        # Este camino conserva compatibilidad con el modelo demo usado por pruebas antiguas.
        predictions = model.predict(request.dataframe.to_dict(orient="records"))
        records = self._build_records(request.dataframe, predictions)
        normal = int((records["resultado"] == "Normal").sum())
        total = len(records)

        return AnalysisResult(
            records=records,
            total=total,
            normal=normal,
            alerts=total - normal,
            model_name=request.model_name,
        )

    def evaluate(self, dataframe: pd.DataFrame, model_name: str) -> ModelEvaluation:
        # La evaluación requiere y_true (misstate); por eso es un flujo separado de execute().
        if dataframe.empty:
            raise ValueError("El archivo no contiene registros para evaluar.")
        return self._service.evaluate(dataframe, self._model_loader.load(model_name))

    @staticmethod
    def _build_records(dataframe: pd.DataFrame, predictions: Any) -> pd.DataFrame:
        # Este adaptador convierte predicciones del modelo demo en la estructura de resultados.
        records = pd.DataFrame(
            {
                "registro": range(1, len(dataframe) + 1),
                "tipo_comprobante": dataframe.apply(_find_document_type, axis=1),
                "clasificacion": list(predictions),
            }
        )
        records["resultado"] = records["clasificacion"].map(
            lambda value: "Normal" if value == "Normal" else "Alerta"
        )
        return records


def _find_document_type(row: pd.Series) -> str:
    # Los CSV pueden usar distintos nombres para el tipo de comprobante.
    for column in ("Tipo de comprobante", "tipo_comprobante", "tipo", "Tipo"):
        if column in row.index and pd.notna(row[column]):
            return str(row[column])
    return "N/D"

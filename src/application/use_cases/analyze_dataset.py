from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.domain.entities.analysis import AnalysisResult
from src.domain.ports.batch_model import BatchModelLoader


@dataclass(frozen=True)
class AnalyzeDatasetInput:
    dataframe: pd.DataFrame
    model_name: str


class AnalyzeDatasetUseCase:
    def __init__(self, model_loader: BatchModelLoader) -> None:
        self._model_loader = model_loader

    def execute(self, request: AnalyzeDatasetInput) -> AnalysisResult:
        if request.dataframe.empty:
            raise ValueError("El archivo no contiene registros para analizar.")

        model = self._model_loader.load()
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

    @staticmethod
    def _build_records(dataframe: pd.DataFrame, predictions: Any) -> pd.DataFrame:
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
    for column in ("Tipo de comprobante", "tipo_comprobante", "tipo", "Tipo"):
        if column in row.index and pd.notna(row[column]):
            return str(row[column])
    return "N/D"

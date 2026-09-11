from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import pandas as pd

from src.domain.entities.analysis import FRAUD_LABELS


LOGGER = logging.getLogger(__name__)


PERCENTAGE_FEATURES = {
    "Margen_Bruto_pct",
    "Margen_Operacional_pct",
    "Margen_Neto_pct",
    "ROA_pct",
    "ROE_pct",
    "Endeudamiento_pct",
    "CxC_Ventas_pct",
    "Inventarios_Ventas_pct",
    "Gastos_Operacionales_Ventas_pct",
    "Costos_Ventas_pct",
    "Pasivo_Corriente_Activo_pct",
    "Pasivo_No_Corriente_Activo_pct",
    "Pasivo_Corriente_Pasivo_Total_pct",
}


@dataclass(frozen=True)
class DatasetValidation:
    expected_features: tuple[str, ...]
    missing_features: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.missing_features


class FraudAnalysisService:
    def validate_columns(self, dataframe: pd.DataFrame, model: Any) -> DatasetValidation:
        expected_features = tuple(str(feature) for feature in getattr(model, "feature_names_in_", ()))
        if not expected_features:
            raise ValueError("El modelo no informa las variables requeridas.")
        missing_features = tuple(
            feature for feature in expected_features if feature not in dataframe.columns
        )
        return DatasetValidation(expected_features, missing_features)

    def analyze(self, dataframe: pd.DataFrame, model: Any, model_name: str) -> Any:
        if dataframe.empty:
            raise ValueError("El archivo no contiene registros para analizar.")

        validation = self.validate_columns(dataframe, model)
        if not validation.is_valid:
            missing = ", ".join(validation.missing_features)
            raise ValueError(f"El archivo no contiene todas las variables requeridas: {missing}")

        X_predict = self._prepare_features(dataframe, validation.expected_features)
        if X_predict.isna().any().any():
            raise ValueError("El archivo contiene valores vacíos en las variables requeridas.")
        LOGGER.info(
            "Matriz de inferencia: shape=%s, dtypes=%s, finitos=%s",
            X_predict.shape,
            X_predict.dtypes.astype(str).unique().tolist(),
            bool(np.isfinite(X_predict.to_numpy(dtype=float)).all()),
        )

        try:
            classes = list(model.classes_)
            predictions, probabilities = self._predict(model, X_predict, classes)
        except Exception as error:
            LOGGER.exception("Error durante la inferencia del modelo %s", model_name)
            raise ValueError(
                "No fue posible analizar los datos. Revise que sus valores tengan el formato esperado."
            ) from error

        records = dataframe.copy()
        records.insert(0, "indice_original", list(dataframe.index))
        records["registro"] = list(dataframe.index)
        records["prediccion_codigo"] = list(predictions)
        records["clasificacion"] = [self._label_for(prediction) for prediction in predictions]
        records["probabilidad"] = [
            float(dict(zip(classes, row))[prediction]) for prediction, row in zip(predictions, probabilities)
        ]
        records["resultado"] = records["clasificacion"].map(
            lambda classification: "Normal" if classification == "Normal" else "Alerta"
        )
        normal = int((records["resultado"] == "Normal").sum())
        return self._analysis_result(records, len(records), normal, model_name)

    @staticmethod
    def _predict(model: Any, features: pd.DataFrame, classes: list[Any]) -> tuple[Any, Any]:
        classifier = model.steps[-1][1] if hasattr(model, "steps") else model
        if classifier.__class__.__module__.startswith("lightgbm") and hasattr(classifier, "booster_"):
            probabilities = classifier.booster_.predict(features.to_numpy(dtype="float64"))
            predictions = [classes[index] for index in probabilities.argmax(axis=1)]
            return predictions, probabilities

        return model.predict(features), model.predict_proba(features)

    @staticmethod
    def _prepare_features(dataframe: pd.DataFrame, expected_features: tuple[str, ...]) -> pd.DataFrame:
        features = dataframe.loc[:, list(expected_features)].copy()
        invalid_columns = []

        for column in expected_features:
            original = features[column]
            values = original.astype("string").str.strip()
            has_percentage = values.str.contains("%", regex=False, na=False)
            normalized = values.str.replace("%", "", regex=False)
            both_separators = normalized.str.contains(",", na=False) & normalized.str.contains(".", regex=False, na=False)
            normalized.loc[both_separators] = (
                normalized.loc[both_separators]
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            normalized = normalized.str.replace(",", ".", regex=False)
            converted = pd.to_numeric(normalized, errors="coerce")
            converted = converted.where(~has_percentage, converted / 100)

            if column in PERCENTAGE_FEATURES:
                converted = converted.where(converted.abs() <= 1, converted / 100)

            newly_invalid = converted.isna() & original.notna()
            if newly_invalid.any():
                invalid_columns.append(column)
            features[column] = converted

        if invalid_columns:
            columns = ", ".join(invalid_columns)
            raise ValueError(f"Hay valores no numéricos en las variables: {columns}")
        return features

    @staticmethod
    def _label_for(prediction: Any) -> str:
        try:
            return FRAUD_LABELS.get(int(prediction), str(prediction))
        except (TypeError, ValueError):
            return str(prediction)

    @staticmethod
    def _analysis_result(records: pd.DataFrame, total: int, normal: int, model_name: str) -> Any:
        from src.domain.entities.analysis import AnalysisResult

        return AnalysisResult(records, total, normal, total - normal, model_name)
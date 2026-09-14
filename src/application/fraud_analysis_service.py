from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.domain.entities.analysis import FRAUD_LABELS, ModelEvaluation


LOGGER = logging.getLogger(__name__)


PERCENTAGE_FEATURES = {
    # Estas columnas representan razones o porcentajes que el entrenamiento guarda en escala decimal.
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
    """Resultado de comprobar las columnas del CSV contra el contrato del modelo."""

    expected_features: tuple[str, ...]
    missing_features: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.missing_features


class FraudAnalysisService:
    """Contiene la lógica de inferencia y evaluación independiente de Streamlit."""

    def evaluate(
        self, dataframe: pd.DataFrame, model: Any, target_column: str = "misstate"
    ) -> ModelEvaluation:
        # La etiqueta real solo se usa para medir el modelo; nunca se envía como entrada.
        if target_column not in dataframe.columns:
            raise ValueError(f"El archivo debe contener la columna real '{target_column}'.")

        y_true = pd.to_numeric(dataframe[target_column], errors="coerce")
        if y_true.isna().any():
            raise ValueError(f"La columna '{target_column}' contiene etiquetas no válidas.")

        # El modelo sigue siendo la fuente de verdad para las 19 variables de entrada.
        validation = self.validate_columns(dataframe, model)
        if not validation.is_valid:
            missing = ", ".join(validation.missing_features)
            raise ValueError(f"Faltan variables requeridas para evaluar: {missing}")

        # Se reutiliza la misma normalización que en la predicción operativa.
        features = self._prepare_features(dataframe, validation.expected_features)
        if features.isna().any().any():
            raise ValueError("Las variables requeridas contienen valores vacíos.")
        classes = [int(value) for value in model.classes_]
        predictions, _ = self._predict(model, features, list(model.classes_))
        predictions = [int(value) for value in predictions]
        labels = tuple(sorted(set(classes) | set(y_true.astype(int))))
        # Se incluyen todas las clases conocidas o presentes para que los soportes cero
        # queden visibles y no se confundan con una ausencia de cálculo.
        report = classification_report(
            y_true.astype(int), predictions, labels=list(labels),
            target_names=[FRAUD_LABELS.get(label, str(label)) for label in labels],
            output_dict=True, zero_division=0,
        )
        report["accuracy"] = float(accuracy_score(y_true.astype(int), predictions))
        return ModelEvaluation(
            report=report,
            confusion_matrix=confusion_matrix(y_true.astype(int), predictions, labels=list(labels)),
            labels=labels,
            total=len(dataframe),
        )

    def validate_columns(self, dataframe: pd.DataFrame, model: Any) -> DatasetValidation:
        # feature_names_in_ conserva tanto los nombres como el orden usado en entrenamiento.
        expected_features = tuple(str(feature) for feature in getattr(model, "feature_names_in_", ()))
        if not expected_features:
            raise ValueError("El modelo no informa las variables requeridas.")
        missing_features = tuple(
            feature for feature in expected_features if feature not in dataframe.columns
        )
        return DatasetValidation(expected_features, missing_features)

    def analyze(self, dataframe: pd.DataFrame, model: Any, model_name: str) -> Any:
        """Ejecuta inferencia sobre datos nuevos y construye el resultado para la UI."""
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
            # predict y predict_proba se ejecutan juntos para conservar la probabilidad
            # correspondiente a la clase que realmente predijo el modelo.
            classes = list(model.classes_)
            predictions, probabilities = self._predict(model, X_predict, classes)
        except Exception as error:
            LOGGER.exception("Error durante la inferencia del modelo %s", model_name)
            raise ValueError(
                "No fue posible analizar los datos. Revise que sus valores tengan el formato esperado."
            ) from error

        # Se conserva el CSV original y solo se agregan columnas de resultado.
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
        # LightGBM necesita una ruta nativa en Windows; los demás modelos usan el API estándar.
        classifier = model.steps[-1][1] if hasattr(model, "steps") else model
        if classifier.__class__.__module__.startswith("lightgbm") and hasattr(classifier, "booster_"):
            probabilities = classifier.booster_.predict(features.to_numpy(dtype="float64"))
            predictions = [classes[index] for index in probabilities.argmax(axis=1)]
            return predictions, probabilities

        return model.predict(features), model.predict_proba(features)

    @staticmethod
    def _prepare_features(dataframe: pd.DataFrame, expected_features: tuple[str, ...]) -> pd.DataFrame:
        """Normaliza solo las columnas requeridas, evitando recorrer todo el CSV."""
        features = dataframe.loc[:, list(expected_features)].copy()
        invalid_columns = []

        for column in expected_features:
            # Cada columna se convierte de forma vectorizada para soportar archivos grandes.
            original = features[column]
            values = original.astype("string").str.strip()
            has_percentage = values.str.contains("%", regex=False, na=False)
            normalized = values.str.replace("%", "", regex=False)
            # Si aparecen punto y coma decimal, se interpreta el punto como separador de miles.
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
                # Acepta tanto 0.25 como 25 para representar el mismo 25%.
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
import pandas as pd
import pytest

from src.application.fraud_analysis_service import FraudAnalysisService


class FakeModel:
    # El orden deliberadamente invertido comprueba que el servicio respeta
    # feature_names_in_ y no el orden accidental del CSV.
    feature_names_in_ = ["segunda", "primera"]
    classes_ = [0, 1, 2]

    def __init__(self):
        self.received_columns = None

    def predict(self, dataframe):
        self.received_columns = list(dataframe.columns)
        return [2, 0]

    def predict_proba(self, dataframe):
        return [[0.1, 0.2, 0.7], [0.4, 0.35, 0.25]]


def test_service_uses_model_features_in_order_and_probability_for_predicted_class():
    # Las columnas extra se conservan, pero no entran al modelo.
    model = FakeModel()
    dataframe = pd.DataFrame({"extra": [9, 8], "primera": [1, 2], "segunda": [3, 4]})

    result = FraudAnalysisService().analyze(dataframe, model, "LightGBM")

    assert model.received_columns == ["segunda", "primera"]
    assert result.records["prediccion_codigo"].tolist() == [2, 0]
    assert result.records["clasificacion"].tolist() == ["Redondeo de cifras", "Normal"]
    assert result.records["probabilidad"].tolist() == [0.7, 0.4]
    assert result.records["resultado"].tolist() == ["Alerta", "Normal"]
    assert "extra" in result.records.columns


def test_service_reports_missing_features():
    # Un archivo incompleto debe detenerse antes de ejecutar inferencia.
    dataframe = pd.DataFrame({"primera": [1]})

    validation = FraudAnalysisService().validate_columns(dataframe, FakeModel())

    assert validation.expected_features == ("segunda", "primera")
    assert validation.missing_features == ("segunda",)
    assert not validation.is_valid


def test_service_rejects_empty_values_in_required_features():
    # Los valores vacíos en variables requeridas no se convierten silenciosamente.
    dataframe = pd.DataFrame({"primera": [1], "segunda": [None]})

    with pytest.raises(ValueError, match="valores vacíos"):
        FraudAnalysisService().analyze(dataframe, FakeModel(), "LightGBM")


def test_service_evaluates_predictions_against_misstate():
    # misstate es y_true; la predicción usa solo las variables esperadas.
    dataframe = pd.DataFrame(
        {"segunda": [3, 4], "primera": [1, 2], "misstate": [2, 0]}
    )

    evaluation = FraudAnalysisService().evaluate(dataframe, FakeModel())

    assert evaluation.total == 2
    assert evaluation.labels == (0, 1, 2)
    assert evaluation.report["accuracy"] == 1.0
    assert evaluation.report["Redondeo de cifras"]["recall"] == 1.0
    assert evaluation.confusion_matrix.tolist() == [[1, 0, 0], [0, 0, 0], [0, 0, 1]]


def test_service_normalizes_regional_numbers_and_percentages():
    # Se validan formatos habituales de Excel: porcentajes y coma decimal.
    dataframe = pd.DataFrame(
        {
            "Margen_Bruto_pct": ["25%"],
            "saldo 1330": ["1,13E+10"],
        }
    )

    prepared = FraudAnalysisService._prepare_features(
        dataframe, ("Margen_Bruto_pct", "saldo 1330")
    )

    assert prepared.loc[0, "Margen_Bruto_pct"] == 0.25
    assert prepared.loc[0, "saldo 1330"] == 1.13e10


def test_service_keeps_decimal_percentage_without_percent_sign():
    # Un porcentaje ya expresado como 0.25 permanece en escala decimal.
    dataframe = pd.DataFrame({"ROA_pct": [0.25]})

    prepared = FraudAnalysisService._prepare_features(dataframe, ("ROA_pct",))

    assert prepared.loc[0, "ROA_pct"] == 0.25
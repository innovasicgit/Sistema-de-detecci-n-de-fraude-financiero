import pandas as pd
import pytest

from src.application.fraud_analysis_service import FraudAnalysisService


class FakeModel:
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
    dataframe = pd.DataFrame({"primera": [1]})

    validation = FraudAnalysisService().validate_columns(dataframe, FakeModel())

    assert validation.expected_features == ("segunda", "primera")
    assert validation.missing_features == ("segunda",)
    assert not validation.is_valid


def test_service_rejects_empty_values_in_required_features():
    dataframe = pd.DataFrame({"primera": [1], "segunda": [None]})

    with pytest.raises(ValueError, match="valores vacíos"):
        FraudAnalysisService().analyze(dataframe, FakeModel(), "LightGBM")


def test_service_normalizes_regional_numbers_and_percentages():
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
    dataframe = pd.DataFrame({"ROA_pct": [0.25]})

    prepared = FraudAnalysisService._prepare_features(dataframe, ("ROA_pct",))

    assert prepared.loc[0, "ROA_pct"] == 0.25
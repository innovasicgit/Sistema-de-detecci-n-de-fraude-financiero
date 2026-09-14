import pandas as pd

from src.application.use_cases.analyze_dataset import AnalyzeDatasetInput, AnalyzeDatasetUseCase
from src.infrastructure.ml.demo_batch_model import DemoBatchModelLoader


def test_analyze_dataset_returns_metrics_and_records():
    # Arrange: el modelo demo produce clases deterministas para tres registros.
    dataframe = pd.DataFrame(
        {
            "Tipo de comprobante": ["CC1", "FV", "NM"],
            "Monto": [100, 200, 300],
        }
    )

    # Act: se ejecuta el caso de uso como lo haría la capa de presentación.
    result = AnalyzeDatasetUseCase(DemoBatchModelLoader()).execute(
        AnalyzeDatasetInput(dataframe, "LightGBM")
    )

    # Assert: se verifica el resumen y la extracción del tipo de comprobante.
    assert result.total == 3
    assert result.normal == 1
    assert result.alerts == 2
    assert result.records["tipo_comprobante"].tolist() == ["CC1", "FV", "NM"]

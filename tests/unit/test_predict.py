from src.application.use_cases.predict import PredictInput, PredictUseCase


class FakeModel:
    # Doble de prueba: devuelve cuántas características recibió.
    def predict(self, features):
        return [len(features[0])]


class FakeLoader:
    # El loader falso aísla el caso de uso del sistema de archivos y joblib.
    def load(self):
        return FakeModel()


def test_predict_use_case_delegates_to_model():
    # La prueba confirma la delegación y el empaquetado del resultado de dominio.
    result = PredictUseCase(FakeLoader()).execute(PredictInput((1.0, 2.0)))

    assert result.value == 2

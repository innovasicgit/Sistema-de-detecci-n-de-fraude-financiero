from src.application.use_cases.predict import PredictInput, PredictUseCase


class FakeModel:
    def predict(self, features):
        return [len(features[0])]


class FakeLoader:
    def load(self):
        return FakeModel()


def test_predict_use_case_delegates_to_model():
    result = PredictUseCase(FakeLoader()).execute(PredictInput((1.0, 2.0)))

    assert result.value == 2

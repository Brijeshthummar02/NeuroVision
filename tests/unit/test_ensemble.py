import numpy as np


class DummyModel:
    def __init__(self, pred):
        self.pred = np.array([pred], dtype=np.float32)

    def predict(self, img, verbose=0):
        return self.pred


def test_ensemble_classification_weighted_average(app_state):
    model_primary = DummyModel([0.2, 0.8])
    model_other = DummyModel([0.6, 0.4])

    app_state.classification_models = [
        ("ResNet-50", model_primary),
        ("Other", model_other),
    ]
    app_state.app.config["USE_ENSEMBLE"] = True

    result = app_state.ensemble_classification_predict(
        np.zeros((1, 256, 256, 3), dtype=np.float32),
        use_tta=False,
    )

    expected = (2.0 / 3.0) * model_primary.pred + (1.0 / 3.0) * model_other.pred
    assert np.allclose(result, expected)


class DummySegModel:
    def __init__(self, pred):
        self.pred = pred

    def predict(self, img, verbose=0):
        return self.pred


def test_ensemble_segmentation_average(app_state):
    pred_a = np.ones((1, 4, 4, 1), dtype=np.float32)
    pred_b = np.zeros((1, 4, 4, 1), dtype=np.float32)

    app_state.segmentation_models = [
        ("ResUNet-MRI", DummySegModel(pred_a)),
        ("Alt", DummySegModel(pred_b)),
    ]
    app_state.app.config["USE_ENSEMBLE"] = True

    result = app_state.ensemble_segmentation_predict(
        np.zeros((1, 4, 4, 3), dtype=np.float32),
        use_tta=False,
    )

    assert np.allclose(result, 0.5)

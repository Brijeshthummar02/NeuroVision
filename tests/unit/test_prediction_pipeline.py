import numpy as np


def test_predict_tumor_has_tumor_path(app_state):
    class DummyClassifier:
        def predict(self, img, verbose=0):
            return np.array([[0.1, 0.9]], dtype=np.float32)

    class DummySegmenter:
        def __init__(self, mask):
            self.mask = mask

        def predict(self, img, verbose=0):
            return self.mask

    mask = np.zeros((1, 256, 256, 1), dtype=np.float32)
    mask[0, 10:20, 10:20, 0] = 1.0

    app_state.classification_models = [("ResNet-50", DummyClassifier())]
    app_state.segmentation_models = [("ResUNet", DummySegmenter(mask))]
    app_state.app.config["USE_TTA"] = False
    app_state.app.config["USE_ENSEMBLE"] = False

    img = np.zeros((256, 256, 3), dtype=np.uint8)
    result = app_state.predict_tumor("unused.png", img_original=img, use_tta=False)

    assert result["has_tumor"] is True
    assert 0.0 <= result["confidence"] <= 1.0
    assert "segmentation" in result
    assert result["segmentation"]["tumor_pixels"] > 0
    assert result["segmentation"]["mask"].startswith("data:image/png;base64,")
    assert result["severity_assessment"]["level"] == "Minimal"


def test_predict_tumor_no_tumor_path(app_state):
    class DummyClassifier:
        def predict(self, img, verbose=0):
            return np.array([[0.9, 0.1]], dtype=np.float32)

    app_state.classification_models = [("ResNet-50", DummyClassifier())]
    app_state.segmentation_models = []
    app_state.app.config["USE_TTA"] = False
    app_state.app.config["USE_ENSEMBLE"] = False

    img = np.zeros((256, 256, 3), dtype=np.uint8)
    result = app_state.predict_tumor("unused.png", img_original=img, use_tta=False)

    assert result["has_tumor"] is False
    assert "segmentation" not in result

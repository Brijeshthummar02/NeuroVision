import numpy as np

from utilities import predict_with_tta_classification, predict_with_tta_segmentation


class RecordingModel:
    def __init__(self):
        self.inputs = []
        self.counter = 0

    def predict(self, img):
        self.inputs.append(img.copy())
        self.counter += 1
        return np.array([[self.counter, self.counter + 1]], dtype=np.float32)


def test_predict_with_tta_classification_averages_and_flips():
    model = RecordingModel()
    img = np.arange(6, dtype=np.float32).reshape(1, 2, 3, 1)

    result = predict_with_tta_classification(model, img)

    assert len(model.inputs) == 3
    assert np.array_equal(model.inputs[1], np.flip(model.inputs[0], axis=2))
    assert np.array_equal(model.inputs[2], np.flip(model.inputs[0], axis=1))

    expected = np.array([[2.0, 3.0]], dtype=np.float32)
    assert np.allclose(result, expected)


class PassThroughModel:
    def predict(self, img):
        return img[..., :1]


def test_predict_with_tta_segmentation_restores_orientation():
    np.random.seed(0)
    model = PassThroughModel()
    img = np.random.rand(1, 4, 5, 1).astype(np.float32)

    result = predict_with_tta_segmentation(model, img)

    assert np.allclose(result, img)

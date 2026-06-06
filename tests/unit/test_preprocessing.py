import numpy as np


def test_preprocess_image_classification_rgb(app_state, sample_image_array):
    output = app_state.preprocess_image_classification(sample_image_array.copy())

    assert output.shape == (1, 256, 256, 3)

    expected = np.array([30, 20, 10], dtype=np.float32) / 255.0
    assert np.allclose(output[0, 0, 0], expected, atol=1e-3)
    assert 0.0 <= output.min() <= output.max() <= 1.0


def test_preprocess_image_classification_grayscale(app_state):
    gray = np.full((10, 10), 128, dtype=np.uint8)
    output = app_state.preprocess_image_classification(gray)

    assert output.shape == (1, 256, 256, 3)
    assert np.allclose(output[0, 0, 0], 128 / 255.0, atol=1e-3)


def test_preprocess_image_segmentation_standardization(app_state):
    gray = np.full((10, 10), 50, dtype=np.uint8)
    output = app_state.preprocess_image_segmentation(gray)

    assert output.shape == (1, 256, 256, 3)
    assert np.allclose(output.mean(), 0.0, atol=1e-6)
    assert np.allclose(output.std(), 0.0, atol=1e-6)

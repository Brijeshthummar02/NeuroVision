import numpy as np

from robustness.corruptions import (
    add_gaussian_noise,
    apply_gaussian_blur,
    adjust_brightness,
    reduce_resolution,
    add_compression_artifacts,
)


def test_corruptions_preserve_shape_and_type():
    np.random.seed(0)
    image = np.full((32, 32, 3), 120, dtype=np.uint8)

    noisy = add_gaussian_noise(image)
    assert noisy.shape == image.shape
    assert noisy.dtype == np.uint8
    assert noisy.min() >= 0 and noisy.max() <= 255
    assert np.any(noisy != image)

    blurred = apply_gaussian_blur(image)
    assert blurred.shape == image.shape
    assert blurred.dtype == np.uint8

    bright = adjust_brightness(image, factor=1.2)
    assert bright.shape == image.shape
    assert bright.dtype == np.uint8
    assert bright.mean() >= image.mean()

    lowres = reduce_resolution(image, scale=0.5)
    assert lowres.shape == image.shape

    compressed = add_compression_artifacts(image, quality=30)
    assert compressed.shape == image.shape
    assert compressed.dtype == np.uint8

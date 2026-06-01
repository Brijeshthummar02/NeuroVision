import cv2
import numpy as np


def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)

    noisy_image = image.astype(np.float32) + noise

    noisy_image = np.clip(noisy_image, 0, 255)

    return noisy_image.astype(np.uint8)


def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)


def adjust_brightness(image, factor=1.2):
    bright_image = image.astype(np.float32) * factor

    bright_image = np.clip(bright_image, 0, 255)

    return bright_image.astype(np.uint8)


def reduce_resolution(image, scale=0.5):
    height, width = image.shape[:2]

    resized = cv2.resize(
        image,
        (int(width * scale), int(height * scale))
    )

    restored = cv2.resize(
        resized,
        (width, height)
    )

    return restored


def add_compression_artifacts(image, quality=20):
    encode_param = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        quality
    ]

    _, encoded_img = cv2.imencode('.jpg', image, encode_param)

    decoded_img = cv2.imdecode(encoded_img, 1)

    return decoded_img
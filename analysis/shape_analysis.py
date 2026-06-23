import cv2
import numpy as np


def get_spread_pattern(contour):
    x, y, w, h = cv2.boundingRect(contour)

    ratio = w / max(h, 1)

    if ratio > 1.5:
        return "Horizontally Spread"
    elif ratio < 0.67:
        return "Vertically Spread"
    return "Localized"


def analyze_shape(mask_binary):
    contours, _ = cv2.findContours(
        mask_binary.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return {
            "shape_complexity": 0.0,
            "spread_pattern": "None"
        }

    largest = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)

    complexity = (
        (perimeter ** 2) / (4 * np.pi * area)
        if area > 0 else 0
    )

    return {
        "shape_complexity": round(complexity, 3),
        "spread_pattern": get_spread_pattern(largest)
    }
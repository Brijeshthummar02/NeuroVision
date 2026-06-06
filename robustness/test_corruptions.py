from pathlib import Path
import cv2

from corruptions import (
    add_gaussian_noise,
    apply_gaussian_blur,
    adjust_brightness,
    reduce_resolution,
    add_compression_artifacts
)

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load sample MRI image
image_path = BASE_DIR / "images" / "segmentation_predictions.png"

image = cv2.imread(str(image_path))

if image is None:
    raise ValueError(f"Could not load image: {image_path}")

# Create output folder
output_dir = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

# Apply corruptions
noise_image = add_gaussian_noise(image)

blur_image = apply_gaussian_blur(image)

bright_image = adjust_brightness(image)

lowres_image = reduce_resolution(image)

compressed_image = add_compression_artifacts(image)

# Save outputs
cv2.imwrite(str(output_dir / "gaussian_noise.jpg"), noise_image)

cv2.imwrite(str(output_dir / "blur.jpg"), blur_image)

cv2.imwrite(str(output_dir / "brightness.jpg"), bright_image)

cv2.imwrite(str(output_dir / "low_resolution.jpg"), lowres_image)

cv2.imwrite(str(output_dir / "compression.jpg"), compressed_image)

print("All corrupted MRI images generated successfully.")
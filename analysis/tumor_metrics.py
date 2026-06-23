import numpy as np

def calculate_tumor_metrics(mask_binary):
    total_pixels = mask_binary.shape[0]*mask_binary.shape[1]
    tumor_pixels = int(np.sum(mask_binary))

    tumor_percentage = (
        tumor_pixels / total_pixels
    )*100

    return {
        "total_pixels":total_pixels,
        "tumor_pixels":tumor_pixels,
        "tumor_percentage":round(tumor_percentage,2),
        "relative_image_occupancy":round(tumor_percentage,2)
    }
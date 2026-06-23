import cv2
import numpy as np

def detect_regions(mask_binary):
    num_labels,_,stats,_ = cv2.connectedComponentsWithStats(
mask_binary.astype(np.uint8),
connectivity=8
    )
    region_count = num_labels - 1

    largest_region = 0

    if(region_count > 0):
        largest_region = int(
            np.max(stats[1:,cv2.CC_STAT_AREA])
        )
    return {
        "region_count":region_count,
        "largest_region":largest_region,
        "multiple_regions":region_count>1
    }
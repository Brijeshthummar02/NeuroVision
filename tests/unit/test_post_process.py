import numpy as np


def test_post_process_segmentation_removes_small_components(app_state):
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[2:5, 2:5] = 1.0
    mask[30:50, 30:50] = 1.0

    cleaned = app_state.post_process_segmentation(mask, min_area=50)

    assert cleaned.shape == mask.shape
    assert cleaned[3, 3] == 0
    assert cleaned[35, 35] == 1

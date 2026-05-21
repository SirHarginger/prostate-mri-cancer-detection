import numpy as np

from prostate_detection.visualization.kaggle_auto_segmentation import choose_prediction_slice


def test_choose_prediction_slice_uses_largest_mask_area() -> None:
    prediction = np.zeros((4, 4, 3), dtype=np.uint8)
    prediction[:, :, 0] = 1
    prediction[:2, :2, 2] = 1

    assert choose_prediction_slice(prediction) == 0


def test_choose_prediction_slice_uses_center_for_empty_prediction() -> None:
    prediction = np.zeros((4, 4, 5), dtype=np.uint8)

    assert choose_prediction_slice(prediction) == 2

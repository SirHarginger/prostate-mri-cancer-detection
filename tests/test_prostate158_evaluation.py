import numpy as np

from prostate_detection.evaluation.prostate158_predictions import (
    binary_dice,
    compute_label_metrics,
    summarize_metrics,
)


def test_binary_dice_matches_segmentation_f1() -> None:
    ground_truth = np.array([1, 1, 0, 0], dtype=bool)
    prediction = np.array([1, 0, 1, 0], dtype=bool)

    assert binary_dice(ground_truth, prediction) == 0.5


def test_binary_dice_is_one_for_two_empty_masks() -> None:
    empty = np.zeros((2, 2), dtype=bool)

    assert binary_dice(empty, empty) == 1.0


def test_compute_label_metrics_for_one_anatomy_label() -> None:
    ground_truth = np.array([[1, 1], [0, 2]])
    prediction = np.array([[1, 0], [1, 2]])

    metrics = compute_label_metrics(
        case_id="001",
        split="valid",
        label_value=1,
        label_name="anatomy_label_1",
        ground_truth=ground_truth,
        prediction=prediction,
    )

    assert metrics.gt_voxels == 2
    assert metrics.pred_voxels == 2
    assert metrics.tp == 1
    assert metrics.fp == 1
    assert metrics.fn == 1
    assert metrics.dice == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5


def test_summarize_metrics_reports_validation_macro_mean() -> None:
    ground_truth = np.array([[1, 2], [0, 0]])
    prediction = np.array([[1, 0], [0, 0]])
    metrics = [
        compute_label_metrics(
            case_id="001",
            split="valid",
            label_value=1,
            label_name="anatomy_label_1",
            ground_truth=ground_truth,
            prediction=prediction,
        ),
        compute_label_metrics(
            case_id="001",
            split="valid",
            label_value=2,
            label_name="anatomy_label_2",
            ground_truth=ground_truth,
            prediction=prediction,
        ),
    ]

    summary = summarize_metrics(metrics)

    assert summary["splits"]["valid"]["n_cases"] == 1
    assert summary["splits"]["valid"]["anatomy_label_1"]["mean_dice"] == 1.0
    assert summary["splits"]["valid"]["anatomy_label_2"]["mean_dice"] == 0.0
    assert summary["splits"]["valid"]["macro_mean_dice"] == 0.5

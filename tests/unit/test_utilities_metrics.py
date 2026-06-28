import numpy as np
import utilities


def test_dice_coefficient_simple():
    y_true = np.array([[[[1], [1]], [[0], [0]]]], dtype=np.float32)
    y_pred = np.array([[[[1], [0]], [[1], [0]]]], dtype=np.float32)

    dice = utilities.dice_coefficient(y_true, y_pred)

    assert np.isclose(dice, 0.5)


def test_tversky_simple():
    y_true = np.array([[[[1], [1]], [[0], [0]]]], dtype=np.float32)
    y_pred = np.array([[[[1], [0]], [[1], [0]]]], dtype=np.float32)

    score = utilities.tversky(y_true, y_pred)

    assert np.isclose(score, 0.5, atol=1e-6)


def test_iou_score_simple():
    y_true = np.array([[[[1], [1]], [[0], [0]]]], dtype=np.float32)
    y_pred = np.array([[[[1], [0]], [[1], [0]]]], dtype=np.float32)

    iou = utilities.iou_score(y_true, y_pred)

    assert np.isclose(iou, 1 / 3)


def test_precision_sensitivity_specificity():
    y_true = np.array([[[[1], [1]], [[0], [0]]]], dtype=np.float32)
    y_pred = np.array([[[[1], [0]], [[1], [0]]]], dtype=np.float32)

    precision = utilities.precision_metric(y_true, y_pred)
    sensitivity = utilities.sensitivity(y_true, y_pred)
    specificity = utilities.specificity(y_true, y_pred)

    assert np.isclose(precision, 0.5)
    assert np.isclose(sensitivity, 0.5)
    assert np.isclose(specificity, 0.5)


def test_bce_dice_loss_is_finite():
    y_true = np.array([[[[1], [0]], [[0], [1]]]], dtype=np.float32)
    y_pred = np.array([[[[0.9], [0.1]], [[0.2], [0.8]]]], dtype=np.float32)

    loss = utilities.bce_dice_loss(y_true, y_pred)

    assert np.all(np.isfinite(loss))

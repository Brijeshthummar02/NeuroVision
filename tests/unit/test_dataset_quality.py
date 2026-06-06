import numpy as np
import pandas as pd
import cv2

import dataset_quality as dq


def test_generate_statistics_counts():
    df = pd.DataFrame(
        {
            "mask": [0, 1, 2],
            "patient_id": ["p1", "p2", "p1"],
            "image_path": ["a.png", "b.png", "c.png"],
            "mask_path": ["m1.png", "m2.png", "m3.png"],
        }
    )

    stats = dq.generate_statistics(df)

    assert stats["total_samples"] == 3
    assert stats["unique_patients"] == 2
    assert stats["invalid_mask_values"] == 1


def test_is_corrupted_flags_missing_and_blank(tmp_path, monkeypatch):
    monkeypatch.setattr(dq, "IMG_ROOT", str(tmp_path))

    good_img = tmp_path / "good.png"
    good_mask = tmp_path / "good_mask.png"

    rng = np.random.default_rng(0)
    cv2.imwrite(str(good_img), (rng.random((10, 10)) * 255).astype(np.uint8))
    cv2.imwrite(str(good_mask), (rng.random((10, 10)) * 255).astype(np.uint8))

    corrupted, reason = dq._is_corrupted("good.png", "good_mask.png")
    assert corrupted is False
    assert reason == ""

    blank_img = tmp_path / "blank.png"
    cv2.imwrite(str(blank_img), np.zeros((10, 10), dtype=np.uint8))
    corrupted, reason = dq._is_corrupted("blank.png", "good_mask.png")
    assert corrupted is True
    assert reason == "image_blank"

    corrupted, reason = dq._is_corrupted("missing.png", "good_mask.png")
    assert corrupted is True
    assert reason == "image_not_found"


def test_filter_corrupted_respects_missing_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(dq, "IMG_ROOT", str(tmp_path))

    good_img = tmp_path / "good.png"
    good_mask = tmp_path / "mask.png"
    cv2.imwrite(str(good_img), np.ones((10, 10), dtype=np.uint8) * 200)
    cv2.imwrite(str(good_mask), np.ones((10, 10), dtype=np.uint8) * 200)

    blank_img = tmp_path / "blank.png"
    cv2.imwrite(str(blank_img), np.zeros((10, 10), dtype=np.uint8))

    df = pd.DataFrame(
        {
            "image_path": ["missing.png", "blank.png", "good.png"],
            "mask_path": ["mask.png", "mask.png", "mask.png"],
            "mask": [0, 1, 0],
            "patient_id": ["p1", "p2", "p3"],
        }
    )

    clean_df, removed = dq.filter_corrupted(df)

    assert len(removed) == 1
    assert removed[0]["image_path"] == "blank.png"
    assert len(clean_df) == 2


def test_class_imbalance_report_severe_case():
    df = pd.DataFrame({"mask": [0] * 8 + [1] * 2})

    report = dq.class_imbalance_report(df)

    assert report["severity"] == "SEVERE"
    assert report["recommended_class_weights"]["0"] is not None
    assert report["recommended_class_weights"]["1"] is not None


def test_plot_distribution_creates_file(tmp_path):
    out_path = tmp_path / "plot.png"

    dq.plot_distribution({"0": 5, "1": 2}, str(out_path))

    assert out_path.exists()

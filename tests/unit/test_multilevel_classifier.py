"""
Unit tests for multilevel_classifier.py (issue #83).

The decision logic, label derivation, preprocessing and orchestration are tested
with plain NumPy and lightweight stub models -- no TensorFlow required, matching
the dependency-light approach in tests/conftest.py. The Keras model builders are
exercised only when a real TensorFlow install is present (importorskip), and with
weights=None so no ImageNet download is needed.
"""

import numpy as np
import pandas as pd
import pytest

import multilevel_classifier as mlc
from multilevel_classifier import (
    MultiLevelTumorClassifier,
    decide_hierarchy,
    grade_stage_consistent,
    preprocess_for_classification,
    derive_type_labels_from_mask_csv,
    class_weights_from_labels,
)


# --------------------------------------------------------------------------- #
# Stub models
# --------------------------------------------------------------------------- #
class RecordingModel:
    """Returns a fixed softmax row and counts how many times it was called."""

    def __init__(self, pred):
        self.pred = np.array([pred], dtype=np.float32)
        self.calls = 0

    def predict(self, batch, verbose=0):
        self.calls += 1
        return self.pred


def _onehot(index, n):
    v = np.zeros(n, dtype=np.float32)
    v[index] = 1.0
    return v


# --------------------------------------------------------------------------- #
# derive_type_labels_from_mask_csv
# --------------------------------------------------------------------------- #
def test_derive_type_labels_maps_binary_mask():
    df = pd.DataFrame({"image_path": ["a", "b", "c"], "mask": [1, 0, 1]})
    out = derive_type_labels_from_mask_csv(df)

    assert list(out["type"]) == ["glioma", "notumor", "glioma"]
    # flow_from_dataframe(class_mode='categorical') needs string labels; accept
    # either object or StringDtype (pandas >=3.0 returns the latter for astype(str)).
    from pandas.api.types import is_object_dtype, is_string_dtype
    assert is_string_dtype(out["type"]) or is_object_dtype(out["type"])
    assert all(isinstance(v, str) for v in out["type"])


def test_derive_type_labels_accepts_string_mask_values():
    df = pd.DataFrame({"image_path": ["a", "b"], "mask": ["1", "0"]})
    out = derive_type_labels_from_mask_csv(df)
    assert list(out["type"]) == ["glioma", "notumor"]


def test_derive_type_labels_does_not_mutate_input():
    df = pd.DataFrame({"image_path": ["a"], "mask": [1]})
    _ = derive_type_labels_from_mask_csv(df)
    assert "type" not in df.columns


def test_derive_type_labels_missing_column_raises():
    df = pd.DataFrame({"image_path": ["a"]})
    with pytest.raises(KeyError):
        derive_type_labels_from_mask_csv(df)


def test_derive_type_labels_rejects_non_binary():
    df = pd.DataFrame({"image_path": ["a", "b"], "mask": [1, 2]})
    with pytest.raises(ValueError):
        derive_type_labels_from_mask_csv(df)


def test_derive_type_labels_custom_names():
    df = pd.DataFrame({"image_path": ["a", "b"], "label": [1, 0]})
    out = derive_type_labels_from_mask_csv(
        df, mask_col="label", tumor_type="tumor", notumor_label="healthy", out_col="y"
    )
    assert list(out["y"]) == ["tumor", "healthy"]


# --------------------------------------------------------------------------- #
# grade_stage_consistent
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "grade,stage,expected",
    [
        ("lgg", "grade_i", True),
        ("lgg", "grade_ii", True),
        ("lgg", "grade_iii", False),
        ("lgg", "grade_iv", False),
        ("hgg", "grade_iii", True),
        ("hgg", "grade_iv", True),
        ("hgg", "grade_i", False),
        ("LGG", "GRADE_II", True),   # case-insensitive
        (None, "grade_i", True),     # missing level -> not an inconsistency
        ("lgg", None, True),
        (None, None, True),
        ("unknown", "grade_i", True),  # unknown grade -> cannot assert conflict
    ],
)
def test_grade_stage_consistency(grade, stage, expected):
    assert grade_stage_consistent(grade, stage) is expected


# --------------------------------------------------------------------------- #
# decide_hierarchy
# --------------------------------------------------------------------------- #
def test_decide_hierarchy_notumor_short_circuits():
    type_probs = _onehot(mlc.TYPE_LABELS.index("notumor"), 4)
    pred = decide_hierarchy(type_probs, grade_probs=_onehot(0, 2), stage_probs=_onehot(0, 4))

    assert pred.tumor_type == "notumor"
    assert pred.is_tumor is False
    assert pred.grade is None and pred.stage is None
    assert pred.levels_evaluated == ["type"]


def test_decide_hierarchy_glioma_full_cascade_consistent():
    type_probs = _onehot(mlc.TYPE_LABELS.index("glioma"), 4)
    grade_probs = _onehot(mlc.GRADE_LABELS.index("lgg"), 2)
    stage_probs = _onehot(mlc.STAGE_LABELS.index("grade_ii"), 4)

    pred = decide_hierarchy(type_probs, grade_probs, stage_probs)

    assert pred.tumor_type == "glioma" and pred.is_tumor is True
    assert pred.grade == "lgg" and pred.stage == "grade_ii"
    assert pred.consistency_warning is None
    assert pred.levels_evaluated == ["type", "grade", "stage"]
    assert pred.low_confidence is False


def test_decide_hierarchy_flags_grade_stage_conflict():
    type_probs = _onehot(mlc.TYPE_LABELS.index("glioma"), 4)
    grade_probs = _onehot(mlc.GRADE_LABELS.index("lgg"), 2)   # LGG
    stage_probs = _onehot(mlc.STAGE_LABELS.index("grade_iv"), 4)  # contradicts LGG

    pred = decide_hierarchy(type_probs, grade_probs, stage_probs)

    assert pred.consistency_warning is not None
    assert pred.low_confidence is True


def test_decide_hierarchy_non_glioma_skips_subtypes():
    type_probs = _onehot(mlc.TYPE_LABELS.index("meningioma"), 4)
    pred = decide_hierarchy(type_probs, grade_probs=_onehot(0, 2), stage_probs=_onehot(0, 4))

    assert pred.tumor_type == "meningioma" and pred.is_tumor is True
    assert pred.grade is None and pred.stage is None
    assert pred.levels_evaluated == ["type"]


def test_decide_hierarchy_low_confidence_on_type():
    # Spread probability so the winner is below the 0.5 default threshold.
    type_probs = np.array([0.4, 0.3, 0.2, 0.1])
    pred = decide_hierarchy(type_probs)
    assert pred.low_confidence is True


def test_decide_hierarchy_accepts_batched_row():
    type_probs = _onehot(0, 4).reshape(1, 4)  # (1, n) like Keras returns
    pred = decide_hierarchy(type_probs)
    assert pred.tumor_type == mlc.TYPE_LABELS[0]


def test_decide_hierarchy_rejects_wrong_class_count():
    with pytest.raises(ValueError):
        decide_hierarchy(np.array([0.5, 0.5]))  # only 2, expected 4 type classes


def test_decide_hierarchy_rejects_multi_row_batch():
    with pytest.raises(ValueError):
        decide_hierarchy(np.zeros((2, 4)))


# --------------------------------------------------------------------------- #
# MultiLevelPrediction.to_dict
# --------------------------------------------------------------------------- #
def test_prediction_to_dict_shape_and_rounding():
    type_probs = _onehot(mlc.TYPE_LABELS.index("glioma"), 4)
    grade_probs = np.array([0.123456, 0.876544])  # hgg, lgg
    stage_probs = _onehot(mlc.STAGE_LABELS.index("grade_ii"), 4)

    d = decide_hierarchy(type_probs, grade_probs, stage_probs).to_dict()

    assert set(d) == {
        "tumor_type", "type_confidence", "type_probabilities", "is_tumor",
        "grade", "grade_confidence", "grade_probabilities",
        "stage", "stage_confidence", "stage_probabilities",
        "low_confidence", "consistency_warning", "levels_evaluated",
    }
    assert d["grade"] == "lgg"
    assert d["grade_probabilities"]["lgg"] == pytest.approx(0.8765, abs=1e-4)


def test_prediction_to_dict_notumor_has_null_subtypes():
    d = decide_hierarchy(_onehot(mlc.TYPE_LABELS.index("notumor"), 4)).to_dict()
    assert d["grade"] is None
    assert d["grade_probabilities"] is None
    assert d["stage_probabilities"] is None


# --------------------------------------------------------------------------- #
# MultiLevelTumorClassifier orchestration
# --------------------------------------------------------------------------- #
def _glioma_type_model():
    return RecordingModel(_onehot(mlc.TYPE_LABELS.index("glioma"), 4))


def test_orchestrator_glioma_runs_all_levels():
    type_m = _glioma_type_model()
    grade_m = RecordingModel(_onehot(mlc.GRADE_LABELS.index("hgg"), 2))
    stage_m = RecordingModel(_onehot(mlc.STAGE_LABELS.index("grade_iv"), 4))

    engine = MultiLevelTumorClassifier(type_m, grade_m, stage_m)
    batch = np.zeros((1, 256, 256, 3), dtype=np.float32)
    pred = engine.predict(batch)

    assert pred.tumor_type == "glioma"
    assert pred.grade == "hgg" and pred.stage == "grade_iv"
    assert grade_m.calls == 1 and stage_m.calls == 1


def test_orchestrator_notumor_skips_subtype_models():
    type_m = RecordingModel(_onehot(mlc.TYPE_LABELS.index("notumor"), 4))
    grade_m = RecordingModel(_onehot(0, 2))
    stage_m = RecordingModel(_onehot(0, 4))

    engine = MultiLevelTumorClassifier(type_m, grade_m, stage_m)
    pred = engine.predict(np.zeros((1, 256, 256, 3), dtype=np.float32))

    assert pred.is_tumor is False
    # The expensive sub-models must NOT run on a healthy scan.
    assert grade_m.calls == 0 and stage_m.calls == 0


def test_orchestrator_meningioma_skips_subtype_models():
    type_m = RecordingModel(_onehot(mlc.TYPE_LABELS.index("meningioma"), 4))
    grade_m = RecordingModel(_onehot(0, 2))
    stage_m = RecordingModel(_onehot(0, 4))

    engine = MultiLevelTumorClassifier(type_m, grade_m, stage_m)
    pred = engine.predict(np.zeros((1, 256, 256, 3), dtype=np.float32))

    assert pred.tumor_type == "meningioma"
    assert grade_m.calls == 0 and stage_m.calls == 0


def test_orchestrator_type_only_returns_null_subtypes():
    engine = MultiLevelTumorClassifier(_glioma_type_model())  # no sub-models
    pred = engine.predict(np.zeros((1, 256, 256, 3), dtype=np.float32))
    assert pred.tumor_type == "glioma"
    assert pred.grade is None and pred.stage is None


def test_orchestrator_requires_type_model():
    with pytest.raises(ValueError):
        MultiLevelTumorClassifier(None)


# --------------------------------------------------------------------------- #
# preprocess_for_classification
# --------------------------------------------------------------------------- #
def test_preprocess_grayscale_to_batch():
    gray = np.full((40, 40), 128, dtype=np.uint8)
    out = preprocess_for_classification(gray)
    assert out.shape == (1, 256, 256, 3)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() <= out.max() <= 1.0
    assert np.allclose(out, 128 / 255.0, atol=1e-3)


def test_preprocess_bgr_three_channel():
    bgr = np.full((10, 10, 3), [10, 20, 30], dtype=np.uint8)
    out = preprocess_for_classification(bgr)
    assert out.shape == (1, 256, 256, 3)
    # BGR -> RGB channel swap, scaled by 1/255
    expected = np.array([30, 20, 10], dtype=np.float32) / 255.0
    assert np.allclose(out[0, 0, 0], expected, atol=1e-3)


def test_preprocess_custom_image_size():
    out = preprocess_for_classification(np.zeros((10, 10, 3), dtype=np.uint8), img_size=128)
    assert out.shape == (1, 128, 128, 3)


# --------------------------------------------------------------------------- #
# class_weights_from_labels
# --------------------------------------------------------------------------- #
def test_class_weights_balanced_keys_and_minority_upweighted():
    labels = ["notumor"] * 8 + ["glioma"] * 2   # 4:1 imbalance
    weights = class_weights_from_labels(labels)

    # keyed by sorted-label index: glioma=0, notumor=1
    assert set(weights) == {0, 1}
    assert weights[0] > weights[1]  # minority (glioma) gets the higher weight


# --------------------------------------------------------------------------- #
# Keras builders (only when a real TensorFlow is installed)
# --------------------------------------------------------------------------- #
def _has_real_tf():
    tf = pytest.importorskip("tensorflow")
    # The conftest stub is a bare module without keras.applications; skip on it.
    if not hasattr(tf, "keras") or not hasattr(tf.keras, "applications"):
        pytest.skip("Stubbed TensorFlow without keras.applications; skipping build test.")
    return tf


def test_build_type_classifier_output_shape():
    _has_real_tf()
    model = mlc.build_tumor_type_classifier(img_size=64, weights=None)
    assert model.output_shape == (None, len(mlc.TYPE_LABELS))


def test_build_grade_and_stage_shapes():
    _has_real_tf()
    grade = mlc.build_glioma_grade_classifier(img_size=64, weights=None)
    stage = mlc.build_tumor_stage_classifier(img_size=64, weights=None, use_attention=True)
    assert grade.output_shape == (None, len(mlc.GRADE_LABELS))
    assert stage.output_shape == (None, len(mlc.STAGE_LABELS))


def test_build_custom_cnn_backbone():
    _has_real_tf()
    model = mlc.build_tumor_type_classifier(img_size=64, backbone="custom_cnn", weights=None)
    assert model.output_shape == (None, len(mlc.TYPE_LABELS))


def test_compile_classifier_sets_optimizer():
    _has_real_tf()
    model = mlc.build_glioma_grade_classifier(img_size=64, weights=None)
    mlc.compile_classifier(model, learning_rate=1e-4)
    assert model.optimizer is not None

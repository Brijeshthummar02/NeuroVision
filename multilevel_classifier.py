"""
multilevel_classifier.py
========================
Multi-Level (hierarchical) tumour classification for NeuroVision v2.0.

Implements issue #83 by extending the existing *binary* tumour detector into a
three-level cascade:

    Level 1  TumorTypeClassifier    -> glioma | meningioma | notumor | pituitary
    Level 2  GliomaGradeClassifier  -> LGG | HGG               (gliomas only)
    Level 3  TumorStageClassifier   -> grade_i .. grade_iv     (gliomas only)

The cascade short-circuits exactly like the v2 detection pipeline: if Level 1
returns ``notumor`` the heavier sub-models are never invoked, and grade/stage are
only evaluated for tumour types that actually carry a WHO glioma grade.

Design decisions
----------------
1. **Framework parity over literal v3 replication.** The public v2 stack is
   TensorFlow/Keras (ResNet50 transfer learning, 256x256 inputs, ``rescale=1/255``).
   The private-v3 diagram in the issue shows PyTorch ``.pth`` / ResNet18 models.
   Because the issue requires the feature to "integrate perfectly with the
   existing model and notebook cells", every builder here mirrors the v2 Keras
   conventions (see ``index.ipynb`` cells 20-27). Backbone, image size and head
   geometry are all configurable if you want to match the v3 parameter budget.

2. **Honest data handling.** The bundled dataset (TCGA-LGG, ``data_mask.csv``)
   only carries a binary ``mask`` column. Every tumour slice in TCGA-LGG is a
   *glioma*, so Level 1 is immediately trainable on the existing data for the
   ``{glioma, notumor}`` subset via :func:`derive_type_labels_from_mask_csv`.
   The remaining type classes (meningioma, pituitary) and the grade/stage levels
   require an extended, labelled corpus. See ``MULTILEVEL_CLASSIFICATION.md``.

3. **Testable core, lazy TensorFlow.** All decision logic (cascade, confidence
   gating, grade/stage consistency, label derivation, preprocessing) is pure
   NumPy/Python and unit-tested without TensorFlow. Keras is imported lazily
   inside the model-building / training helpers, so this module imports cleanly
   even when TensorFlow is not installed (matching the dependency-light test
   setup in ``tests/conftest.py``).

The module is intentionally framework-thin in its hot path: the orchestrator
operates on the probability vectors returned by ``model.predict(...)``, so it can
be driven by real Keras models, ensembles, or lightweight stubs in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Label vocabularies
# --------------------------------------------------------------------------- #
# Ordering matters: the index of each label is the column index of the
# corresponding softmax output. Keep these in sync with the generators created
# by build_type_data_generators (Keras sorts class folders/labels alphabetically,
# so these lists are kept alphabetical to match flow_from_dataframe behaviour).

TYPE_LABELS: List[str] = ["glioma", "meningioma", "notumor", "pituitary"]
GRADE_LABELS: List[str] = ["hgg", "lgg"]                       # alphabetical
STAGE_LABELS: List[str] = ["grade_i", "grade_ii", "grade_iii", "grade_iv"]

NOTUMOR_LABEL: str = "notumor"

# Tumour types for which grade/stage are clinically meaningful in this pipeline.
# The v3 diagram scopes grade (LGG/HGG) and stage (I-IV) to gliomas only.
GRADE_STAGE_APPLIES_TO: Tuple[str, ...] = ("glioma",)

# Maps a coarse glioma grade to the fine WHO stage band it is consistent with.
# WHO CNS grading: LGG == grades I-II, HGG == grades III-IV.
_GRADE_TO_STAGES: Dict[str, frozenset] = {
    "lgg": frozenset({"grade_i", "grade_ii"}),
    "hgg": frozenset({"grade_iii", "grade_iv"}),
}

DEFAULT_IMG_SIZE: int = 256          # matches index.ipynb IMG_SIZE
DEFAULT_MIN_CONFIDENCE: float = 0.50  # below this a level is flagged low-confidence


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class MultiLevelPrediction:
    """Structured output of the multi-level cascade for a single MRI slice."""

    tumor_type: str
    type_confidence: float
    type_probs: Dict[str, float]

    is_tumor: bool

    grade: Optional[str] = None
    grade_confidence: Optional[float] = None
    grade_probs: Optional[Dict[str, float]] = None

    stage: Optional[str] = None
    stage_confidence: Optional[float] = None
    stage_probs: Optional[Dict[str, float]] = None

    low_confidence: bool = False
    consistency_warning: Optional[str] = None
    levels_evaluated: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serialisable view, shaped to slot into app.predict_tumor()."""
        return {
            "tumor_type": self.tumor_type,
            "type_confidence": round(float(self.type_confidence), 4),
            "type_probabilities": {k: round(float(v), 4) for k, v in self.type_probs.items()},
            "is_tumor": bool(self.is_tumor),
            "grade": self.grade,
            "grade_confidence": (
                round(float(self.grade_confidence), 4)
                if self.grade_confidence is not None else None
            ),
            "grade_probabilities": (
                {k: round(float(v), 4) for k, v in self.grade_probs.items()}
                if self.grade_probs is not None else None
            ),
            "stage": self.stage,
            "stage_confidence": (
                round(float(self.stage_confidence), 4)
                if self.stage_confidence is not None else None
            ),
            "stage_probabilities": (
                {k: round(float(v), 4) for k, v in self.stage_probs.items()}
                if self.stage_probs is not None else None
            ),
            "low_confidence": bool(self.low_confidence),
            "consistency_warning": self.consistency_warning,
            "levels_evaluated": list(self.levels_evaluated),
        }


# --------------------------------------------------------------------------- #
# Pure helpers (no TensorFlow)
# --------------------------------------------------------------------------- #
def _as_prob_vector(probs, n_expected: Optional[int] = None) -> np.ndarray:
    """Coerce a model output into a 1-D probability vector.

    Accepts a 1-D array ``(n,)`` or a batched array ``(1, n)`` (the shape Keras
    returns for a single example). Raises on ambiguous batch sizes so callers
    fail loudly rather than silently scoring the wrong row.
    """
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[0] != 1:
            raise ValueError(
                f"Expected a single-example batch (1, n); got batch of "
                f"{arr.shape[0]}. Use predict() per image."
            )
        arr = arr[0]
    elif arr.ndim != 1:
        raise ValueError(f"Probability output must be 1-D or (1, n); got shape {arr.shape}.")
    if n_expected is not None and arr.shape[0] != n_expected:
        raise ValueError(
            f"Expected {n_expected} class scores but model returned {arr.shape[0]}."
        )
    return arr


def _probs_to_dict(probs: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
    return {label: float(probs[i]) for i, label in enumerate(labels)}


def grade_stage_consistent(grade: Optional[str], stage: Optional[str]) -> bool:
    """True when a coarse glioma grade agrees with the fine WHO stage band.

    LGG is consistent with grades I-II, HGG with grades III-IV. If either input
    is ``None`` (level not evaluated) the pair is treated as consistent.
    """
    if grade is None or stage is None:
        return True
    allowed = _GRADE_TO_STAGES.get(grade.lower())
    if allowed is None:
        # Unknown grade label -> cannot assert an inconsistency.
        return True
    return stage.lower() in allowed


def decide_hierarchy(
    type_probs,
    grade_probs=None,
    stage_probs=None,
    *,
    type_labels: Sequence[str] = TYPE_LABELS,
    grade_labels: Sequence[str] = GRADE_LABELS,
    stage_labels: Sequence[str] = STAGE_LABELS,
    notumor_label: str = NOTUMOR_LABEL,
    grade_stage_applies_to: Sequence[str] = GRADE_STAGE_APPLIES_TO,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> MultiLevelPrediction:
    """Resolve the full hierarchy from raw per-level probability vectors.

    This is the heart of the feature and is deliberately free of any framework
    dependency: it consumes plain probability arrays (whatever produced them) and
    returns a :class:`MultiLevelPrediction`. ``grade_probs`` / ``stage_probs`` may
    be ``None`` when the caller chose not to evaluate those levels (e.g. because
    Level 1 did not return a glioma).

    Confidence gating: any evaluated level whose top-1 probability falls below
    ``min_confidence`` flags the whole prediction as ``low_confidence`` so the UI
    can surface an "uncertain" badge instead of an over-confident label.
    """
    type_vec = _as_prob_vector(type_probs, len(type_labels))
    ti = int(np.argmax(type_vec))
    tumor_type = type_labels[ti]
    type_conf = float(type_vec[ti])
    is_tumor = tumor_type != notumor_label

    levels_evaluated = ["type"]
    low_conf = type_conf < min_confidence

    grade = grade_conf = grade_dict = None
    stage = stage_conf = stage_dict = None

    evaluate_subtypes = tumor_type in tuple(grade_stage_applies_to)

    if evaluate_subtypes and grade_probs is not None:
        grade_vec = _as_prob_vector(grade_probs, len(grade_labels))
        gi = int(np.argmax(grade_vec))
        grade = grade_labels[gi]
        grade_conf = float(grade_vec[gi])
        grade_dict = _probs_to_dict(grade_vec, grade_labels)
        levels_evaluated.append("grade")
        low_conf = low_conf or grade_conf < min_confidence

    if evaluate_subtypes and stage_probs is not None:
        stage_vec = _as_prob_vector(stage_probs, len(stage_labels))
        si = int(np.argmax(stage_vec))
        stage = stage_labels[si]
        stage_conf = float(stage_vec[si])
        stage_dict = _probs_to_dict(stage_vec, stage_labels)
        levels_evaluated.append("stage")
        low_conf = low_conf or stage_conf < min_confidence

    warning = None
    if not grade_stage_consistent(grade, stage):
        warning = (
            f"Grade '{grade}' and stage '{stage}' disagree "
            f"(LGG=I/II, HGG=III/IV); treat sub-classification as uncertain."
        )
        low_conf = True

    return MultiLevelPrediction(
        tumor_type=tumor_type,
        type_confidence=type_conf,
        type_probs=_probs_to_dict(type_vec, type_labels),
        is_tumor=is_tumor,
        grade=grade,
        grade_confidence=grade_conf,
        grade_probs=grade_dict,
        stage=stage,
        stage_confidence=stage_conf,
        stage_probs=stage_dict,
        low_confidence=low_conf,
        consistency_warning=warning,
        levels_evaluated=levels_evaluated,
    )


def preprocess_for_classification(img, img_size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    """Preprocess an image to a classifier-ready batch of shape (1, S, S, 3).

    Mirrors ``app.preprocess_image_classification`` and the notebook's
    ``ImageDataGenerator(rescale=1./255.)`` exactly: BGR/grayscale/RGBA inputs are
    converted to RGB, resized to ``img_size`` and scaled to ``[0, 1]``. Keeping
    this identical to the training-time transform is what lets the same weights
    serve all three levels without a train/serve skew.
    """
    import cv2  # local import; cv2 is a hard dep of the repo but keep TF-free

    arr = np.asarray(img)
    arr = cv2.resize(arr, (img_size, img_size))

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    arr = arr.astype(np.float32) / 255.0
    return np.reshape(arr, (1, img_size, img_size, 3))


def derive_type_labels_from_mask_csv(
    df,
    *,
    mask_col: str = "mask",
    tumor_type: str = "glioma",
    notumor_label: str = NOTUMOR_LABEL,
    out_col: str = "type",
):
    """Bootstrap Level-1 ``type`` labels from the binary TCGA-LGG ``mask`` column.

    Every tumour-positive slice in TCGA-LGG is a glioma, so ``mask == 1`` maps to
    ``glioma`` and ``mask == 0`` maps to ``notumor``. This yields a real,
    immediately trainable two-class subset of the Level-1 problem on the dataset
    that already ships with the repo. The returned frame adds ``out_col`` as a
    *string* column (required by ``flow_from_dataframe`` with
    ``class_mode='categorical'``); the original frame is not mutated.

    Raises
    ------
    KeyError
        If ``mask_col`` is absent.
    ValueError
        If ``mask_col`` contains values outside {0, 1} (as int or str).
    """
    if mask_col not in df.columns:
        raise KeyError(f"Column '{mask_col}' not found in dataframe.")

    out = df.copy()
    mask_int = out[mask_col].apply(lambda v: int(str(v).strip()))
    bad = sorted(set(mask_int.unique()) - {0, 1})
    if bad:
        raise ValueError(f"'{mask_col}' must be binary 0/1; found unexpected values {bad}.")

    out[out_col] = mask_int.map({1: tumor_type, 0: notumor_label}).astype(str)
    return out


def class_weights_from_labels(labels: Sequence[str]) -> Dict[int, float]:
    """Balanced class weights keyed by the *sorted* label index.

    Useful for the imbalanced derived dataset (TCGA-LGG is ~2:1 notumor:glioma).
    Index ordering follows ``sorted(set(labels))`` to match the class indices
    Keras assigns in ``flow_from_dataframe``.
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.array(sorted(set(labels)))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.asarray(labels))
    return {int(i): float(w) for i, w in enumerate(weights)}


# --------------------------------------------------------------------------- #
# Keras model builders (TensorFlow imported lazily)
# --------------------------------------------------------------------------- #
def _require_tf():
    """Import TensorFlow on demand with a clear, actionable error message."""
    try:
        import tensorflow as tf  # noqa: F401
        return tf
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "TensorFlow is required to build/train multi-level models. "
            "Install it with `pip install tensorflow` (or tensorflow-cpu). "
            "Pure inference orchestration and decision logic do not need TF."
        ) from exc


def _se_block(x, ratio: int = 16, name: str = "se"):
    """Squeeze-and-Excitation channel-attention block.

    Recalibrates channel responses, matching the SE attention the v2 ResUNet
    already uses (utilities.py / notebook cell 48). Used as the lightweight
    "attention" stage of the stage classifier in lieu of a bespoke ResNet18+Attn.
    """
    tf = _require_tf()
    from tensorflow.keras import layers

    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D(name=f"{name}_squeeze")(x)
    se = layers.Dense(max(channels // ratio, 1), activation="relu", name=f"{name}_reduce")(se)
    se = layers.Dense(channels, activation="sigmoid", name=f"{name}_expand")(se)
    se = layers.Reshape((1, 1, channels), name=f"{name}_reshape")(se)
    return layers.multiply([x, se], name=f"{name}_scale")


def _classification_head(
    x,
    num_classes: int,
    *,
    dense_units: Sequence[int] = (512, 256, 128),
    dropouts: Sequence[float] = (0.4, 0.3, 0.2),
    l2_reg: float = 0.001,
    name: str = "head",
):
    """Regularised GAP -> Dense stack, identical in spirit to notebook cell 25.

    The final softmax is forced to float32 so the head stays numerically stable
    under the global ``mixed_float16`` policy enabled in notebook cell 2.
    """
    _require_tf()
    from tensorflow.keras import layers
    from tensorflow.keras.regularizers import l2

    if len(dense_units) != len(dropouts):
        raise ValueError("dense_units and dropouts must have equal length.")

    x = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    for i, (units, drop) in enumerate(zip(dense_units, dropouts)):
        x = layers.Dense(units, kernel_regularizer=l2(l2_reg), name=f"{name}_dense{i}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.Activation("relu", name=f"{name}_relu{i}")(x)
        x = layers.Dropout(drop, name=f"{name}_drop{i}")(x)
    return layers.Dense(num_classes, activation="softmax", dtype="float32", name=f"{name}_out")(x)


def _build_custom_cnn(img_size: int, num_classes: int, name: str = "custom_cnn"):
    """Compact from-scratch CNN (~2-3M params) for the Level-1 type classifier.

    Provided so the type classifier can match the private-v3 "Custom CNN" budget
    and run in environments without ImageNet weight access. ResNet50 transfer
    learning (the default backbone) is still recommended where weights are
    available.
    """
    _require_tf()
    from tensorflow.keras import layers
    from tensorflow.keras.models import Model

    inp = layers.Input(shape=(img_size, img_size, 3), name=f"{name}_input")
    x = inp
    for i, filters in enumerate((32, 64, 128, 256)):
        x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv{i}a")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}a")(x)
        x = layers.Activation("relu", name=f"{name}_relu{i}a")(x)
        x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv{i}b")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}b")(x)
        x = layers.Activation("relu", name=f"{name}_relu{i}b")(x)
        x = layers.MaxPooling2D(2, name=f"{name}_pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    x = layers.Dense(256, activation="relu", name=f"{name}_fc")(x)
    x = layers.Dropout(0.3, name=f"{name}_drop")(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32", name=f"{name}_out")(x)
    return Model(inp, out, name=name)


def _build_backbone_classifier(
    num_classes: int,
    *,
    img_size: int,
    backbone: str,
    weights: Optional[str],
    freeze_base: bool,
    use_attention: bool,
    name: str,
):
    """Shared builder for the three transfer-learning classifiers."""
    tf = _require_tf()
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model

    backbone = backbone.lower()
    if backbone == "custom_cnn":
        if use_attention:
            raise ValueError("use_attention is only supported with a CNN backbone (resnet50).")
        return _build_custom_cnn(img_size, num_classes, name=name)

    if backbone == "resnet50":
        base_cls = tf.keras.applications.ResNet50
    elif backbone == "resnet50v2":
        base_cls = tf.keras.applications.ResNet50V2
    else:
        raise ValueError(f"Unsupported backbone '{backbone}'. Use 'resnet50', 'resnet50v2' or 'custom_cnn'.")

    base = base_cls(weights=weights, include_top=False,
                    input_tensor=Input(shape=(img_size, img_size, 3)))
    if freeze_base:
        for layer in base.layers:
            layer.trainable = False

    x = base.output
    if use_attention:
        x = _se_block(x, name=f"{name}_se")
    out = _classification_head(x, num_classes, name=f"{name}_head")
    return Model(inputs=base.input, outputs=out, name=name)


def build_tumor_type_classifier(
    num_classes: int = len(TYPE_LABELS),
    *,
    img_size: int = DEFAULT_IMG_SIZE,
    backbone: str = "resnet50",
    weights: Optional[str] = "imagenet",
    freeze_base: bool = True,
):
    """Level 1: tumour-type classifier (glioma/meningioma/notumor/pituitary)."""
    return _build_backbone_classifier(
        num_classes, img_size=img_size, backbone=backbone, weights=weights,
        freeze_base=freeze_base, use_attention=False, name="tumor_type_classifier",
    )


def build_glioma_grade_classifier(
    *,
    img_size: int = DEFAULT_IMG_SIZE,
    backbone: str = "resnet50",
    weights: Optional[str] = "imagenet",
    freeze_base: bool = True,
):
    """Level 2: glioma grade classifier (LGG/HGG)."""
    return _build_backbone_classifier(
        len(GRADE_LABELS), img_size=img_size, backbone=backbone, weights=weights,
        freeze_base=freeze_base, use_attention=False, name="glioma_grade_classifier",
    )


def build_tumor_stage_classifier(
    *,
    img_size: int = DEFAULT_IMG_SIZE,
    backbone: str = "resnet50",
    weights: Optional[str] = "imagenet",
    freeze_base: bool = True,
    use_attention: bool = True,
):
    """Level 3: WHO stage classifier (grade I-IV) with optional SE attention."""
    return _build_backbone_classifier(
        len(STAGE_LABELS), img_size=img_size, backbone=backbone, weights=weights,
        freeze_base=freeze_base, use_attention=use_attention, name="tumor_stage_classifier",
    )


def compile_classifier(model, *, learning_rate: float = 1e-4, label_smoothing: float = 0.1):
    """Compile a classifier with the v2 notebook's settings (cell 27)."""
    tf = _require_tf()
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_type_data_generators(
    df,
    *,
    label_col: str = "type",
    image_col: str = "image_path",
    directory: str = "./",
    img_size: int = DEFAULT_IMG_SIZE,
    batch_size: int = 16,
    val_split: float = 0.15,
    augment: bool = True,
):
    """Train/val ``flow_from_dataframe`` generators, mirroring notebook cells 20-21.

    The training generator uses the same augmentation knobs as the v2 classifier
    (rotation/shift/shear/zoom/flips, constant fill); the validation generator
    only rescales. ``label_col`` must be a string column (use
    :func:`derive_type_labels_from_mask_csv` for the bundled data).
    """
    _require_tf()
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found; did you derive it first?")

    aug_kwargs = dict(
        rotation_range=20, width_shift_range=0.15, height_shift_range=0.15,
        shear_range=0.15, zoom_range=0.15, horizontal_flip=True,
        vertical_flip=True, fill_mode="constant", cval=0,
    ) if augment else {}

    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=val_split, **aug_kwargs)
    valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=val_split)

    common = dict(dataframe=df, directory=directory, x_col=image_col, y_col=label_col,
                  batch_size=batch_size, class_mode="categorical", target_size=(img_size, img_size))

    train_gen = train_datagen.flow_from_dataframe(subset="training", shuffle=True, **common)
    valid_gen = valid_datagen.flow_from_dataframe(subset="validation", shuffle=False, **common)
    return train_gen, valid_gen


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class MultiLevelTumorClassifier:
    """Cascading inference engine over the three classification levels.

    Only ``type_model`` is required. ``grade_model`` / ``stage_model`` are
    optional; when absent (or when Level 1 does not return a glioma) those levels
    are skipped and the corresponding fields are ``None``. Each model only needs a
    ``.predict(batch, verbose=0)`` method returning a softmax array, so real Keras
    models, the app's ensemble callables, or test stubs all work.

    Parameters
    ----------
    type_model, grade_model, stage_model
        Objects exposing ``predict(np.ndarray) -> np.ndarray``.
    *_labels
        Ordered label vocabularies matching each model's output columns.
    min_confidence
        Top-1 threshold below which a level is flagged ``low_confidence``.
    """

    def __init__(
        self,
        type_model,
        grade_model=None,
        stage_model=None,
        *,
        type_labels: Sequence[str] = TYPE_LABELS,
        grade_labels: Sequence[str] = GRADE_LABELS,
        stage_labels: Sequence[str] = STAGE_LABELS,
        notumor_label: str = NOTUMOR_LABEL,
        grade_stage_applies_to: Sequence[str] = GRADE_STAGE_APPLIES_TO,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        img_size: int = DEFAULT_IMG_SIZE,
    ):
        if type_model is None:
            raise ValueError("type_model is required for the multi-level cascade.")
        self.type_model = type_model
        self.grade_model = grade_model
        self.stage_model = stage_model
        self.type_labels = list(type_labels)
        self.grade_labels = list(grade_labels)
        self.stage_labels = list(stage_labels)
        self.notumor_label = notumor_label
        self.grade_stage_applies_to = tuple(grade_stage_applies_to)
        self.min_confidence = float(min_confidence)
        self.img_size = int(img_size)

    @staticmethod
    def _predict(model, batch) -> np.ndarray:
        return np.asarray(model.predict(batch, verbose=0))

    def predict(self, image, *, preprocess: bool = False) -> MultiLevelPrediction:
        """Run the cascade on a single image.

        Parameters
        ----------
        image
            Either a preprocessed ``(1, S, S, 3)`` batch (``preprocess=False``,
            the default) or a raw BGR/grayscale image (``preprocess=True``), in
            which case :func:`preprocess_for_classification` is applied first.
        """
        batch = preprocess_for_classification(image, self.img_size) if preprocess else image

        type_probs = self._predict(self.type_model, batch)

        # Peek the type to decide whether the glioma-only sub-models should run,
        # mirroring the v2 "skip the heavy model on healthy scans" optimisation.
        ti = int(np.argmax(_as_prob_vector(type_probs, len(self.type_labels))))
        predicted_type = self.type_labels[ti]
        run_subtypes = predicted_type in self.grade_stage_applies_to

        grade_probs = (
            self._predict(self.grade_model, batch)
            if run_subtypes and self.grade_model is not None else None
        )
        stage_probs = (
            self._predict(self.stage_model, batch)
            if run_subtypes and self.stage_model is not None else None
        )

        return decide_hierarchy(
            type_probs, grade_probs, stage_probs,
            type_labels=self.type_labels, grade_labels=self.grade_labels,
            stage_labels=self.stage_labels, notumor_label=self.notumor_label,
            grade_stage_applies_to=self.grade_stage_applies_to,
            min_confidence=self.min_confidence,
        )


__all__ = [
    "TYPE_LABELS", "GRADE_LABELS", "STAGE_LABELS", "NOTUMOR_LABEL",
    "GRADE_STAGE_APPLIES_TO", "DEFAULT_IMG_SIZE", "DEFAULT_MIN_CONFIDENCE",
    "MultiLevelPrediction", "MultiLevelTumorClassifier",
    "decide_hierarchy", "grade_stage_consistent",
    "preprocess_for_classification", "derive_type_labels_from_mask_csv",
    "class_weights_from_labels", "build_tumor_type_classifier",
    "build_glioma_grade_classifier", "build_tumor_stage_classifier",
    "compile_classifier", "build_type_data_generators",
]

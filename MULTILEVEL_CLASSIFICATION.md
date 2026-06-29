# Multi-Level Tumor Classification

Hierarchical tumour classification for NeuroVision v2.0 — the feature behind
[issue #83](https://github.com/Brijeshthummar02/NeuroVision/issues/83).

It extends the existing **binary** detector (*tumour / no tumour*) into a
**three-level cascade**, implemented in [`multilevel_classifier.py`](./multilevel_classifier.py):

| Level | Model builder | Output classes | Runs when |
|------:|---------------|----------------|-----------|
| 1 | `build_tumor_type_classifier` | `glioma` · `meningioma` · `notumor` · `pituitary` | always |
| 2 | `build_glioma_grade_classifier` | `lgg` · `hgg` | type == glioma |
| 3 | `build_tumor_stage_classifier` | `grade_i` · `grade_ii` · `grade_iii` · `grade_iv` | type == glioma |

The orchestrator (`MultiLevelTumorClassifier`) cascades these and **short-circuits**
exactly like the detector→segmenter pipeline: if Level 1 returns `notumor`, the
heavier sub-models never run; grade and stage are evaluated only for tumour types
that carry a WHO glioma grade.

---

## How the cascade works

```
            ┌─────────────────────────────┐
  MRI ────▶ │ Level 1  TumorTypeClassifier │
            └──────────────┬──────────────┘
                           │  argmax
          ┌────────────────┼─────────────────────────┐
          ▼                ▼                          ▼
       notumor        meningioma / pituitary       glioma
          │                │                          │
        STOP             STOP (type only)             ▼
                                        ┌──────────────────────────────┐
                                        │ Level 2  GliomaGradeClassifier│ → LGG / HGG
                                        ├──────────────────────────────┤
                                        │ Level 3  TumorStageClassifier │ → grade I–IV
                                        └──────────────┬───────────────┘
                                                       ▼
                                       grade↔stage consistency check
                                       (LGG⇔I/II, HGG⇔III/IV) + confidence gating
```

Every evaluated level whose top-1 probability is below `min_confidence`
(default `0.50`) flags the whole prediction `low_confidence`, so the UI can show
an *uncertain* badge instead of an over-confident label. If the coarse grade and
the fine WHO stage disagree, a `consistency_warning` is attached and the
prediction is flagged low-confidence.

---

## Two design decisions worth knowing

**1. Framework parity over literal v3 replication.** The private-v3 diagram in the
issue shows PyTorch `.pth` / ResNet18 models at 224×224. The public v2 stack is
**TensorFlow/Keras** (ResNet50 transfer learning, 256×256, `rescale=1/255`, the
cell-25 head and cell-27 compile settings). Because the issue requires the feature
to *"integrate perfectly with the existing model and notebook cells"*, every
builder here mirrors the v2 Keras conventions. Consequences:

- Backbones are **ResNet50** (Keras ships no ResNet18), so parameter counts are
  larger than the diagram's (~25M vs ~11M). Use `backbone='custom_cnn'` on the
  type classifier for a lightweight (~2–3M) from-scratch alternative that also
  works without ImageNet weight access.
- The stage classifier's *attention* is a **Squeeze-and-Excitation** block — the
  same attention primitive the v2 ResUNet already uses — rather than a bespoke
  ResNet18+Attention.
- `img_size`, `backbone`, `weights` and head geometry are all parameters, so the
  v3 budget can be matched exactly if desired.

**2. Honest data handling.** The bundled dataset (**TCGA-LGG**, `data_mask.csv`)
only carries a binary `mask` column. Every tumour-positive slice in TCGA-LGG is a
*glioma*, so Level 1 is immediately trainable for the `{glioma, notumor}` subset
on the data already in the repo, via `derive_type_labels_from_mask_csv`. The other
two type classes and the grade/stage levels need an extended labelled corpus
(see [Data requirements](#data-requirements)). Nothing here fabricates labels the
dataset does not contain.

---

## Quick start

```python
from multilevel_classifier import (
    derive_type_labels_from_mask_csv, class_weights_from_labels,
    build_tumor_type_classifier, build_glioma_grade_classifier,
    build_tumor_stage_classifier, compile_classifier,
    build_type_data_generators, MultiLevelTumorClassifier,
)
import pandas as pd, cv2

# 1. Bootstrap Level-1 labels from the existing binary mask column.
brain_df = pd.read_csv("data_mask.csv")
type_df  = derive_type_labels_from_mask_csv(brain_df)        # adds 'type' column

# 2. Train the Level-1 type classifier on the {glioma, notumor} subset.
train_gen, valid_gen = build_type_data_generators(type_df, label_col="type")
type_model = compile_classifier(build_tumor_type_classifier())
type_model.fit(
    train_gen, validation_data=valid_gen, epochs=30,
    class_weight=class_weights_from_labels(type_df["type"].tolist()),
)

# 3. Cascade inference (grade/stage models optional).
grade_model = compile_classifier(build_glioma_grade_classifier())
stage_model = compile_classifier(build_tumor_stage_classifier(use_attention=True))
engine = MultiLevelTumorClassifier(type_model, grade_model, stage_model)

pred = engine.predict(cv2.imread("scan.png"), preprocess=True)
print(pred.to_dict())
```

Example `to_dict()` output:

```json
{
  "tumor_type": "glioma",
  "type_confidence": 0.94,
  "type_probabilities": {"glioma": 0.94, "meningioma": 0.03, "notumor": 0.01, "pituitary": 0.02},
  "is_tumor": true,
  "grade": "hgg",
  "grade_confidence": 0.81,
  "grade_probabilities": {"hgg": 0.81, "lgg": 0.19},
  "stage": "grade_iv",
  "stage_confidence": 0.77,
  "stage_probabilities": {"grade_i": 0.05, "grade_ii": 0.06, "grade_iii": 0.12, "grade_iv": 0.77},
  "low_confidence": false,
  "consistency_warning": null,
  "levels_evaluated": ["type", "grade", "stage"]
}
```

The same flow is wired into the notebook in the new
**“Multi-Level Tumor Classification (Issue #83)”** section (cells inserted after
the ROC-curve cell). The training cell is gated behind `RUN_MULTILEVEL_TRAINING`
so the notebook still runs end-to-end quickly.

---

## Data requirements

Each level trains from a dataframe with an `image_path` column plus one **string**
label column. The orchestrator and decision logic are dataset-agnostic — only the
per-level training data changes.

| Level | Label column | Values | Source for the full problem |
|------:|--------------|--------|-----------------------------|
| 1 | `type` | glioma / meningioma / notumor / pituitary | TCGA-LGG gives `{glioma, notumor}`; add a meningioma/pituitary source (e.g. the *Brain Tumor MRI Dataset*) for all four |
| 2 | `grade` | lgg / hgg | TCGA glioma-grade clinical metadata, or a BraTS LGG/HGG split |
| 3 | `stage` | grade_i … grade_iv | per-case WHO grade annotations |

Label vocabularies are kept **alphabetical** (`TYPE_LABELS`, `GRADE_LABELS`,
`STAGE_LABELS`) to match the class indices Keras assigns in `flow_from_dataframe`.
Pass your own ordered vocabularies to the builders/orchestrator if your generators
differ.

---

## Public API

| Function / class | Purpose |
|------------------|---------|
| `derive_type_labels_from_mask_csv(df, …)` | Bootstrap `{glioma, notumor}` `type` labels from the binary `mask` column |
| `class_weights_from_labels(labels)` | Balanced class weights (handles the ~2:1 TCGA-LGG imbalance) |
| `build_tumor_type_classifier(…)` | Level-1 model (ResNet50 or `custom_cnn`) |
| `build_glioma_grade_classifier(…)` | Level-2 model (LGG/HGG) |
| `build_tumor_stage_classifier(use_attention=True, …)` | Level-3 model (I–IV, optional SE attention) |
| `compile_classifier(model, …)` | Compile with the v2 notebook's Adam + label-smoothing + metrics |
| `build_type_data_generators(df, …)` | Train/val `flow_from_dataframe` generators matching the v2 augmentation |
| `MultiLevelTumorClassifier(...)` | Cascading inference engine; `.predict(img, preprocess=…)` → `MultiLevelPrediction` |
| `decide_hierarchy(type_probs, grade_probs, stage_probs, …)` | Pure decision logic over probability vectors |
| `grade_stage_consistent(grade, stage)` | WHO grade↔stage agreement check |
| `preprocess_for_classification(img, img_size)` | Train/serve-consistent preprocessing (matches `app.preprocess_image_classification`) |

`MultiLevelPrediction` is a dataclass with a `to_dict()` method shaped to slot into
`app.predict_tumor()`’s result payload.

---

## Testing

All decision logic, label derivation, preprocessing and orchestration are covered
by [`tests/unit/test_multilevel_classifier.py`](./tests/unit/test_multilevel_classifier.py)
and run **without TensorFlow** (matching the dependency-light stubs in
`tests/conftest.py`). The Keras builders are additionally verified under a real
TensorFlow install (`importorskip`, `weights=None`, so no ImageNet download).

```bash
pytest tests/unit/test_multilevel_classifier.py -v
```

TensorFlow is imported lazily inside the builder/training helpers, so the module
imports cleanly even when TensorFlow is not installed.

> ⚠️ **Research use only.** Like the rest of NeuroVision, this is not a clinical
> diagnostic tool and has not undergone regulatory validation.

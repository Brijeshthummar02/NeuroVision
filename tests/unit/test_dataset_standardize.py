import numpy as np
import pandas as pd
import cv2

import dataset_standardize as ds


# --- metadata / labeling helpers ----------------------------------------
def test_derive_patient_id_uses_folder():
    path = "TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1.tif"
    assert ds.derive_patient_id(path) == "TCGA_CS_4941_19960909"
    # windows separators are handled too
    assert ds.derive_patient_id("TCGA_DU_5849_19950405\\slice_3.tif") == "TCGA_DU_5849_19950405"


def test_derive_slice_index():
    assert ds.derive_slice_index("TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_34.tif") == 34
    assert ds.derive_slice_index("foo/bar_baz.tif") is None


# --- image standardization ----------------------------------------------
def test_standardize_image_grayscale_to_rgb():
    gray = np.full((40, 60), 128, dtype=np.uint8)
    out = ds.standardize_image(gray, target_size=(256, 256))

    assert out.shape == (256, 256, 3)
    assert out.dtype == np.uint8


def test_standardize_image_rgba_and_odd_size():
    rgba = np.zeros((100, 73, 4), dtype=np.uint8)
    out = ds.standardize_image(rgba, target_size=(128, 128))

    assert out.shape == (128, 128, 3)
    assert out.dtype == np.uint8


def test_standardize_image_uint16_is_scaled_to_uint8():
    img16 = (np.arange(32 * 32, dtype=np.uint16) % 4096).reshape(32, 32)
    out = ds.standardize_image(img16, target_size=(32, 32))

    assert out.dtype == np.uint8
    assert out.max() <= 255


# --- mask standardization -----------------------------------------------
def test_standardize_mask_is_binary_single_channel():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:20, 10:20] = 200  # a tumor blob

    out = ds.standardize_mask(mask, target_size=(64, 64))

    assert out.ndim == 2
    assert out.shape == (64, 64)
    # only {0, 255} may appear — nearest-neighbour resize never invents values
    assert set(np.unique(out)).issubset({0, 255})
    assert out.max() == 255


def test_standardize_mask_nearest_neighbour_no_intermediate_values():
    # a thin diagonal would smear into gray values under bilinear resize
    mask = np.zeros((48, 48), dtype=np.uint8)
    np.fill_diagonal(mask, 255)

    out = ds.standardize_mask(mask, target_size=(16, 16))

    assert set(np.unique(out)).issubset({0, 255})


def test_label_from_mask():
    empty = np.zeros((8, 8), dtype=np.uint8)
    tumor = empty.copy()
    tumor[3, 3] = 255

    assert ds.label_from_mask(empty) == 0
    assert ds.label_from_mask(tumor) == 1


# --- manifest standardization -------------------------------------------
def test_standardize_manifest_fixes_patient_id_and_columns():
    df = pd.DataFrame(
        {
            # deliberately wrong / constant patient_id, like the shipped CSV
            "patient_id": ["WRONG", "WRONG", "WRONG"],
            "image_path": [
                "TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1.tif",
                "TCGA_CS_4942_19970222/TCGA_CS_4942_19970222_2.tif",
                "TCGA_DU_5849_19950405/TCGA_DU_5849_19950405_5.tif",
            ],
            "mask_path": [
                "TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1_mask.tif",
                "TCGA_CS_4942_19970222/TCGA_CS_4942_19970222_2_mask.tif",
                "TCGA_DU_5849_19950405/TCGA_DU_5849_19950405_5_mask.tif",
            ],
            "mask": [0, 1, 0],
        }
    )

    out, summary = ds.standardize_manifest(df)

    assert list(out.columns) == ds.MANIFEST_COLUMNS
    assert out["patient_id"].tolist() == [
        "TCGA_CS_4941_19960909",
        "TCGA_CS_4942_19970222",
        "TCGA_DU_5849_19950405",
    ]
    assert out["slice"].tolist() == [1, 2, 5]
    assert summary["patient_id_corrected"] == 3
    assert summary["unique_patients"] == 3
    # original label preserved for traceability
    assert out["original_mask"].tolist() == [0, 1, 0]


def test_standardize_manifest_missing_column_raises():
    df = pd.DataFrame({"image_path": ["a/b_1.tif"]})  # no mask_path
    try:
        ds.standardize_manifest(df)
        assert False, "expected KeyError"
    except KeyError:
        pass


# --- end-to-end pixel processing ----------------------------------------
def _write_pair(root, patient, idx, mask_value):
    folder = root / patient
    folder.mkdir(parents=True, exist_ok=True)
    img_rel = f"{patient}/{patient}_{idx}.tif"
    mask_rel = f"{patient}/{patient}_{idx}_mask.tif"

    rng = np.random.default_rng(idx)
    cv2.imwrite(str(root / img_rel), (rng.random((90, 110)) * 255).astype(np.uint8))

    mask = np.zeros((90, 110), dtype=np.uint8)
    if mask_value:
        mask[20:40, 20:40] = 255
    cv2.imwrite(str(root / mask_rel), mask)
    return img_rel, mask_rel


def test_process_pixels_end_to_end(tmp_path):
    img1, mask1 = _write_pair(tmp_path, "TCGA_CS_4941_19960909", 1, mask_value=True)
    img2, mask2 = _write_pair(tmp_path, "TCGA_CS_4942_19970222", 2, mask_value=False)

    df = pd.DataFrame(
        {
            "patient_id": ["x", "x"],
            "image_path": [img1, img2],
            "mask_path": [mask1, mask2],
            # original labels are BOTH wrong on purpose -> must be corrected
            "mask": [0, 1],
        }
    )
    manifest, _ = ds.standardize_manifest(df)

    out_dir = tmp_path / "std"
    manifest, summary = ds.process_pixels(
        manifest,
        img_root=str(tmp_path),
        out_dir=str(out_dir),
        target_size=(128, 128),
    )

    # both rows processed, none missing
    assert summary["processed"] == 2
    assert summary["missing_files"] == 0

    # labels re-derived from mask content
    assert manifest["mask"].tolist() == [1, 0]
    assert summary["mask_label_corrected"] == 2

    # standardized files actually written, at the canonical resolution
    for rel in manifest["std_image_path"]:
        f = out_dir / rel
        assert f.exists()
        assert cv2.imread(str(f)).shape == (128, 128, 3)

    # masks on disk are binary {0,255}
    for rel in manifest["std_mask_path"]:
        m = cv2.imread(str(out_dir / rel), cv2.IMREAD_GRAYSCALE)
        assert set(np.unique(m)).issubset({0, 255})

    # consistency check passes
    cc = summary["consistency_check"]
    assert cc["uniform_resolution"] and cc["uniform_dtype"]
    assert cc["resolution"] == "128x128"


def test_process_pixels_handles_missing_files(tmp_path):
    df = pd.DataFrame(
        {
            "patient_id": ["x"],
            "image_path": ["TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1.tif"],
            "mask_path": ["TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1_mask.tif"],
            "mask": [0],
        }
    )
    manifest, _ = ds.standardize_manifest(df)

    manifest, summary = ds.process_pixels(
        manifest, img_root=str(tmp_path), out_dir=str(tmp_path / "std")
    )

    # nothing on disk -> skipped gracefully, manifest still intact
    assert summary["processed"] == 0
    assert summary["missing_files"] == 1
    assert manifest["std_image_path"].tolist() == [""]

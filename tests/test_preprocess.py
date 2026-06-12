"""
NeuroVision — Preprocessing Unit Tests
=======================================
Run with:
    pytest tests/test_preprocess.py -v
"""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image
import tempfile, os

# ── import the module under test ──────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocess import (
    normalise_to_uint8,
    binarise_mask,
    resize_image,
    resize_mask,
    has_tumour,
    compute_mask_stats,
    standardise_image,
    standardise_mask,
)

IMG_SIZE = 256


# ─── normalise_to_uint8 ───────────────────────────────────────────────────────

class TestNormaliseToUint8:
    def test_clips_below_zero(self):
        arr = np.array([[[-10.0, 0.0, 300.0]]], dtype=np.float32)
        out = normalise_to_uint8(arr)
        assert out[0, 0, 0] == 0
        assert out[0, 0, 2] == 255

    def test_dtype_is_uint8(self):
        arr = np.ones((4, 4, 3), dtype=np.float32) * 128
        out = normalise_to_uint8(arr)
        assert out.dtype == np.uint8

    def test_identity_on_valid_range(self):
        arr = np.array([[[0.0, 127.0, 255.0]]], dtype=np.float32)
        out = normalise_to_uint8(arr)
        np.testing.assert_array_equal(out, [[[0, 127, 255]]])


# ─── binarise_mask ────────────────────────────────────────────────────────────

class TestBinariseMask:
    def test_0_255_input_stays_binary(self):
        mask = np.array([[0.0, 255.0, 0.0, 255.0]], dtype=np.float32)
        out = binarise_mask(mask)
        assert set(out.flatten().tolist()) == {0, 255}

    def test_soft_edges_binarised(self):
        mask = np.array([[0.0, 128.0, 200.0, 50.0]], dtype=np.float32)
        out = binarise_mask(mask)
        # 128 (>0.5*255) → 255; 50 (<0.5*255) → 0
        assert out[0, 1] == 255
        assert out[0, 3] == 0

    def test_all_zero_mask(self):
        mask = np.zeros((16, 16), dtype=np.float32)
        out = binarise_mask(mask)
        assert out.max() == 0

    def test_all_positive_mask(self):
        mask = np.ones((16, 16), dtype=np.float32) * 255
        out = binarise_mask(mask)
        assert out.min() == 255
        assert out.max() == 255

    def test_probability_map_input(self):
        mask = np.array([[0.0, 0.3, 0.6, 1.0]], dtype=np.float32)
        out = binarise_mask(mask, threshold=0.5)
        assert out[0, 0] == 0
        assert out[0, 1] == 0
        assert out[0, 2] == 255
        assert out[0, 3] == 255


# ─── resize helpers ───────────────────────────────────────────────────────────

class TestResize:
    def test_image_resize_shape(self):
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8).astype(np.float32)
        out = resize_image(img, IMG_SIZE)
        assert out.shape == (IMG_SIZE, IMG_SIZE, 3)

    def test_mask_resize_preserves_binary(self):
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        out = resize_mask(mask, IMG_SIZE)
        unique = set(out.flatten().tolist())
        # Nearest-neighbour resize must not introduce intermediate values
        assert unique.issubset({0, 255}), f"Unexpected values: {unique - {0, 255}}"

    def test_mask_resize_shape(self):
        mask = np.zeros((512, 512), dtype=np.uint8)
        out = resize_mask(mask, IMG_SIZE)
        assert out.shape == (IMG_SIZE, IMG_SIZE)


# ─── has_tumour / compute_mask_stats ─────────────────────────────────────────

class TestMaskStats:
    def test_has_tumour_true(self):
        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        mask[50, 50] = 255
        assert has_tumour(mask) is True

    def test_has_tumour_false(self):
        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        assert has_tumour(mask) is False

    def test_compute_mask_stats_zero_mask(self):
        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        stats = compute_mask_stats(mask)
        assert stats["tumour_pixels"] == 0
        assert stats["tumour_fraction"] == 0.0
        assert stats["background_pixels"] == IMG_SIZE * IMG_SIZE

    def test_compute_mask_stats_full_mask(self):
        mask = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
        stats = compute_mask_stats(mask)
        assert stats["tumour_pixels"] == IMG_SIZE * IMG_SIZE
        assert stats["background_pixels"] == 0
        assert stats["tumour_fraction"] == 1.0


# ─── end-to-end standardise (using temp PNG files) ───────────────────────────

class TestStandardiseFunctions:
    def _write_tiff(self, arr, path):
        Image.fromarray(arr).save(path)

    def test_standardise_image_rgb_output(self):
        raw = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmp = Path(f.name)
        try:
            self._write_tiff(raw, tmp)
            out = standardise_image(tmp, IMG_SIZE)
            assert out.shape == (IMG_SIZE, IMG_SIZE, 3)
            assert out.dtype == np.uint8
        finally:
            os.unlink(tmp)

    def test_standardise_image_grayscale_promoted_to_rgb(self):
        raw = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmp = Path(f.name)
        try:
            self._write_tiff(raw, tmp)
            out = standardise_image(tmp, IMG_SIZE)
            assert out.shape == (IMG_SIZE, IMG_SIZE, 3)
        finally:
            os.unlink(tmp)

    def test_standardise_mask_binary_output(self):
        raw = np.zeros((512, 512), dtype=np.uint8)
        raw[100:300, 100:300] = 200   # soft value — should be binarised to 255
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmp = Path(f.name)
        try:
            self._write_tiff(raw, tmp)
            out = standardise_mask(tmp, IMG_SIZE)
            unique = set(out.flatten().tolist())
            assert unique.issubset({0, 255}), f"Non-binary values in mask: {unique}"
            assert out.shape == (IMG_SIZE, IMG_SIZE)
            assert out.dtype == np.uint8
        finally:
            os.unlink(tmp)

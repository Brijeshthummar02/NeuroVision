import importlib.util
import os
import sys
import types

import cv2
import numpy as np
import pytest

REQUIRED_ENV_VARS = [
    "FLASK_SECRET_KEY",
    "CLOUDINARY_CLOUD_NAME",
    "CLOUDINARY_API_KEY",
    "CLOUDINARY_API_SECRET",
]

for key in REQUIRED_ENV_VARS:
    os.environ.setdefault(key, "test")


def _install_tensorflow_stub():
    if importlib.util.find_spec("tensorflow") is not None:
        return

    tf = types.ModuleType("tensorflow")
    keras = types.ModuleType("tensorflow.keras")
    utils = types.ModuleType("tensorflow.keras.utils")
    backend = types.ModuleType("tensorflow.keras.backend")
    losses = types.ModuleType("tensorflow.keras.losses")
    optimizers = types.ModuleType("tensorflow.keras.optimizers")
    metrics = types.ModuleType("tensorflow.keras.metrics")
    models = types.ModuleType("tensorflow.keras.models")

    class DummySequence:
        pass

    class DummyModel:
        def __init__(self):
            self.compiled = False

        def load_weights(self, *args, **kwargs):
            return None

        def compile(self, *args, **kwargs):
            self.compiled = True

        def predict(self, img, verbose=0):
            return np.array([[0.5, 0.5]], dtype=np.float32)

    class DummyLoss:
        def __init__(self, *args, **kwargs):
            pass

    class DummyOptimizer:
        def __init__(self, *args, **kwargs):
            pass

    class DummyMetric:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get("name")

    def _sum(x, axis=None):
        if isinstance(axis, list):
            axis = tuple(axis)
        return np.sum(x, axis=axis)

    def _mean(x, axis=None):
        if isinstance(axis, list):
            axis = tuple(axis)
        return np.mean(x, axis=axis)

    backend.flatten = lambda x: np.ravel(x)
    backend.sum = _sum
    backend.mean = _mean
    backend.pow = np.power
    backend.abs = np.abs
    backend.clip = np.clip
    backend.round = np.round
    backend.epsilon = lambda: 1e-7

    def binary_crossentropy(y_true, y_pred):
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    losses.binary_crossentropy = binary_crossentropy
    losses.CategoricalCrossentropy = DummyLoss
    optimizers.Adam = DummyOptimizer
    metrics.Precision = DummyMetric
    metrics.Recall = DummyMetric
    metrics.AUC = DummyMetric

    utils.Sequence = DummySequence
    models.model_from_json = lambda *args, **kwargs: DummyModel()
    models.load_model = lambda *args, **kwargs: DummyModel()

    keras.utils = utils
    keras.backend = backend
    keras.losses = losses
    keras.optimizers = optimizers
    keras.metrics = metrics
    keras.models = models
    keras.Model = DummyModel

    tf.keras = keras

    sys.modules["tensorflow"] = tf
    sys.modules["tensorflow.keras"] = keras
    sys.modules["tensorflow.keras.utils"] = utils
    sys.modules["tensorflow.keras.backend"] = backend
    sys.modules["tensorflow.keras.losses"] = losses
    sys.modules["tensorflow.keras.optimizers"] = optimizers
    sys.modules["tensorflow.keras.metrics"] = metrics
    sys.modules["tensorflow.keras.models"] = models


def _install_seaborn_stub():
    if importlib.util.find_spec("seaborn") is None:
        sys.modules["seaborn"] = types.ModuleType("seaborn")


def _install_skimage_stub():
    if importlib.util.find_spec("skimage") is not None:
        return

    skimage = types.ModuleType("skimage")
    io_module = types.ModuleType("skimage.io")

    def imread(path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        return img

    io_module.imread = imread
    skimage.io = io_module

    sys.modules["skimage"] = skimage
    sys.modules["skimage.io"] = io_module


_install_tensorflow_stub()
_install_seaborn_stub()
_install_skimage_stub()


@pytest.fixture(scope="session")
def app_module():
    import app
    return app


@pytest.fixture()
def app_state(app_module, monkeypatch, tmp_path):
    app_module.app.config.update(TESTING=True)

    upload_dir = tmp_path / "uploads"
    history_dir = tmp_path / "scan_history"
    upload_dir.mkdir()
    history_dir.mkdir()

    app_module.app.config["UPLOAD_FOLDER"] = str(upload_dir)
    app_module.app.config["SCAN_HISTORY_FOLDER"] = str(history_dir)

    app_module.models_loaded = True
    app_module.classification_models = []
    app_module.segmentation_models = []
    app_module.memory_storage = {"users": {}, "scan_history": {}}

    monkeypatch.setattr(app_module, "STORAGE_FILE", str(tmp_path / "storage.json"), raising=False)
    monkeypatch.setattr(app_module, "mongodb_available", False, raising=False)
    monkeypatch.setattr(app_module, "mongodb_init_attempted", True, raising=False)
    monkeypatch.setattr(app_module, "save_storage", lambda data: None)

    return app_module


@pytest.fixture()
def client(app_state):
    return app_state.app.test_client()


@pytest.fixture()
def sample_image_array():
    img = np.full((64, 64, 3), [10, 20, 30], dtype=np.uint8)
    return img


@pytest.fixture()
def sample_image_bytes(sample_image_array):
    ok, buf = cv2.imencode(".png", sample_image_array)
    assert ok
    return buf.tobytes()

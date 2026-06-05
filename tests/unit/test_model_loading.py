def test_get_custom_objects_includes_core_metrics(app_state):
    custom_objects = app_state.get_custom_objects()

    required_keys = {
        "Functional",
        "tversky",
        "tversky_loss",
        "focal_tversky",
        "dice_coefficient",
        "dice_loss",
        "bce_dice_loss",
        "iou_score",
        "sensitivity",
        "specificity",
        "precision_metric",
    }

    assert required_keys.issubset(custom_objects.keys())


def test_load_models_with_stubs(app_state, monkeypatch):
    class DummyModel:
        def __init__(self):
            self.loaded = False
            self.compiled = False

        def load_weights(self, path):
            self.loaded = True

        def compile(self, *args, **kwargs):
            self.compiled = True

    monkeypatch.setattr(
        app_state.tf.keras.models,
        "model_from_json",
        lambda *args, **kwargs: DummyModel(),
    )
    monkeypatch.setattr(
        app_state.tf.keras.models,
        "load_model",
        lambda *args, **kwargs: DummyModel(),
    )

    app_state.classification_models = []
    app_state.segmentation_models = []
    app_state.models_loaded = False

    assert app_state.load_models() is True
    assert app_state.models_loaded is True
    assert len(app_state.classification_models) >= 1
    assert len(app_state.segmentation_models) >= 1

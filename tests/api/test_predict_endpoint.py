import base64
import io


def _fake_prediction():
    dummy = base64.b64encode(b"mask").decode("ascii")
    return {
        "has_tumor": True,
        "confidence": 0.91,
        "classification_scores": {"no_tumor": 0.09, "tumor": 0.91},
        "analysis_method": {
            "tta_enabled": False,
            "ensemble_enabled": False,
            "classification_models_used": 1,
            "segmentation_models_used": 1,
        },
        "segmentation": {
            "mask": f"data:image/png;base64,{dummy}",
            "overlay": f"data:image/png;base64,{dummy}",
            "tumor_area_percentage": 1.5,
            "tumor_pixels": 100,
            "total_pixels": 256 * 256,
            "bounding_box": {
                "x_min": 10,
                "y_min": 10,
                "x_max": 20,
                "y_max": 20,
                "width": 10,
                "height": 10,
            },
            "centroid": {"x": 15, "y": 15},
            "mask_confidence": 0.6,
        },
        "severity_assessment": {
            "level": "Low",
            "severity_color": "#10b981",
            "urgency": "Routine",
            "tumor_coverage": "1.50%",
            "recommendation": "Follow-up with primary care physician",
        },
        "detailed_report": {
            "scan_id": "TEST1234",
            "scan_date": "2026-01-01T00:00:00",
            "analysis_metadata": {"ai_version": "test"},
        },
    }


def test_predict_missing_file_returns_400(client, app_state):
    response = client.post("/api/predict")

    assert response.status_code == 400
    assert response.get_json()["error"] == "No file uploaded"


def test_predict_empty_filename_returns_400(client, app_state, sample_image_bytes):
    data = {
        "file": (io.BytesIO(sample_image_bytes), "")
    }

    response = client.post("/api/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.get_json()["error"] == "No file selected"


def test_predict_invalid_extension_returns_400(client, app_state, sample_image_bytes):
    data = {
        "file": (io.BytesIO(sample_image_bytes), "scan.txt")
    }

    response = client.post("/api/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 400
    assert "Invalid file type" in response.get_json()["error"]


def test_predict_models_not_loaded_returns_500(client, app_state, sample_image_bytes):
    app_state.models_loaded = False

    data = {
        "file": (io.BytesIO(sample_image_bytes), "scan.png")
    }

    response = client.post("/api/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 500
    assert "Models not loaded" in response.get_json()["error"]


def test_predict_success_returns_payload(client, app_state, sample_image_bytes, monkeypatch):
    monkeypatch.setattr(app_state, "predict_tumor", lambda *args, **kwargs: _fake_prediction())

    data = {
        "file": (io.BytesIO(sample_image_bytes), "scan.png")
    }

    response = client.post("/api/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["has_tumor"] is True
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["segmentation"]["mask"].startswith("data:image/png;base64,")
    assert payload["original_image"].startswith("data:image/png;base64,")

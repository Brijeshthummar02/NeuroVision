import base64
import io


def _login(client):
    with client.session_transaction() as session:
        session["user_id"] = "user-1"
        session["user_email"] = "user@example.com"
        session["user_name"] = "User"


def _fake_prediction():
    dummy = base64.b64encode(b"mask").decode("ascii")
    return {
        "has_tumor": True,
        "confidence": 0.93,
        "classification_scores": {"no_tumor": 0.07, "tumor": 0.93},
        "analysis_method": {
            "tta_enabled": False,
            "ensemble_enabled": False,
            "classification_models_used": 1,
            "segmentation_models_used": 1,
        },
        "segmentation": {
            "mask": f"data:image/png;base64,{dummy}",
            "overlay": f"data:image/png;base64,{dummy}",
            "tumor_area_percentage": 2.4,
            "tumor_pixels": 180,
            "total_pixels": 256 * 256,
            "bounding_box": {
                "x_min": 12,
                "y_min": 12,
                "x_max": 24,
                "y_max": 24,
                "width": 12,
                "height": 12,
            },
            "centroid": {"x": 18, "y": 18},
            "mask_confidence": 0.7,
        },
        "severity_assessment": {
            "level": "Low",
            "severity_color": "#10b981",
            "urgency": "Routine",
            "tumor_coverage": "2.40%",
            "recommendation": "Follow-up with primary care physician",
        },
        "detailed_report": {
            "scan_id": "TESTFLOW",
            "scan_date": "2026-01-01T00:00:00",
            "analysis_metadata": {"ai_version": "test"},
        },
    }


def test_prediction_and_history_flow(client, app_state, sample_image_bytes, monkeypatch):
    _login(client)

    monkeypatch.setattr(app_state, "predict_tumor", lambda *args, **kwargs: _fake_prediction())
    monkeypatch.setattr(
        app_state,
        "upload_image_to_cloudinary",
        lambda *args, **kwargs: {"url": "http://example.com/img.png", "public_id": "pid"},
    )

    data = {
        "file": (io.BytesIO(sample_image_bytes), "scan.png")
    }
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200

    prediction = response.get_json()

    response = client.post("/api/history/save", json=prediction)
    assert response.status_code == 200

    response = client.get("/api/history")
    assert response.status_code == 200
    assert response.get_json()["total"] == 1

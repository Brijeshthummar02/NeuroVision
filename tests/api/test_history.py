import base64


def _login(client):
    with client.session_transaction() as session:
        session["user_id"] = "user-1"
        session["user_email"] = "user@example.com"
        session["user_name"] = "User"


def _payload():
    dummy = base64.b64encode(b"image").decode("ascii")
    return {
        "has_tumor": True,
        "confidence": 0.88,
        "original_image": f"data:image/png;base64,{dummy}",
        "segmentation": {
            "mask": f"data:image/png;base64,{dummy}",
            "overlay": f"data:image/png;base64,{dummy}",
            "tumor_area_percentage": 2.0,
        },
        "severity_assessment": {"level": "Low"},
    }


def test_save_scan_requires_login(client, app_state):
    response = client.post("/api/history/save", json=_payload())

    assert response.status_code == 401
    assert response.get_json()["authenticated"] is False


def test_history_flow_with_file_storage(client, app_state, monkeypatch):
    _login(client)

    monkeypatch.setattr(
        app_state,
        "upload_image_to_cloudinary",
        lambda *args, **kwargs: {"url": "http://example.com/img.png", "public_id": "pid"},
    )
    monkeypatch.setattr(app_state, "delete_image_from_cloudinary", lambda *args, **kwargs: True)

    response = client.post("/api/history/save", json=_payload())
    assert response.status_code == 200

    scan_id = response.get_json()["scan_id"]

    response = client.get("/api/history")
    assert response.status_code == 200
    history = response.get_json()
    assert history["total"] == 1

    response = client.get("/api/history/does-not-exist")
    assert response.status_code == 404

    response = client.delete(f"/api/history/{scan_id}")
    assert response.status_code == 200
    assert response.get_json()["success"] is True

def test_health_check_returns_status(client, app_state):
    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True


def test_config_get_and_update(client, app_state):
    response = client.get("/api/config")
    assert response.status_code == 200

    data = response.get_json()
    assert data["use_tta"] == app_state.app.config["USE_TTA"]

    update = {
        "use_tta": False,
        "use_ensemble": False,
        "confidence_threshold": 0.7,
    }
    response = client.post("/api/config", json=update)

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "updated"
    assert data["use_tta"] is False
    assert data["use_ensemble"] is False
    assert data["confidence_threshold"] == 0.7

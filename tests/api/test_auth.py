def test_signup_and_login_flow(client, app_state):
    response = client.post(
        "/api/auth/signup",
        json={"email": "user@example.com", "password": "secret1", "name": "User"},
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True

    response = client.get("/api/auth/status")
    assert response.status_code == 200
    assert response.get_json()["authenticated"] is True

    response = client.post("/api/auth/logout")
    assert response.status_code == 200
    assert response.get_json()["success"] is True

    response = client.post(
        "/api/auth/login",
        json={"email": "user@example.com", "password": "wrong"},
    )

    assert response.status_code == 401

    response = client.post(
        "/api/auth/login",
        json={"email": "user@example.com", "password": "secret1"},
    )

    assert response.status_code == 200
    assert response.get_json()["success"] is True


def test_signup_validation_errors(client, app_state):
    response = client.post("/api/auth/signup", json={})
    assert response.status_code == 400

    response = client.post(
        "/api/auth/signup",
        json={"email": "invalid", "password": "123456", "name": "User"},
    )
    assert response.status_code == 400

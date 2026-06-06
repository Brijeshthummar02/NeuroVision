import pytest


def test_validate_required_env_missing_raises(app_state, monkeypatch):
    monkeypatch.delenv("CLOUDINARY_API_KEY", raising=False)

    with pytest.raises(EnvironmentError):
        app_state.validate_required_env()

import asyncio
import json
from unittest.mock import MagicMock

from cogito.api.handlers import create_health_check_handler
from cogito.core.utils import readiness_context


def _call_health_check(readiness_file: str):
    handler = create_health_check_handler(readiness_file)
    return asyncio.run(handler(MagicMock()))


def test_health_check_returns_ok_when_readiness_file_is_ready(tmp_path):
    readiness_file = tmp_path / "readiness.lock"
    readiness_file.write_text("ready")

    response = _call_health_check(str(readiness_file))

    assert response.status_code == 200
    assert json.loads(response.body) == {"status": "OK"}


def test_health_check_returns_503_when_readiness_file_is_missing(tmp_path):
    readiness_file = tmp_path / "readiness.lock"

    response = _call_health_check(str(readiness_file))

    assert response.status_code == 503
    assert json.loads(response.body) == {
        "status": "ERROR",
        "message": "Service is not ready",
    }


def test_health_check_returns_503_when_readiness_file_content_is_invalid(tmp_path):
    readiness_file = tmp_path / "readiness.lock"
    readiness_file.write_text("not-ready")

    response = _call_health_check(str(readiness_file))

    assert response.status_code == 503


def test_health_check_returns_503_when_readiness_file_is_a_directory(tmp_path):
    readiness_file = tmp_path / "readiness.lock"
    readiness_file.mkdir()

    response = _call_health_check(str(readiness_file))

    assert response.status_code == 503


def test_health_check_expands_env_vars_in_readiness_file_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "readiness.lock").write_text("ready")

    response = _call_health_check("$HOME/readiness.lock")

    assert response.status_code == 200


def test_health_check_expands_user_home_in_readiness_file_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "readiness.lock").write_text("ready")

    response = _call_health_check("~/readiness.lock")

    assert response.status_code == 200


def test_health_check_follows_readiness_context_lifecycle(tmp_path):
    readiness_file = str(tmp_path / "readiness.lock")

    assert _call_health_check(readiness_file).status_code == 503

    with readiness_context(readiness_file):
        assert _call_health_check(readiness_file).status_code == 200

    assert _call_health_check(readiness_file).status_code == 503

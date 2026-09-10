import os

import pytest

from cogito.core.utils import get_readiness_file_path, is_ready, readiness_context


def test_readiness_context_creates_file_and_removes_it_on_exit(tmp_path):
    readiness_file = str(tmp_path / "readiness.lock")

    with readiness_context(readiness_file):
        assert os.path.isfile(readiness_file)

    assert not os.path.exists(readiness_file)


def test_is_ready_agrees_with_readiness_context(tmp_path):
    readiness_file = str(tmp_path / "readiness.lock")

    assert is_ready(readiness_file) is False

    with readiness_context(readiness_file):
        assert is_ready(readiness_file) is True

    assert is_ready(readiness_file) is False


def test_readiness_context_accepts_a_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    readiness_file = ".cogito-readiness.lock"

    with readiness_context(readiness_file):
        assert os.path.isfile(readiness_file)

    assert not os.path.exists(readiness_file)


def test_readiness_context_removes_file_when_body_raises(tmp_path):
    readiness_file = str(tmp_path / "readiness.lock")

    with pytest.raises(RuntimeError):
        with readiness_context(readiness_file):
            assert os.path.isfile(readiness_file)
            raise RuntimeError("boom")

    assert not os.path.exists(readiness_file)


def test_get_readiness_file_path_expands_env_vars_and_user_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    assert get_readiness_file_path("$HOME/readiness.lock") == str(
        tmp_path / "readiness.lock"
    )
    assert get_readiness_file_path("~/readiness.lock") == str(
        tmp_path / "readiness.lock"
    )


def test_readiness_context_does_not_raise_when_file_removed_externally(tmp_path):
    """An operator manually draining traffic (per the README) removes the
    readiness file directly; the context manager must not blow up on exit."""
    readiness_file = str(tmp_path / "readiness.lock")

    with readiness_context(readiness_file):
        os.remove(readiness_file)

    assert not os.path.exists(readiness_file)


def test_readiness_context_propagates_original_error_when_file_also_missing(tmp_path):
    """The original exception must not be masked by the cleanup's own
    FileNotFoundError when both happen together."""
    readiness_file = str(tmp_path / "readiness.lock")

    with pytest.raises(RuntimeError, match="boom"):
        with readiness_context(readiness_file):
            os.remove(readiness_file)
            raise RuntimeError("boom")

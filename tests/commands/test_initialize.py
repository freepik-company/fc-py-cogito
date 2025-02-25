import os
import pytest
from click.testing import CliRunner
from unittest.mock import patch

from cogito.commands.initialize import init
from cogito.core.config import ConfigFile


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def clean_config():
    # Remove config file if exists before each test
    if os.path.exists("cogito.yaml"):
        os.remove("cogito.yaml")
    yield
    # Cleanup after test
    if os.path.exists("cogito.yaml"):
        os.remove("cogito.yaml")


def test_init_default(runner, clean_config):
    result = runner.invoke(init, ["--default"])
    assert result.exit_code == 0
    assert "Initializing..." in result.output
    assert os.path.exists("./cogito.yaml")

    # Verify config file was created with default values as per cogito.yaml
    config = ConfigFile.load_from_file("./cogito.yaml")

    # Server-level defaults
    assert config.cogito.server.name == "Cogito ergo sum"
    assert config.cogito.server.version == "0.1.0"
    assert config.cogito.server.cache_dir == "/tmp"
    assert config.cogito.server.description == "Inference server"
    assert config.cogito.server.threads == 1

    # FastAPI defaults
    assert config.cogito.server.fastapi.host == "0.0.0.0"
    assert config.cogito.server.fastapi.port == 8000
    assert config.cogito.server.readiness_file == "$HOME/readiness.lock"
    assert config.cogito.server.fastapi.access_log is False
    assert config.cogito.server.fastapi.debug is False

    # Route defaults (note: args and response are no longer present)
    assert config.cogito.server.route.name == "Predict"
    assert config.cogito.server.route.description == "Make a single prediction"
    assert config.cogito.server.route.path == "/v1/predict"
    assert config.cogito.server.route.predictor == "predict:Predictor"
    assert config.cogito.server.route.tags == ["predict"]


def test_init_prompted(runner, clean_config):
    # Mock user input values
    inputs = [
        "Test Project",  # name
        "Test Description",  # description
        "0.1.0",  # version
        "localhost",  # host
        "8080",  # port
        "n",  # debug mode
        "n",  # access log
        "y",  # add default route
        "/tmp/cache",  # cache_dir
        ".cogito-readiness.lock",  # readiness file
    ]

    result = runner.invoke(init, input="\n".join(inputs))
    assert result.exit_code == 0
    assert "Initializing..." in result.output
    assert os.path.exists("cogito.yaml")

    # Verify config file was created with prompted values
    config = ConfigFile.load_from_file("./cogito.yaml")
    assert config.cogito.server.name == "Test Project"
    assert config.cogito.server.description == "Test Description"
    assert config.cogito.server.version == "0.1.0"
    assert config.cogito.server.fastapi.host == "localhost"
    assert config.cogito.server.fastapi.port == 8080
    assert config.cogito.server.fastapi.debug is False
    assert config.cogito.server.fastapi.access_log is False
    assert config.cogito.server.cache_dir == "/tmp/cache"

    # Route defaults remain (since the prompted version still uses the default values)
    assert config.cogito.server.route.name == "Predict"
    assert config.cogito.server.route.description == "Make a single prediction"
    assert config.cogito.server.route.path == "/v1/predict"
    assert config.cogito.server.route.predictor == "predict:Predictor"
    assert config.cogito.server.route.tags == ["predict"]
    assert config.cogito.server.readiness_file == ".cogito-readiness.lock"


def test_init_already_exists(runner, clean_config):
    # Create initial config
    runner.invoke(init, ["--default"])

    # Try to initialize again
    result = runner.invoke(init)
    assert result.exit_code == 0
    assert "Already initialized." in result.output


def test_init_force(runner, clean_config):
    # Create initial config
    runner.invoke(init, ["--default"])

    # Force new initialization
    result = runner.invoke(init, ["--force", "--default"])
    assert result.exit_code == 0
    assert "Already initialized." not in result.output
    assert "Initializing..." in result.output

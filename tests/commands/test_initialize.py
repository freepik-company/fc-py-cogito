import os
import pytest
import tempfile

import yaml
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch

from cogito.commands.initialize import init
from cogito.core.config.file import ConfigFile


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def temp_dir():
    # Create a temporary directory for config files
    with tempfile.TemporaryDirectory() as temp_dir:
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(original_dir)


def test_init_default(runner, temp_dir):
    result = runner.invoke(init, ["--default"])
    assert result.exit_code == 0
    assert "Initializing..." in result.output
    assert os.path.exists("cogito.yaml")

    # Verify config file was created with default values as per cogito.yaml
    config = ConfigFile.load_from_file("cogito.yaml")
    assert config.config_version == 2

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

    # Predictor defaults
    assert len(config.cogito.predictors) == 1
    predictor = config.cogito.predictors[0]
    assert predictor.name == "predictor"
    assert predictor.class_path == "predict:Predictor"
    assert predictor.threads == 1
    assert predictor.resources.gpus == 0
    assert predictor.route.name == "Predict"
    assert predictor.route.description == "Make a single prediction"
    assert predictor.route.path == "/v1/predict"
    assert predictor.route.tags == ["predict"]


def test_init_prompted(runner, temp_dir):
    # Mock user input values
    inputs = [
        "Test Project",  # name
        "Test Description",  # description
        "0.1.0",  # version
        "localhost",  # host
        "8080",  # port
        "n",  # debug mode
        "n",  # access log
        "",  # predictor name (default)
        "",  # predictor class (default)
        "",  # route path (default)
        "",  # threads (default)
        "n",  # add another predictor
        "/tmp/cache",  # cache_dir
        ".cogito-readiness.lock",  # readiness file
    ]

    result = runner.invoke(init, input="\n".join(inputs))
    assert result.exit_code == 0
    assert "Initializing..." in result.output
    assert os.path.exists("cogito.yaml")

    # Verify config file was created with prompted values
    config = ConfigFile.load_from_file("cogito.yaml")
    assert config.cogito.server.name == "Test Project"
    assert config.cogito.server.description == "Test Description"
    assert config.cogito.server.version == "0.1.0"
    assert config.cogito.server.fastapi.host == "localhost"
    assert config.cogito.server.fastapi.port == 8080
    assert config.cogito.server.fastapi.debug is False
    assert config.cogito.server.fastapi.access_log is False
    assert config.cogito.server.cache_dir == "/tmp/cache"
    assert config.config_version == 2

    # Predictor defaults remain (the prompts were answered with the defaults)
    assert len(config.cogito.predictors) == 1
    predictor = config.cogito.predictors[0]
    assert predictor.name == "predictor"
    assert predictor.class_path == "predict:Predictor"
    assert predictor.threads == 1
    assert predictor.route.name == "Predict"
    assert predictor.route.description == "Make a single prediction"
    assert predictor.route.path == "/v1/predict"
    assert predictor.route.tags == ["predict"]
    assert config.cogito.server.readiness_file == ".cogito-readiness.lock"


def test_init_already_exists(runner, temp_dir):
    # Create initial config
    runner.invoke(init, ["--default"])

    # Try to initialize again
    result = runner.invoke(init)
    assert result.exit_code == 0
    assert "Already initialized." in result.output


def test_init_force(runner, temp_dir):
    # Create initial config
    runner.invoke(init, ["--default"])

    # Force new initialization
    result = runner.invoke(init, ["--force", "--default"])
    assert result.exit_code == 0
    assert "Already initialized." not in result.output
    assert "Initializing..." in result.output


SERVER_INPUTS = [
    "Test Project",  # name
    "Test Description",  # description
    "0.1.0",  # version
    "localhost",  # host
    "8080",  # port
    "n",  # debug mode
    "n",  # access log
]
FILE_INPUTS = [
    "/tmp/cache",  # cache_dir
    ".cogito-readiness.lock",  # readiness file
]


def test_init_prompted_several_predictors(runner, temp_dir):
    inputs = (
        SERVER_INPUTS
        + [
            "sdxl",  # predictor name
            "predict_sdxl:Predictor",  # predictor class
            "/v1/sdxl",  # route path
            "2",  # threads
            "y",  # add another predictor
            "clip",  # predictor name
            "",  # predictor class (default: predict_clip:Predictor)
            "",  # route path (default: /v1/clip)
            "",  # threads (default)
            "n",  # add another predictor
        ]
        + FILE_INPUTS
    )

    result = runner.invoke(init, input="\n".join(inputs))
    assert result.exit_code == 0, result.output

    config = ConfigFile.load_from_file("cogito.yaml")
    assert config.config_version == 2
    assert config.cogito.get_predictor_names == ["sdxl", "clip"]

    sdxl = config.cogito.get_predictor_config("sdxl")
    assert sdxl.class_path == "predict_sdxl:Predictor"
    assert sdxl.route.path == "/v1/sdxl"
    assert sdxl.threads == 2

    clip = config.cogito.get_predictor_config("clip")
    assert clip.class_path == "predict_clip:Predictor"
    assert clip.route.path == "/v1/clip"
    assert clip.route.name == "clip"
    assert clip.route.tags == ["clip"]
    assert clip.threads == 1


def test_init_prompted_rejects_duplicates_and_bad_class(runner, temp_dir):
    inputs = (
        SERVER_INPUTS
        + [
            "",  # predictor name (default: predictor)
            "",  # predictor class (default)
            "",  # route path (default: /v1/predict)
            "",  # threads (default)
            "y",  # add another predictor
            "predictor",  # duplicate name, asked again
            "other",  # predictor name
            "no_colon",  # invalid class, asked again
            "predict_other:Other",  # predictor class
            "/v1/predict",  # duplicate route path, asked again
            "/v1/other",  # route path
            "0",  # invalid threads, asked again
            "1",  # threads
            "n",  # add another predictor
        ]
        + FILE_INPUTS
    )

    result = runner.invoke(init, input="\n".join(inputs))
    assert result.exit_code == 0, result.output
    assert "'predictor' is already used by another predictor." in result.output
    assert "'/v1/predict' is already used by another predictor." in result.output
    assert "module:Class" in result.output

    config = ConfigFile.load_from_file("cogito.yaml")
    assert config.cogito.get_predictor_names == ["predictor", "other"]
    other = config.cogito.get_predictor_config("other")
    assert other.class_path == "predict_other:Other"
    assert other.route.path == "/v1/other"


def test_init_writes_the_v2_schema(runner, temp_dir):
    result = runner.invoke(init, ["--default"])
    assert result.exit_code == 0

    with open("cogito.yaml") as file:
        data = yaml.safe_load(file)

    assert data["config_version"] == 2
    assert "predictor" not in data["cogito"]
    assert "route" not in data["cogito"]["server"]
    assert data["cogito"]["predictors"][0]["class"] == "predict:Predictor"
    assert "class_path" not in data["cogito"]["predictors"][0]


def test_init_default_scaffold(runner, temp_dir):
    result = runner.invoke(init, ["--default", "--scaffold"])
    assert result.exit_code == 0, result.output
    assert os.path.exists("predict.py")

    with open("predict.py") as file:
        content = file.read()
    compile(content, "predict.py", "exec")
    assert "class Predictor(BasePredictor):" in content
    assert "Route path: /v1/predict" in content


def test_init_scaffold_several_predictors(runner, temp_dir):
    inputs = (
        SERVER_INPUTS
        + [
            "sdxl",
            "predict_sdxl:Predictor",
            "/v1/sdxl",
            "1",
            "y",
            "clip",
            "predict_clip:Predictor",
            "/v1/clip",
            "1",
            "n",
        ]
        + FILE_INPUTS
    )

    result = runner.invoke(init, ["--scaffold"], input="\n".join(inputs))
    assert result.exit_code == 0, result.output

    for file_name, path in (
        ("predict_sdxl.py", "/v1/sdxl"),
        ("predict_clip.py", "/v1/clip"),
    ):
        with open(file_name) as file:
            content = file.read()
        compile(content, file_name, "exec")
        assert "class Predictor(BasePredictor):" in content
        assert f"Route path: {path}" in content

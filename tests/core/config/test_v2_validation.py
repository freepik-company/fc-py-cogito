import re

import pytest
from pydantic import ValidationError

from cogito.core.config.v2 import CogitoConfig, ResourcesConfig, ServerConfig


def make_config(*predictors) -> CogitoConfig:
    return CogitoConfig(
        server=ServerConfig.default().model_dump(), predictors=list(predictors)
    )


def predictor(name, path=None):
    data = {"name": name, "class": f"predict_{name}:Predictor"}
    if path is not None:
        data["route"] = {"path": path}
    return data


@pytest.mark.parametrize("gpus", [0, 0.25, 0.5, 1, 1.0, 2, 2.0, 8])
def test_valid_gpus(gpus):
    assert ResourcesConfig(gpus=gpus).gpus == gpus


@pytest.mark.parametrize("gpus", [-1, -0.5, 1.5, 2.25, float("inf")])
def test_gpus_must_be_a_fraction_or_a_positive_integer(gpus):
    message = (
        "gpus must be a fraction in (0, 1] or a positive integer (0 for no GPU), "
        f"got {float(gpus)}"
    )

    with pytest.raises(ValidationError, match=re.escape(message)):
        ResourcesConfig(gpus=gpus)


@pytest.mark.parametrize("field", ["gpus", "gpu_compute"])
@pytest.mark.parametrize("value", [True, False])
def test_resources_reject_booleans(field, value):
    with pytest.raises(
        ValidationError, match=re.escape(f"{field} must be a number, got {value}")
    ):
        ResourcesConfig(**{field: value})


@pytest.mark.parametrize("gpu_compute", [None, 1, 50, 100])
def test_valid_gpu_compute(gpu_compute):
    assert ResourcesConfig(gpu_compute=gpu_compute).gpu_compute == gpu_compute


@pytest.mark.parametrize("gpu_compute", [0, -10, 101, 1000])
def test_gpu_compute_must_be_a_percentage(gpu_compute):
    message = f"gpu_compute must be a percentage in 1..100, got {gpu_compute}"

    with pytest.raises(ValidationError, match=re.escape(message)):
        ResourcesConfig(gpu_compute=gpu_compute)


def test_gpu_compute_must_be_an_integer():
    with pytest.raises(ValidationError, match="gpu_compute"):
        ResourcesConfig(gpu_compute=50.5)


def test_invalid_resources_fail_the_whole_config():
    sdxl = {**predictor("sdxl"), "resources": {"gpus": 1.5}}

    with pytest.raises(ValidationError, match=r"predictors\.0\.resources\.gpus"):
        make_config(sdxl)


def test_predictor_names_must_be_unique():
    message = "predictor names must be unique, 'sdxl' is declared more than once"

    with pytest.raises(ValidationError, match=re.escape(message)):
        make_config(predictor("sdxl"), predictor("clip"), predictor("sdxl"))


def test_route_paths_must_be_unique():
    message = (
        "route paths must be unique, '/v1/predict' is declared by predictors "
        "'sdxl' and 'clip'"
    )

    with pytest.raises(ValidationError, match=re.escape(message)):
        make_config(predictor("sdxl", "/v1/predict"), predictor("clip", "/v1/predict"))


def test_unique_names_and_paths_are_valid():
    config = make_config(
        predictor("sdxl", "/v1/sdxl"), predictor("clip", "/v1/clip"), predictor("vae")
    )

    assert config.get_predictor_names == ["sdxl", "clip", "vae"]


def test_predictors_without_a_route_do_not_collide():
    config = make_config(predictor("sdxl"), predictor("clip"))

    assert config.get_predictor_names == ["sdxl", "clip"]

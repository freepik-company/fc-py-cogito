import pytest
import yaml
from pydantic import ValidationError

from cogito.core.config.v2 import (
    BatchingConfig,
    CogitoConfig,
    PredictorConfig,
    ResourcesConfig,
    RouteConfig,
    ServerConfig,
)

V2_CONFIG = """
server:
  name: Multi-model
  description: Two predictors on one GPU
  fastapi:
    host: 0.0.0.0
    port: 8000
  cache_dir: /tmp
predictors:
  - name: sdxl
    class: predict_sdxl:Predictor
    route: {path: /v1/sdxl}
    threads: 2
    resources: {gpus: 0.5, gpu_memory: 0.5, gpu_compute: 50}
    batching: {enabled: true, max_batch_size: 8, max_wait_ms: 10}
  - name: clip
    class: predict_clip:Predictor
    resources: {gpus: 0.25}
"""


def load_v2_config() -> CogitoConfig:
    return CogitoConfig(**yaml.safe_load(V2_CONFIG))


def test_parses_the_v2_schema():
    config = load_v2_config()

    assert isinstance(config.server, ServerConfig)
    assert config.server.route is None
    assert config.get_predictor_names == ["sdxl", "clip"]

    sdxl = config.get_predictor_config("sdxl")
    assert sdxl.get_class == "predict_sdxl:Predictor"
    assert sdxl.get_route_path == "/v1/sdxl"
    assert sdxl.get_threads == 2
    assert sdxl.get_resources == ResourcesConfig(
        gpus=0.5, gpu_memory=0.5, gpu_compute=50
    )
    assert sdxl.get_batching == BatchingConfig(
        enabled=True, max_batch_size=8, max_wait_ms=10
    )


def test_predictor_defaults():
    clip = load_v2_config().get_predictor_config("clip")

    assert clip.get_route is None
    assert clip.get_route_path is None
    assert clip.get_threads == 1
    assert clip.get_resources == ResourcesConfig(gpus=0.25)
    assert clip.get_resources.gpu_memory is None
    assert clip.get_resources.gpu_compute is None
    assert clip.get_batching is None


def test_resources_default_to_no_gpu_constraint():
    predictor = PredictorConfig(name="cpu", class_path="predict:Predictor")

    assert predictor.get_resources == ResourcesConfig.default()
    assert predictor.get_resources.gpus == 0


def test_predictor_class_accepts_alias_and_field_name():
    by_alias = PredictorConfig(**{"name": "a", "class": "m:P"})
    by_name = PredictorConfig(name="a", class_path="m:P")

    assert by_alias == by_name
    assert by_alias.model_dump(by_alias=True)["class"] == "m:P"


def test_default_route_is_a_v2_route():
    route = RouteConfig.default()

    assert type(route) is RouteConfig
    assert route.path == "/v1/predict"
    predictor = PredictorConfig(name="a", class_path="m:P", route=route)
    assert predictor.get_route_path == "/v1/predict"
    assert ServerConfig(**{**ServerConfig.default().model_dump(), "route": route})


def test_route_only_needs_a_path():
    route = RouteConfig(path="/v1/sdxl")

    assert route.name is None
    assert route.tags == []


@pytest.mark.parametrize(
    "predictor",
    [
        {"class": "m:P"},
        {"name": "a"},
        {"name": "a", "class": "m:P", "threads": "many"},
        {"name": "a", "class": "m:P", "threads": None},
        {"name": "a", "class": "m:P", "threads": 0},
        {"name": "a", "class": "m:P", "resources": {"gpus": "half"}},
        {"name": "a", "class": "m:P", "resources": {"gpu_compute": "most"}},
        {"name": "a", "class": "m:P", "batching": {"max_batch_size": "big"}},
    ],
)
def test_invalid_predictor_fails_validation(predictor):
    with pytest.raises(ValidationError):
        PredictorConfig(**predictor)


def test_predictors_are_required():
    data = yaml.safe_load(V2_CONFIG)
    del data["predictors"]

    with pytest.raises(ValidationError):
        CogitoConfig(**data)


def test_predictors_cannot_be_empty():
    data = yaml.safe_load(V2_CONFIG)
    data["predictors"] = []

    with pytest.raises(ValidationError):
        CogitoConfig(**data)


def test_v1_predictor_field_is_not_dumped():
    assert "predictor" not in load_v2_config().model_dump()


def test_unknown_predictor_name():
    with pytest.raises(KeyError, match="Unknown predictor 'missing'"):
        load_v2_config().get_predictor_config("missing")


def test_get_predictor_with_a_single_predictor():
    config = CogitoConfig.default()

    assert config.get_predictor == "predict:Predictor"


def test_get_predictor_with_several_predictors():
    with pytest.raises(ValueError, match="use get_predictors"):
        load_v2_config().get_predictor


def test_default_config():
    config = CogitoConfig.default()

    assert config.get_trainer == "train:Trainer"
    assert config.server.route is None
    assert config.get_server_name == "Cogito ergo sum"
    assert config.get_fastapi_port == 8000
    [predictor] = config.get_predictors
    assert predictor.name == "predictor"
    assert predictor.get_route_path == "/v1/predict"
    assert predictor.get_resources == ResourcesConfig()


def test_dump_round_trip():
    config = load_v2_config()

    dumped = yaml.safe_dump(config.model_dump(by_alias=True, exclude_none=True))

    assert CogitoConfig(**yaml.safe_load(dumped)) == config

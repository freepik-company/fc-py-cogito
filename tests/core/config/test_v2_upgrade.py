from pathlib import Path

import pytest
import yaml

from cogito.core.config.file import ConfigFile
from cogito.core.config.v0.base import CogitoConfig as CogitoConfigV0
from cogito.core.config.v1.base import CogitoConfig as CogitoConfigV1
from cogito.core.config.v2 import (
    CogitoConfig,
    FastAPIConfig,
    ResourcesConfig,
    RouteConfig,
    ServerConfig,
)

EXAMPLE_CONFIG = Path(__file__).parents[3] / "examples" / "cogito.yaml"

V0_CONFIG = """
server:
  name: Legacy
  description: A v0 server
  fastapi:
    host: 0.0.0.0
    port: 9000
  cache_dir: /tmp/legacy
  threads: 2
  route:
    name: Generate
    description: Generate an image
    path: /v1/generate
    predictor: generate:Predictor
    tags: [generate]
trainer: train:Trainer
"""


def load_v1_example() -> CogitoConfigV1:
    return ConfigFile.load_from_file(str(EXAMPLE_CONFIG)).cogito


def load_v0_config() -> CogitoConfigV0:
    return CogitoConfigV0(**yaml.safe_load(V0_CONFIG))


def upgrade(version: int, config) -> CogitoConfig:
    return CogitoConfig.default().upgrade(version, config)


def test_example_config_loads_unchanged_as_v1():
    config_file = ConfigFile.load_from_file(str(EXAMPLE_CONFIG))

    assert config_file.config_version == 1
    assert type(config_file.cogito) is CogitoConfigV1
    assert config_file.cogito.get_predictor == "predict:Predictor"
    assert config_file.cogito.get_route_path == "/v1/predict"
    assert config_file.cogito.get_server_threads == 4


def test_upgrade_v1_to_v2():
    v1_config = load_v1_example()

    config = upgrade(1, v1_config)

    assert type(config) is CogitoConfig
    assert type(config.server) is ServerConfig
    assert type(config.server.fastapi) is FastAPIConfig
    assert config.server.route is None
    assert config.get_trainer == "train:Trainer"
    assert config.get_server_name == "Cogito ergo sum"
    assert config.get_server_cache_dir == "/tmp/cogito"
    assert config.get_server_readiness_file == "/tmp/cogito-readiness.lock"
    assert config.server.return_input_on_response is False
    assert config.get_fastapi_host == "127.0.0.1"
    assert config.get_fastapi_access_log is True
    assert config.get_server_threads == 4

    [predictor] = config.get_predictors
    assert predictor.name == "predictor"
    assert predictor.get_class == "predict:Predictor"
    assert predictor.get_route == RouteConfig(
        name="Predict",
        description="Make a single prediction",
        path="/v1/predict",
        tags=["predict"],
    )
    assert predictor.get_threads == 4
    assert predictor.get_resources == ResourcesConfig()
    assert predictor.get_batching is None
    assert config.get_predictor == "predict:Predictor"


def test_upgrade_v0_to_v2():
    config = upgrade(0, load_v0_config())

    assert config.server.route is None
    assert config.get_server_name == "Legacy"
    assert config.get_fastapi_port == 9000
    assert config.get_trainer == "train:Trainer"

    [predictor] = config.get_predictors
    assert predictor.get_class == "generate:Predictor"
    assert predictor.get_route_path == "/v1/generate"
    assert predictor.get_route.name == "Generate"
    assert predictor.get_route.predictor is None
    assert predictor.get_threads == 2
    assert predictor.get_resources == ResourcesConfig()


@pytest.mark.parametrize(
    "version, load_config", [(0, load_v0_config), (1, load_v1_example)]
)
def test_upgrade_does_not_mutate_the_input(version, load_config):
    original = load_config()
    config = original.model_copy(deep=True)

    upgrade(version, config)

    assert config == original
    assert type(config.server) is type(original.server)


def test_upgrade_v1_route_without_tags():
    data = yaml.safe_load(EXAMPLE_CONFIG.read_text())["cogito"]
    del data["server"]["route"]["tags"]

    [predictor] = upgrade(1, CogitoConfigV1(**data)).get_predictors

    assert predictor.get_route.tags == []


def test_upgrade_v1_without_route():
    data = yaml.safe_load(EXAMPLE_CONFIG.read_text())["cogito"]
    data["server"]["route"] = None

    [predictor] = upgrade(1, CogitoConfigV1(**data)).get_predictors

    assert predictor.get_route is None


def test_upgrade_v1_without_threads():
    data = yaml.safe_load(EXAMPLE_CONFIG.read_text())["cogito"]
    data["server"]["threads"] = None

    [predictor] = upgrade(1, CogitoConfigV1(**data)).get_predictors

    assert predictor.get_threads == 1


def test_upgrade_v2_is_a_no_op():
    config = CogitoConfig.default()

    assert upgrade(2, config) is config


def test_upgraded_config_round_trips():
    config = upgrade(1, load_v1_example())

    dumped = yaml.safe_dump(config.model_dump(by_alias=True, exclude_none=True))

    assert "predictor" not in yaml.safe_load(dumped)
    assert CogitoConfig(**yaml.safe_load(dumped)) == config

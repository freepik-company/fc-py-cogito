import json
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from cogito.commands.predict import predict
from cogito.core.exceptions import ConfigFileNotFoundError


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def mock_context():
    ctx = MagicMock()
    ctx.get.return_value = "/path/to/config.yaml"
    return ctx


class TestPredictCommand:
    @patch("cogito.commands.predict.prediction")
    def test_predict_success(self, mock_prediction, cli_runner, mock_context):
        # Arrange
        payload = {"key": "value"}
        mock_result = MagicMock()
        mock_result.model_dump_json.return_value = '{"prediction": "result"}'
        mock_prediction.return_value = mock_result

        # Act
        result = cli_runner.invoke(
            predict, ["--payload", json.dumps(payload)], obj=mock_context
        )

        # Assert
        assert result.exit_code == 0
        assert "prediction" in result.output
        assert "result" in result.output
        mock_prediction.assert_called_once_with("/path/to/config.yaml", payload)

    def test_predict_invalid_json(self, cli_runner, mock_context):
        # Act
        result = cli_runner.invoke(
            predict, ["--payload", "invalid_json"], obj=mock_context
        )

        # Assert
        assert result.exit_code == 1
        assert "Error:" in result.output

    @patch("cogito.commands.predict.prediction")
    def test_predict_config_not_found(self, mock_prediction, cli_runner, mock_context):
        # Arrange
        payload = {"key": "value"}
        mock_prediction.side_effect = ConfigFileNotFoundError("Config not found")

        # Act
        result = cli_runner.invoke(
            predict, ["--payload", json.dumps(payload)], obj=mock_context
        )

        # Assert
        assert result.exit_code == 1
        assert "No configuration file found" in result.output

    @patch("cogito.commands.predict.prediction")
    def test_predict_generic_error(self, mock_prediction, cli_runner, mock_context):
        # Arrange
        payload = {"key": "value"}
        mock_prediction.side_effect = ValueError("Some error occurred")

        # Act
        result = cli_runner.invoke(
            predict, ["--payload", json.dumps(payload)], obj=mock_context
        )

        # Assert
        assert result.exit_code == 1
        assert "Error: Some error occurred" in result.output

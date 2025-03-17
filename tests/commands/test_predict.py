import json
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from cogito.commands.predict import predict
from cogito.core.exceptions import ConfigFileNotFoundError


class TestPredictCommand:
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()

    @pytest.fixture
    def mock_context(self):
        ctx = MagicMock()
        ctx.get.return_value = "/path/to/cogito.yaml"
        return ctx

    @patch("cogito.commands.predict.run")
    def test_predict_success(self, mock_run, cli_runner, mock_context):
        # Arrange
        mock_run.return_value = MagicMock(
            model_dump_json=lambda indent: '{"prediction": "test", "probability": 0.95}'
        )
        payload = json.dumps({"text": "sample text for prediction"})

        # Act
        result = cli_runner.invoke(predict, ["--payload", payload], obj=mock_context)

        # Assert
        assert result.exit_code == 0
        mock_run.assert_called_once_with(
            "/path/to/cogito.yaml",
            {"text": "sample text for prediction"},
            run_setup=True,
        )
        assert "prediction" in result.output
        assert "test" in result.output
        assert "0.95" in result.output

    @patch("cogito.commands.predict.run")
    def test_predict_config_not_found(self, mock_run, cli_runner, mock_context):
        # Arrange
        mock_run.side_effect = ConfigFileNotFoundError(
            file_path="/path/to/nonexistent/cogito.yaml"
        )
        payload = json.dumps({"text": "sample text for prediction"})

        # Act
        result = cli_runner.invoke(predict, ["--payload", payload], obj=mock_context)

        # Assert
        assert result.exit_code == 1
        assert "No configuration file found" in result.output

    @patch("cogito.commands.predict.run")
    def test_predict_generic_error(self, mock_run, cli_runner, mock_context):
        # Arrange
        mock_run.side_effect = Exception("Test error")
        payload = json.dumps({"text": "sample text for prediction"})

        # Act
        result = cli_runner.invoke(predict, ["--payload", payload], obj=mock_context)

        # Assert
        assert result.exit_code == 1
        assert "Error: Test error" in result.output

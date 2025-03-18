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

    @patch("cogito.commands.predict.Predict")
    def test_predict_success(self, mock_predict_class, cli_runner, mock_context):
        # Arrange
        mock_predict_instance = MagicMock()
        mock_predict_class.return_value = mock_predict_instance

        mock_result = MagicMock()
        mock_result.model_dump_json = (
            lambda indent: '{"prediction": "test", "probability": 0.95}'
        )
        mock_predict_instance.run.return_value = mock_result

        payload = json.dumps({"text": "sample text for prediction"})

        # Act
        result = cli_runner.invoke(predict, ["--payload", payload], obj=mock_context)

        # Assert
        assert result.exit_code == 0
        mock_predict_class.assert_called_once_with("/path/to/cogito.yaml")
        mock_predict_instance.setup.assert_called_once()
        mock_predict_instance.run.assert_called_once_with(
            {"text": "sample text for prediction"}
        )
        assert "prediction" in result.output
        assert "test" in result.output
        assert "0.95" in result.output

    @patch("cogito.commands.predict.Predict")
    def test_predict_config_not_found(
        self, mock_predict_class, cli_runner, mock_context
    ):
        # Arrange
        mock_predict_class.side_effect = ConfigFileNotFoundError(
            file_path="/path/to/nonexistent/cogito.yaml"
        )
        payload = json.dumps({"text": "sample text for prediction"})

        # Act
        result = cli_runner.invoke(predict, ["--payload", payload], obj=mock_context)

        # Assert
        assert result.exit_code == 1
        assert "Config file not found" in result.output

    @patch("cogito.commands.predict.Predict")
    def test_predict_generic_error(self, mock_predict_class, cli_runner, mock_context):
        # Arrange
        mock_predict_class.side_effect = Exception("Test error")
        payload = json.dumps({"text": "sample text for prediction"})

        # Act
        result = cli_runner.invoke(predict, ["--payload", payload], obj=mock_context)

        # Assert
        assert result.exit_code == 1
        assert "Error initializing the predictor: Test error" in result.output

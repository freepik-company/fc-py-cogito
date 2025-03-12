import json
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from cogito.commands.train import train
from cogito.core.exceptions import ConfigFileNotFoundError


class TestTrainCommand:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def valid_payload(self):
        return json.dumps(
            {"model": "test_model", "data": {"x": [1, 2, 3], "y": [4, 5, 6]}}
        )

    @patch("cogito.commands.train.setup")
    @patch("cogito.commands.train.run")
    def test_train_success(self, mock_run, mock_setup, runner, valid_payload):
        # Arrange
        mock_ctx = MagicMock()
        mock_ctx.get.return_value = "/path/to/cogito.yaml"

        mock_run.return_value = {"accuracy": 0.95, "loss": 0.05}

        # Act
        result = runner.invoke(train, ["--payload", valid_payload], obj=mock_ctx)

        # Assert
        assert result.exit_code == 0
        mock_setup.assert_called_once_with("/path/to/cogito.yaml")
        mock_run.assert_called_once()
        assert "accuracy" in result.output
        assert "0.95" in result.output

    @patch("cogito.commands.train.setup")
    @patch("cogito.commands.train.run")
    def test_train_config_not_found(self, mock_run, mock_setup, runner, valid_payload):
        # Arrange
        mock_ctx = MagicMock()
        mock_ctx.get.return_value = "/path/to/nonexistent/cogito.yaml"

        mock_setup.side_effect = ConfigFileNotFoundError(
            file_path="/path/to/nonexistent/cogito.yaml"
        )

        # Act
        result = runner.invoke(train, ["--payload", valid_payload], obj=mock_ctx)

        # Assert
        assert result.exit_code == 1
        assert "No configuration file found" in result.output

    @patch("cogito.commands.train.setup")
    @patch("cogito.commands.train.run")
    def test_train_general_exception(self, mock_run, mock_setup, runner, valid_payload):
        # Arrange
        mock_ctx = MagicMock()
        mock_ctx.get.return_value = "/path/to/cogito.yaml"

        mock_setup.side_effect = Exception("Something went wrong during training")

        # Act
        result = runner.invoke(train, ["--payload", valid_payload], obj=mock_ctx)

        # Assert
        assert result.exit_code == 1
        assert "Error: Something went wrong during training" in result.output

    def test_train_invalid_json(self, runner):
        # Arrange
        mock_ctx = MagicMock()
        mock_ctx.get.return_value = "/path/to/cogito.yaml"

        # Act
        result = runner.invoke(train, ["--payload", "invalid-json"], obj=mock_ctx)

        # Assert
        assert result.exit_code == 1
        assert "Error: " in result.output

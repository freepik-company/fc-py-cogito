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
    def mock_context(self):
        ctx = MagicMock()
        ctx.get.return_value = "/path/to/cogito.yaml"
        return ctx

    @pytest.fixture
    def valid_payload(self):
        return json.dumps(
            {"model": "test_model", "data": {"x": [1, 2, 3], "y": [4, 5, 6]}}
        )

    @patch("cogito.commands.train.Trainer")
    def test_train_success(
        self, mock_trainer_class, runner, mock_context, valid_payload
    ):
        # Arrange
        mock_trainer_instance = MagicMock()
        mock_trainer_class.return_value = mock_trainer_instance
        mock_trainer_instance.run.return_value = "Training completed successfully"

        # Act
        result = runner.invoke(train, ["--payload", valid_payload], obj=mock_context)

        # Assert
        assert result.exit_code == 0
        mock_trainer_class.assert_called_once_with("/path/to/cogito.yaml")
        mock_trainer_instance.setup.assert_called_once()
        mock_trainer_instance.run.assert_called_once_with(
            {"model": "test_model", "data": {"x": [1, 2, 3], "y": [4, 5, 6]}},
            run_setup=True,
        )
        assert "Training completed successfully" in result.output

    @patch("cogito.commands.train.Trainer")
    def test_train_config_not_found(
        self, mock_trainer_class, runner, mock_context, valid_payload
    ):
        # Arrange
        mock_trainer_class.side_effect = ConfigFileNotFoundError(
            file_path="/path/to/nonexistent/cogito.yaml"
        )

        # Act
        result = runner.invoke(train, ["--payload", valid_payload], obj=mock_context)

        # Assert
        assert result.exit_code == 1
        assert "Config file not found" in result.output

    @patch("cogito.commands.train.Trainer")
    def test_train_general_exception(
        self, mock_trainer_class, runner, mock_context, valid_payload
    ):
        # Arrange
        mock_trainer_class.side_effect = Exception("Test error")

        # Act
        result = runner.invoke(train, ["--payload", valid_payload], obj=mock_context)

        # Assert
        assert result.exit_code == 1
        assert "Error: Test error" in result.output

    def test_train_invalid_json(self, runner):
        # Arrange
        mock_ctx = MagicMock()
        mock_ctx.get.return_value = "/path/to/cogito.yaml"

        # Act
        result = runner.invoke(train, ["--payload", "invalid-json"], obj=mock_ctx)

        # Assert
        assert result.exit_code == 1
        assert "Error: " in result.output

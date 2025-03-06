import json
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from cogito.commands.train import train
from cogito.core.exceptions import ConfigFileNotFoundError


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def valid_payload():
    return json.dumps({"model": "test_model", "data": {"x": [1, 2, 3], "y": [4, 5, 6]}})


class TestTrainCommand:
    @patch("cogito.commands.train.training")
    def test_train_success(self, mock_training, runner, valid_payload):
        """Test train command with valid payload"""
        # Setup
        mock_training.return_value = {"status": "success", "model_id": "123"}
        ctx = {"config_path": "/path/to/config.yaml"}

        # Execute
        result = runner.invoke(train, ["--payload", valid_payload], obj=ctx)

        # Verify
        assert result.exit_code == 0
        mock_training.assert_called_once_with(
            "/path/to/config.yaml", json.loads(valid_payload)
        )
        assert "{'status': 'success', 'model_id': '123'}" in result.output

    @patch("cogito.commands.train.training")
    def test_train_config_not_found(self, mock_training, runner, valid_payload):
        """Test train command when config file is not found"""
        # Setup
        mock_training.side_effect = ConfigFileNotFoundError("Config file not found")
        ctx = {"config_path": "/path/to/config.yaml"}

        # Execute
        result = runner.invoke(train, ["--payload", valid_payload], obj=ctx)

        # Verify
        assert result.exit_code == 1
        assert "No configuration file found" in result.output

    @patch("cogito.commands.train.training")
    def test_train_general_exception(self, mock_training, runner, valid_payload):
        """Test train command when an unexpected error occurs"""
        # Setup
        mock_training.side_effect = Exception("Some unexpected error")
        ctx = {"config_path": "/path/to/config.yaml"}

        # Execute
        result = runner.invoke(train, ["--payload", valid_payload], obj=ctx)

        # Verify
        assert result.exit_code == 1
        assert "Error: Some unexpected error" in result.output

    def test_train_invalid_json(self, runner):
        """Test train command with invalid JSON payload"""
        # Setup
        invalid_payload = "not valid json"
        ctx = {"config_path": "/path/to/config.yaml"}

        # Execute
        result = runner.invoke(train, ["--payload", invalid_payload], obj=ctx)

        # Verify
        assert result.exit_code == 1
        assert "Error:" in result.output

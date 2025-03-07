import os
import sys
import unittest
from unittest.mock import patch, MagicMock, AsyncMock, call
import asyncio

# Add the parent directory to sys.path to allow importing cogito modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from cogito.lib.training import training
from cogito.core.exceptions import ConfigFileNotFoundError


class MockTrainer:
    async def setup(self):
        pass

    def train(self, **kwargs):
        return {"model_accuracy": 0.95, "training_complete": True}


class TestTraining(unittest.TestCase):

    @patch("cogito.lib.training.ConfigFile.load_from_file")
    @patch("cogito.lib.training.instance_class")
    @patch("cogito.lib.training.asyncio.run")
    def test_training_success(
        self, mock_asyncio_run, mock_instance_class, mock_load_config
    ):
        # Setup
        mock_config = MagicMock()
        mock_config.cogito.get_trainer = "path.to.MockTrainer"
        mock_load_config.return_value = mock_config

        mock_trainer = MockTrainer()
        mock_instance_class.return_value = mock_trainer

        # Spy on the train method to verify it's called with the correct arguments
        original_train = mock_trainer.train
        mock_trainer.train = MagicMock(wraps=original_train)

        # Test data
        config_path = "/path/to/cogito.yaml"
        payload_data = {"data_path": "/path/to/data", "epochs": 10}

        # Execute
        result = training(config_path, payload_data)

        # Verify
        # Check that the config was loaded with the correct path
        mock_load_config.assert_called_once_with(config_path)

        # Check that instance_class was called with the correct trainer path
        mock_instance_class.assert_called_once_with("path.to.MockTrainer")

        # Check that setup was run asynchronously
        mock_asyncio_run.assert_called_once()
        self.assertEqual(mock_asyncio_run.call_args[0][0].__name__, "setup")

        # Check that train was called with the correct payload
        mock_trainer.train.assert_called_once_with(**payload_data)

        # Check the result
        expected_result = {"model_accuracy": 0.95, "training_complete": True}
        self.assertEqual(result, expected_result)

    @patch("cogito.lib.training.ConfigFile.load_from_file")
    def test_training_config_not_found(self, mock_load_config):
        # Setup - simulate config file not found
        # Pass the required file_path argument to the ConfigFileNotFoundError constructor
        mock_load_config.side_effect = ConfigFileNotFoundError(
            file_path="/path/to/nonexistent/cogito.yaml"
        )

        # Execute and verify exception is raised
        with self.assertRaises(ConfigFileNotFoundError):
            training("/path/to/nonexistent/cogito.yaml", {})

    @patch("cogito.lib.training.ConfigFile.load_from_file")
    @patch("cogito.lib.training.instance_class")
    @patch("cogito.lib.training.asyncio.run")
    @patch("cogito.lib.training.sys.path")
    def test_training_with_app_dir_in_path(
        self, mock_sys_path, mock_asyncio_run, mock_instance_class, mock_load_config
    ):
        # Setup
        mock_config = MagicMock()
        mock_config.cogito.get_trainer = "path.to.MockTrainer"
        mock_load_config.return_value = mock_config

        mock_trainer = MockTrainer()
        mock_instance_class.return_value = mock_trainer

        # Test data
        config_path = "/path/to/cogito.yaml"
        payload_data = {"data_path": "/path/to/data"}

        # Execute
        training(config_path, payload_data)

        # Verify sys.path was modified correctly
        # Instead of patching sys.path.insert, we patch sys.path and check that insert was called
        mock_sys_path.insert.assert_called_once_with(0, "/path/to")


if __name__ == "__main__":
    unittest.main()

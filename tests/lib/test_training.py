import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path to allow importing cogito modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from cogito.lib.training import run
from cogito.core.exceptions import ConfigFileNotFoundError


class MockTrainer:
    def __init__(self):
        # Use MagicMock instead of an async function to avoid creating coroutines
        self.setup = MagicMock()

    def train(self, **kwargs):
        return {"model_accuracy": 0.95, "training_complete": True}


class TestTraining(unittest.TestCase):

    @patch("cogito.lib.training.build_config_file")
    @patch("cogito.lib.training.instance_class")
    def test_run_success(self, mock_instance_class, mock_build_config_file):
        # Setup
        mock_config = MagicMock()
        mock_config.cogito.get_trainer = "path.to.MockTrainer"
        mock_build_config_file.return_value = mock_config

        mock_trainer = MockTrainer()
        mock_instance_class.return_value = mock_trainer

        # Spy on the train method to verify it's called with the correct arguments
        original_train = mock_trainer.train
        mock_trainer.train = MagicMock(wraps=original_train)

        # Test data
        config_path = "/path/to/cogito.yaml"
        payload_data = {"data_path": "/path/to/data", "epochs": 10}

        # Execute
        result = run(config_path, payload_data)

        # Verify
        # Check that the config was loaded with the correct path
        mock_build_config_file.assert_called_once_with(config_path)

        # Check that instance_class was called with the correct trainer path
        mock_instance_class.assert_called_once_with("path.to.MockTrainer")

        # Check that train was called with the correct payload
        mock_trainer.train.assert_called_once_with(**payload_data)

        # Check the result
        expected_result = {"model_accuracy": 0.95, "training_complete": True}
        self.assertEqual(result, expected_result)

    @patch("cogito.lib.training.build_config_file")
    def test_run_config_not_found(self, mock_build_config_file):
        # Setup - simulate config file not found
        # Pass the required file_path argument to the ConfigFileNotFoundError constructor
        mock_build_config_file.side_effect = ConfigFileNotFoundError(
            file_path="/path/to/nonexistent/cogito.yaml"
        )

        # Execute and verify exception is raised
        with self.assertRaises(ConfigFileNotFoundError):
            run("/path/to/nonexistent/cogito.yaml", {})

    @patch("cogito.lib.training.build_config_file")
    @patch("cogito.lib.training.instance_class")
    def test_run_general_exception(self, mock_instance_class, mock_build_config_file):
        # Setup
        mock_config = MagicMock()
        mock_config.cogito.get_trainer = "path.to.MockTrainer"
        mock_build_config_file.return_value = mock_config

        # Create and setup the exception
        error_message = "Training error"
        mock_instance_class.side_effect = Exception(error_message)

        # Execute and verify exception is raised
        with self.assertRaises(Exception) as context:
            run("/path/to/cogito.yaml", {})

        self.assertIn(error_message, str(context.exception))

    @patch("cogito.lib.training.build_config_file")
    @patch("cogito.lib.training.instance_class")
    @patch("sys.path")
    def test_run_with_app_dir_in_path(
        self, mock_sys_path, mock_instance_class, mock_build_config_file
    ):
        # Setup
        mock_config = MagicMock()
        mock_config.cogito.get_trainer = "path.to.MockTrainer"
        mock_build_config_file.return_value = mock_config

        mock_trainer = MockTrainer()
        mock_instance_class.return_value = mock_trainer

        # Test data
        config_path = "/path/to/cogito.yaml"
        payload_data = {"data_path": "/path/to/data"}

        # Execute
        run(config_path, payload_data)

        # Verify sys.path was modified correctly
        # Instead of patching sys.path.insert, we patch sys.path and check that insert was called
        mock_sys_path.insert.assert_not_called()  # Should not be called in our implementation


if __name__ == "__main__":
    unittest.main()

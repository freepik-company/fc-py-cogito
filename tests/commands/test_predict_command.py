import json
import unittest
from unittest import mock
from click.testing import CliRunner

from cogito.api.responses import ResultResponse
from cogito.commands.predict import predict
from cogito.core.config import ConfigFile
from cogito.core.exceptions import ConfigFileNotFoundError


class MockPredictor:
    def predict(self, prompt: str, temperature: float = 0.5):
        return f"Generated text for '{prompt}' with temperature {temperature}"


class MockInputModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self):
        return {key: value for key, value in self.__dict__.items()}

    def dict(self):
        return self.model_dump()


class MockResponseModel(ResultResponse):
    def model_dump_json(self, indent=None):
        return json.dumps(
            {
                "inference_time_seconds": self.inference_time_seconds,
                "input": self.input,
                "result": self.result,
            },
            indent=indent,
        )


class TestPredictCommand(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

        # Create a nested mock structure instead of using spec
        self.mock_config = mock.MagicMock()
        self.mock_config.cogito = mock.MagicMock()
        self.mock_config.cogito.server = mock.MagicMock()
        self.mock_config.cogito.server.route = mock.MagicMock()
        self.mock_config.cogito.server.route.predictor = "test.module:MockPredictor"

        # Setup predictor instance
        self.mock_predictor = MockPredictor()

        # Create a mock input model class
        self.mock_input_model_class = mock.MagicMock()
        self.mock_input_model_class.return_value = MockInputModel(
            prompt="A cow", temperature=0.2
        )

        # Create a mock response model class
        self.mock_response_model = MockResponseModel

        # Setup patch for response instance
        self.mock_response = MockResponseModel(
            inference_time_seconds=0.1,
            input={"prompt": "A cow", "temperature": 0.2},
            result="Generated text for 'A cow' with temperature 0.2",
        )

    @mock.patch("cogito.commands.predict.ConfigFile.load_from_file")
    @mock.patch("cogito.commands.predict.load_predictor")
    @mock.patch("cogito.commands.predict.create_request_model")
    @mock.patch("cogito.commands.predict.get_predictor_handler_return_type")
    @mock.patch("cogito.commands.predict.wrap_handler")
    def test_predict_command_success(
        self,
        mock_wrap_handler,
        mock_get_return_type,
        mock_create_request_model,
        mock_load_predictor,
        mock_load_config,
    ):
        # Configure mocks
        mock_load_config.return_value = self.mock_config
        mock_load_predictor.return_value = self.mock_predictor
        mock_create_request_model.return_value = (
            "MockPredictor",
            self.mock_input_model_class,
        )
        mock_get_return_type.return_value = self.mock_response_model
        mock_wrap_handler.return_value = mock.MagicMock(return_value=self.mock_response)

        # Run command
        result = self.runner.invoke(
            predict,
            ["--payload", '{"prompt": "A cow", "temperature": 0.2}'],
            obj={"config_path": "/fake/path/cogito.yaml"},
        )

        # Verify results
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Generated text for 'A cow' with temperature 0.2", result.output)

        # Verify function calls
        mock_load_config.assert_called_once_with("/fake/path/cogito.yaml")
        mock_load_predictor.assert_called_once_with("test.module:MockPredictor")
        mock_create_request_model.assert_called_once()
        mock_get_return_type.assert_called_once_with(self.mock_predictor)
        mock_wrap_handler.assert_called_once()

    @mock.patch("cogito.commands.predict.ConfigFile.load_from_file")
    def test_predict_command_config_not_found(self, mock_load_config):
        # Make load_config raise ConfigFileNotFoundError
        mock_load_config.side_effect = ConfigFileNotFoundError("/fake/path/cogito.yaml")

        # Run command
        result = self.runner.invoke(
            predict,
            ["--payload", '{"prompt": "A cow"}'],
            obj={"config_path": "/fake/path/cogito.yaml"},
        )

        # Verify results
        self.assertEqual(result.exit_code, 1)
        self.assertIn("No configuration file found", result.output)

    @mock.patch("cogito.commands.predict.ConfigFile.load_from_file")
    @mock.patch("cogito.commands.predict.load_predictor")
    def test_predict_command_invalid_json(self, mock_load_predictor, mock_load_config):
        # Configure mocks
        mock_load_config.return_value = self.mock_config

        # Run command with invalid JSON
        result = self.runner.invoke(
            predict,
            ["--payload", "{prompt: invalid}"],
            obj={"config_path": "/fake/path/cogito.yaml"},
        )

        # Verify results
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Error", result.output)

    @mock.patch("cogito.commands.predict.ConfigFile.load_from_file")
    @mock.patch("cogito.commands.predict.load_predictor")
    @mock.patch("cogito.commands.predict.create_request_model")
    @mock.patch("cogito.commands.predict.get_predictor_handler_return_type")
    @mock.patch("cogito.commands.predict.wrap_handler")
    def test_predict_command_handler_error(
        self,
        mock_wrap_handler,
        mock_get_return_type,
        mock_create_request_model,
        mock_load_predictor,
        mock_load_config,
    ):
        # Configure mocks
        mock_load_config.return_value = self.mock_config
        mock_load_predictor.return_value = self.mock_predictor
        mock_create_request_model.return_value = (
            "MockPredictor",
            self.mock_input_model_class,
        )
        mock_get_return_type.return_value = self.mock_response_model

        # Make wrap_handler raise an exception
        mock_wrap_handler.side_effect = Exception("Handler error")

        # Run command
        result = self.runner.invoke(
            predict,
            ["--payload", '{"prompt": "A cow", "temperature": 0.2}'],
            obj={"config_path": "/fake/path/cogito.yaml"},
        )

        # Verify results
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Error: Handler error", result.output)


if __name__ == "__main__":
    unittest.main()

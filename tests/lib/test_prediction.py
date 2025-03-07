import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

from cogito.lib.prediction import prediction
from cogito.core.exceptions import ConfigFileNotFoundError


class MockPredictor:
    def __init__(self):
        # Create a synchronous function instead of AsyncMock to avoid creating coroutines that need to be awaited
        self.setup = MagicMock()

    def predict(self, input_data):
        return {"prediction": "test_prediction", "probability": 0.95}


@pytest.fixture
def mock_config_file():
    config_mock = MagicMock()
    config_mock.cogito.get_predictor = "test_predictor"
    return config_mock


@pytest.fixture
def test_payload():
    return {"text": "sample text for prediction"}

# Patch asyncio.run globally for all tests in this module
# This prevents warnings about unawaited coroutines
@pytest.fixture(autouse=True)
def patch_asyncio():
    with patch("cogito.lib.prediction.asyncio.run", return_value=None) as mock_run:
        yield mock_run


@patch("cogito.lib.prediction.ConfigFile")
@patch("cogito.lib.prediction.instance_class")
@patch("cogito.lib.prediction.create_request_model")
@patch("cogito.lib.prediction.get_predictor_handler_return_type")
@patch("cogito.lib.prediction.wrap_handler")
def test_prediction_successful(
    mock_wrap_handler,
    mock_get_return_type,
    mock_create_request,
    mock_instance_class,
    mock_config_file_class,
    mock_config_file,
    test_payload,
    patch_asyncio,
):
    # Setup mocks
    mock_config_file_class.load_from_file.return_value = mock_config_file

    predictor_instance = MockPredictor()
    mock_instance_class.return_value = predictor_instance

    input_model_class = MagicMock()
    input_model = MagicMock()
    mock_create_request.return_value = (None, input_model_class)
    input_model_class.return_value = input_model

    response_model = MagicMock()
    mock_get_return_type.return_value = response_model

    handler_mock = MagicMock()
    expected_response = {"prediction": "test_prediction", "probability": 0.95}
    handler_mock.return_value = expected_response
    mock_wrap_handler.return_value = handler_mock

    # Call the function
    config_path = "/path/to/cogito.yaml"
    result = prediction(config_path, test_payload)

    # Assertions
    mock_config_file_class.load_from_file.assert_called_once_with(config_path)
    mock_instance_class.assert_called_once_with("test_predictor")
    patch_asyncio.assert_called_once()
    mock_create_request.assert_called_once()
    mock_get_return_type.assert_called_once_with(predictor_instance)
    mock_wrap_handler.assert_called_once()
    handler_mock.assert_called_once_with(input_model)

    # Verify the returned result matches the expected response
    assert result == expected_response


@patch("cogito.lib.prediction.ConfigFile")
def test_prediction_config_not_found(mock_config_file_class, test_payload):
    # Setup mock to raise the exception
    mock_config_file_class.load_from_file.side_effect = ConfigFileNotFoundError(
        file_path="/path/to/nonexistent/cogito.yaml"
    )

    # Call the function and check for raised exception
    with pytest.raises(ConfigFileNotFoundError):
        prediction("/path/to/nonexistent/cogito.yaml", test_payload)


@patch("cogito.lib.prediction.ConfigFile")
@patch("cogito.lib.prediction.instance_class")
def test_prediction_general_exception(
    mock_instance_class, mock_config_file_class, mock_config_file, test_payload, patch_asyncio
):
    # Setup mocks
    mock_config_file_class.load_from_file.return_value = mock_config_file
    
    # Create a regular exception instead of a mock that creates a coroutine
    class TestException(Exception):
        pass
    
    # Raise the exception immediately when instance_class is called, 
    # before any AsyncMock methods would be accessed
    mock_instance_class.side_effect = TestException("Test exception")
    
    # Call the function and check for raised exception - exception should be raised
    # before asyncio.run is ever called
    with pytest.raises(TestException) as excinfo:
        prediction("/path/to/cogito.yaml", test_payload)
    
    assert "Test exception" in str(excinfo.value)
    patch_asyncio.assert_not_called()  # Should never reach this point

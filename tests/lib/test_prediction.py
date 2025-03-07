import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from cogito.lib.prediction import prediction
from cogito.core.exceptions import ConfigFileNotFoundError


class MockPredictor:
<<<<<<< HEAD
    async def setup(self):
=======
    def setup(self):
>>>>>>> main
        # Mock implementation for setup
        pass

    def predict(self, input_data):
        return {"prediction": "test_prediction", "probability": 0.95}


@pytest.fixture
def mock_config_file():
    config_mock = MagicMock()
<<<<<<< HEAD
    config_mock.cogito.get_predictor = "test_predictor"
=======
    config_mock.cogito.server.route.predictor = "test_predictor"
>>>>>>> main
    return config_mock


@pytest.fixture
def test_payload():
    return {"text": "sample text for prediction"}


@patch("cogito.lib.prediction.ConfigFile")
@patch("cogito.lib.prediction.instance_class")
@patch("cogito.lib.prediction.create_request_model")
@patch("cogito.lib.prediction.get_predictor_handler_return_type")
@patch("cogito.lib.prediction.wrap_handler")
@patch("cogito.lib.prediction.asyncio.run")
def test_prediction_successful(
    mock_asyncio_run,
    mock_wrap_handler,
    mock_get_return_type,
    mock_create_request,
    mock_instance_class,
    mock_config_file_class,
    mock_config_file,
    test_payload,
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
    mock_asyncio_run.assert_called_once()
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
    mock_instance_class, mock_config_file_class, mock_config_file, test_payload
):
    # Setup mocks
    mock_config_file_class.load_from_file.return_value = mock_config_file

    # Set up the predictor mock to raise an exception without creating a coroutine
    mock_instance_class.side_effect = Exception("Test exception")

    # Patch asyncio.run separately to ensure it's not used in this test path
    with patch("cogito.lib.prediction.asyncio.run"):
        # Call the function and check for raised exception
        with pytest.raises(Exception) as excinfo:
            prediction("/path/to/cogito.yaml", test_payload)

    assert "Test exception" in str(excinfo.value)

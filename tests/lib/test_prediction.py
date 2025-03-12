import pytest
from unittest.mock import patch, MagicMock

from cogito.lib.prediction import run, setup
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


@patch("cogito.lib.prediction._config_file_path")
@patch("cogito.lib.prediction._get_instance_class")
@patch("cogito.lib.prediction.create_request_model")
@patch("cogito.lib.prediction.get_predictor_handler_return_type")
@patch("cogito.lib.prediction.wrap_handler")
def test_run_successful(
    mock_wrap_handler,
    mock_get_return_type,
    mock_create_request,
    mock_instance_class,
    mock_config_file_class,
    mock_config_file,
    test_payload,
):
    # Setup mocks
    mock_config_file_class.return_value = mock_config_file

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
    result = run(config_path, test_payload)

    # Assertions
    mock_config_file_class.assert_called_once_with(config_path)
    mock_instance_class.assert_called_once_with(mock_config_file.cogito.get_predictor)
    mock_create_request.assert_called_once()
    mock_get_return_type.assert_called_once_with(predictor_instance)
    mock_wrap_handler.assert_called_once()
    handler_mock.assert_called_once_with(input_model)

    # Verify the returned result matches the expected response
    assert result == expected_response


@patch("cogito.lib.prediction._config_file_path")
def test_run_config_not_found(mock_config_file_class, test_payload):
    # Setup mock to raise the exception
    mock_config_file_class.side_effect = ConfigFileNotFoundError(
        file_path="/path/to/nonexistent/cogito.yaml"
    )

    # Call the function and check for raised exception
    with pytest.raises(ConfigFileNotFoundError):
        run("/path/to/nonexistent/cogito.yaml", test_payload)


@patch("cogito.lib.prediction._config_file_path")
@patch("cogito.lib.prediction._get_instance_class")
def test_run_general_exception(
    mock_instance_class,
    mock_config_file_class,
    mock_config_file,
    test_payload,
):
    # Setup mocks
    mock_config_file_class.return_value = mock_config_file

    # Create a regular exception instead of a mock that creates a coroutine
    class TestException(Exception):
        pass

    # Raise the exception immediately when instance_class is called
    mock_instance_class.side_effect = TestException("Test exception")

    # Call the function and check for raised exception
    with pytest.raises(TestException) as excinfo:
        run("/path/to/cogito.yaml", test_payload)

    assert "Test exception" in str(excinfo.value)


@patch("cogito.lib.prediction._config_file_path")
@patch("cogito.lib.prediction._get_instance_class")
def test_setup_successful(
    mock_instance_class, mock_config_file_class, mock_config_file
):
    # Setup mocks
    mock_config_file_class.return_value = mock_config_file

    predictor_instance = MockPredictor()
    mock_instance_class.return_value = predictor_instance

    # Call setup function
    config_path = "/path/to/cogito.yaml"
    setup(config_path)

    # Assertions
    mock_config_file_class.assert_called_once_with(config_path)
    mock_instance_class.assert_called_once_with(mock_config_file.cogito.get_predictor)
    predictor_instance.setup.assert_called_once()


@patch("cogito.lib.prediction._config_file_path")
def test_setup_config_not_found(mock_config_file_class):
    # Setup mock to raise the exception
    mock_config_file_class.side_effect = ConfigFileNotFoundError(
        file_path="/path/to/nonexistent/cogito.yaml"
    )

    # Call the function and check for raised exception
    with pytest.raises(ConfigFileNotFoundError):
        setup("/path/to/nonexistent/cogito.yaml")


@patch("cogito.lib.prediction._config_file_path")
@patch("cogito.lib.prediction._get_instance_class")
def test_setup_general_exception(
    mock_instance_class, mock_config_file_class, mock_config_file
):
    # Setup mocks
    mock_config_file_class.return_value = mock_config_file

    # Create and setup the exception
    mock_predictor = MockPredictor()
    mock_instance_class.return_value = mock_predictor

    error_message = "Error in setup"
    mock_predictor.setup.side_effect = Exception(error_message)

    # Call the function and check for raised exception
    with pytest.raises(Exception) as excinfo:
        setup("/path/to/cogito.yaml")

    assert error_message in str(excinfo.value)

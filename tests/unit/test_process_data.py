"""Module containing unit tests for the process_data module."""

from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import pytest
import yaml

from src.cifar_classifier.data.process_data import _get_transformations

# Constants for mocking
TRAIN = "train"
VAL = "val"


@pytest.fixture
def mock_config() -> dict:
    """Mock configuration data."""
    return {
        "data": {
            "image_size": 32,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "random_rotation": 10,
            "random_horizontal_flip": 0.5,
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
            },
            "random_sharpness": {
                "factor": 2.0,
                "p": 0.5,
            },
            "random_erasing": {
                "probability": 0.5,
                "scale": [0.02, 0.33],
                "value": 0,
                "inplace": False,
            },
        },
    }


@pytest.fixture
def mock_transforms() -> Mock:
    """Mock the transforms module."""
    with mock.patch("src.cifar_classifier.data.process_data.transforms") as mock_trans:
        yield mock_trans


@pytest.fixture
def mock_logger() -> Mock:
    """Mock the logger."""
    with mock.patch("src.cifar_classifier.data.process_data.logger") as mock_log:
        yield mock_log


@pytest.fixture
def mock_open(mocker: Mock, mock_config: dict) -> Mock:
    """Mock the open function."""
    return mocker.patch(
        "builtins.open", mocker.mock_open(read_data=yaml.dump(mock_config))
    )


@pytest.fixture
def config_path(tmp_path: Path, mock_config: dict) -> Path:
    """Create a temporary config file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(mock_config))
    return config_file


def test_get_transformations_train(
    mock_transforms: Mock,
    mock_open: Mock,
    config_path: Path,
    mock_logger: Mock,
) -> None:
    """Test getting transformations for the training dataset."""
    part = TRAIN
    _get_transformations(part, config_path)

    # Check that transforms.Compose was called with expected operations
    assert mock_transforms.Compose.call_count == 1
    calls = mock_transforms.Compose.call_args_list[0][0][0]

    assert len(calls) == 8  # Number of transformations for TRAIN
    mock_logger.debug.assert_called_with(
        "Getting transformations for training dataset."
    )
    mock_logger.info.assert_called_with("Transformations for training dataset loaded.")


def test_get_transformations_test(
    mock_transforms: Mock,
    mock_open: Mock,
    config_path: Path,
    mock_logger: Mock,
) -> None:
    """Test getting transformations for the val dataset."""
    part = VAL
    _get_transformations(part, config_path)

    # Check that transforms.Compose was called with expected operations
    assert mock_transforms.Compose.call_count == 1
    calls = mock_transforms.Compose.call_args_list[0][0][0]

    assert len(calls) == 3  # Number of transformations for VAL
    mock_logger.debug.assert_called_with("Getting transformations for val dataset.")
    mock_logger.info.assert_called_with("Transformations for val dataset loaded.")


def test_invalid_part(mock_logger: Mock, config_path: Path) -> None:
    """Test that the function exits on an invalid dataset part."""
    invalid_part = "invalid"

    with mock.patch("sys.exit") as mock_exit:
        _get_transformations(invalid_part, config_path)
        mock_logger.error.assert_called_with(f"Invalid part {invalid_part}.")
        mock_exit.assert_called_once_with(1)


def test_file_not_found_error(mock_logger: Mock) -> None:
    """Test handling of a missing configuration file."""
    config_path = Path("non_existent_file.yaml")

    with mock.patch("sys.exit") as mock_exit:
        _get_transformations(TRAIN, config_path)
        mock_logger.error.assert_called_with(
            f"Configuration file not found in {config_path}."
        )
        mock_exit.assert_called_once_with(1)


def test_yaml_error(mock_open: Mock, mock_logger: Mock) -> None:
    """Test handling of a YAML parsing error."""
    mock_open.side_effect = yaml.YAMLError("mock error")

    with mock.patch("sys.exit") as mock_exit:
        _get_transformations(TRAIN, Path("mock_path"))
        mock_logger.error.assert_called()
        mock_exit.assert_called_once_with(1)


def test_general_exception(mock_open: Mock, mock_logger: Mock) -> None:
    """Test handling of a general exception."""
    mock_open.side_effect = Exception("mock error")

    with mock.patch("sys.exit") as mock_exit:
        _get_transformations(TRAIN, Path("mock_path"))
        mock_logger.error.assert_called()
        mock_exit.assert_called_once_with(1)

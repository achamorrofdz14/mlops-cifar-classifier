import pytest
from unittest import mock
from unittest.mock import patch, MagicMock

import torch
import pickle
from src.cifar_classifier.model.train_model import train_model, load_datasets, device

# Mock constants for paths
TRAIN_PATH = "mock_train_path"
VAL_PATH = "mock_val_path"


@pytest.fixture
def mock_config():
    return {
        "training": {
            "model": {
                "out_conv_1": 64,
                "out_conv_2": 128,
                "out_conv_3": 256,
                "dropout_p": 0.5,
            },
            "batch_size": 32,
            "epochs": 2,
            "learning_rate": 0.01,
            "momentum": 0.9,
        }
    }


@pytest.fixture
def mock_train_data():
    return [(torch.randn(3, 32, 32), torch.tensor(1)) for _ in range(64)]


@pytest.fixture
def mock_val_data():
    return [(torch.randn(3, 32, 32), torch.tensor(1)) for _ in range(16)]


def test_train_model(mock_config, mock_train_data, mock_val_data):
    # Mock the CIFARCNN class
    with mock.patch("src.cifar_classifier.model.train_model.CIFARCNN") as MockCIFARCNN:
        # Create a mock instance of CIFARCNN
        mock_model = MockCIFARCNN.return_value

        # Ensure mock_model.to(device) works as expected
        mock_model.to.return_value = mock_model

        # Continue with other necessary mocks
        mock_optimizer = mock.Mock()
        mock_criterion = mock.Mock()

        # Mocking the loss to behave like a tensor with a .item() method
        mock_loss = mock.Mock()
        mock_loss.item.return_value = (
            1.0  # Assuming the loss value is 1.0 for simplicity
        )
        mock_criterion.return_value = mock_loss

        # Mocking the output of the model (z) to behave like a tensor
        mock_z = mock.Mock()
        mock_z.data = torch.randn(
            32, 10
        )  # Example output with batch size 32 and 10 classes
        mock_model.return_value = mock_z

        # Mocking other components as necessary
        with mock.patch(
            "src.cifar_classifier.model.train_model.torch.optim.SGD",
            return_value=mock_optimizer,
        ):
            with mock.patch(
                "src.cifar_classifier.model.train_model.nn.CrossEntropyLoss",
                return_value=mock_criterion,
            ):
                with mock.patch(
                    "src.cifar_classifier.model.train_model.load_datasets",
                    return_value=(mock_train_data, mock_val_data),
                ):
                    with mock.patch(
                        "src.cifar_classifier.model.train_model.load_config",
                        return_value=mock_config,
                    ):
                        with mock.patch(
                            "src.cifar_classifier.model.train_model.torch.utils.data.DataLoader"
                        ) as mock_dataloader:
                            mock_dataloader.side_effect = (
                                lambda dataset, batch_size: dataset
                            )

                            # Catch the SystemExit exception
                            with pytest.raises(SystemExit) as excinfo:
                                train_model()

                            # Verify the exit code was 0
                            assert excinfo.value.code == 0

                            # Perform assertions
                            mock_model.to.assert_called_once_with(
                                torch.device(
                                    "cuda" if torch.cuda.is_available() else "cpu"
                                )
                            )
                            mock_optimizer.zero_grad.assert_called()
                            mock_optimizer.step.assert_called()
                            mock_criterion.assert_called()
                            mock_model.train.assert_called()
                            mock_model.eval.assert_called()


def test_load_datasets(mock_train_data, mock_val_data):

    # Mock paths to avoid using actual file system paths
    mock_train_path = "mock_train_path"
    mock_val_path = "mock_val_path"

    # Create two different mock objects for the two different datasets
    mock_open_train = mock.mock_open(read_data=pickle.dumps(mock_train_data))
    mock_open_val = mock.mock_open(read_data=pickle.dumps(mock_val_data))

    # Use side_effect to return the correct mock object based on the input path
    def mock_open_side_effect(path, *args, **kwargs):
        if path == TRAIN_PATH:
            return mock_open_train(path, *args, **kwargs)
        elif path == VAL_PATH:
            return mock_open_val(path, *args, **kwargs)
        else:
            raise ValueError(f"Unrecognized path: {path}")

    with mock.patch(
        "src.cifar_classifier.model.train_model.TRAIN_PATH", "mock_train_path"
    ):
        with mock.patch(
            "src.cifar_classifier.model.train_model.VAL_PATH", "mock_val_path"
        ):
            with patch(
                "src.cifar_classifier.model.train_model.Path.open",
                side_effect=mock_open_side_effect,
            ):
                train_dataset, val_dataset = load_datasets()

                # Ensure the datasets have the correct lengths
                assert len(train_dataset) == len(mock_train_data)
                assert len(val_dataset) == len(mock_val_data)

                # Check that the datasets match the mock data
                for t1, t2 in zip(train_dataset, mock_train_data):
                    assert torch.equal(t1[0], t2[0])
                    assert t1[1] == t2[1]

                for t1, t2 in zip(val_dataset, mock_val_data):
                    assert torch.equal(t1[0], t2[0])
                    assert t1[1] == t2[1]

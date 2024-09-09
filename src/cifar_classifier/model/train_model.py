# ruff: noqa: PLW2901
"""Implementation of the train_model function."""

import pickle
from pathlib import Path

import click
import mlflow
import torch
from loguru import logger
from torch import nn

from src.cifar_classifier import MODEL_FPATH, TRAIN_PATH, VAL_PATH
from src.cifar_classifier.model.model import CIFARCNN
from src.cifar_classifier.utils.config_utils import load_config
from src.cifar_classifier.utils.git_utils import get_git_user_name

# Assign GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_datasets() -> tuple:
    """Load the train and validation datasets."""
    with Path.open(TRAIN_PATH, "rb") as f:
        train_dataset = pickle.load(f)
    with Path.open(VAL_PATH, "rb") as f:
        validation_dataset = pickle.load(f)
    return train_dataset, validation_dataset


def load_model(config: dict) -> CIFARCNN:
    """Load the CIFAR model."""
    config_model = config["training"]["model"]

    model = CIFARCNN(
        out_1=config_model["out_conv_1"],
        out_2=config_model["out_conv_2"],
        out_3=config_model["out_conv_3"],
        p=config_model["dropout_p"],
    )
    model = model.to(device)

    logger.debug("Model loaded")
    return model


@click.command("train-model")
@click.option("--experiment-name", help="Name of the experiment")
@click.option("--run-reason", help="Reason for running the training")
@click.option("--team", help="Team responsible for the training")
def train_model(
    experiment_name: str,
    run_reason: str,
    team: str,
) -> None:
    """Train the CIFAR model."""
    mlflow.set_experiment(experiment_name)

    config = load_config()

    model = load_model(config)

    train_dataset, validation_dataset = load_datasets()

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config["training"]["batch_size"],
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=config["training"]["batch_size"],
    )

    logger.debug("Train and test dataset loaded")

    n_test = len(validation_dataset)

    accuracy_list = []
    train_cost_list = []
    val_cost_list = []

    criterion = nn.CrossEntropyLoss()
    n_epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=config["training"]["momentum"],
    )

    for epoch in range(n_epochs):
        train_cost = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            model.train()
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            train_cost += loss.item()

        train_cost = train_cost / len(train_loader)
        train_cost_list.append(train_cost)
        correct = 0

        val_cost = 0
        for x_test, y_test in validation_loader:
            model.eval()
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            z = model(x_test)
            val_loss = criterion(z, y_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
            val_cost += val_loss.item()

        val_cost = val_cost / len(validation_loader)
        val_cost_list.append(val_cost)

        accuracy = correct / n_test
        accuracy_list.append(accuracy)

        logger.info(
            f"Epoch Results {epoch + 1}, train loss: {round(train_cost,4)}, "
            f"val loss: {round(val_cost,4)}, val accuracy: {round(accuracy * 100, 2)}",
        )

    with mlflow.start_run():

        mlflow.set_tags(
            {
                "Run Reason": run_reason,
                "Responsible User": get_git_user_name(),
                "Team": team,
                "Model Type": "CNN",
            },
        )

        mlflow.log_params(config["training"])
        mlflow.log_metrics(
            {
                "train_loss": train_cost,
                "val_loss": val_cost,
                "val_accuracy": accuracy,
            },
        )
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="cifar_classifier_model",
            code_paths=[MODEL_FPATH],
        )


if __name__ == "__main__":
    train_model()

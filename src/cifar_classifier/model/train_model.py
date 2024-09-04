# ruff: noqa: PLW2901
"""Implementation of the train_model function."""

import pickle
from pathlib import Path

import click
import torch
from loguru import logger
from torch import nn

from src.cifar_classifier import TRAIN_PATH, VAL_PATH
from src.cifar_classifier.model.model import CIFARCNN
from src.cifar_classifier.utils.config_utils import load_config

# Assign GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_datasets() -> tuple:
    """Load the train and validation datasets."""
    with Path.open(TRAIN_PATH, "rb") as f:
        train_dataset = pickle.load(f)
    with Path.open(VAL_PATH, "rb") as f:
        validation_dataset = pickle.load(f)
    return train_dataset, validation_dataset


@click.command("train-model")
def train_model() -> None:
    """Train the CIFAR model."""
    logger.info(
        "-" * 20,
        "Training started",
        "-" * 20,
    )

    config = load_config()
    config_model = config["training"]["model"]

    model = CIFARCNN(
        out_1=config_model["out_conv_1"],
        out_2=config_model["out_conv_2"],
        out_3=config_model["out_conv_3"],
        p=config_model["dropout_p"],
    )
    model = model.to(device)

    logger.debug("Model loaded")

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


if __name__ == "__main__":
    train_model()

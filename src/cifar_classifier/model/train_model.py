# ruff: noqa
"""Implementation of the train_model function."""

import torch
import yaml
from loguru import logger
from torch.nn import nn

from src.cifar_classifier import CONFIG_DIR, DATA_DIR
from src.cifar_classifier.model.model import CIFARCNN

# Assign GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    train_loader,
    validation_loader,
):

    logger.info(
        "-" * 20,
        "Training started",
        "-" * 20,
    )

    config_path = CONFIG_DIR / "config.yml"

    # Load the configuration file
    with config_path.open() as file:
        config = yaml.safe_load(file)

    logger.debug(f"Configuration file loaded from {config_path}")

    config_model = config["training"]["model"]

    model = CIFARCNN(
        out_1=config_model["out_conv_1"],
        out_2=config_model["out_conv_2"],
        out_3=config_model["out_conv_3"],
        p=config_model["dropout_p"],
    )
    model = model.to(device)

    logger.debug("Model loaded")

    # TODO: Read dataset

    # Create train and validation batch for training
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config["training"]["batch_size"],
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=config["training"]["batch_size"],
    )

    logger.debug("Train and test dataset loaded")

    # Global variable
    N_test = len(validation_dataset)

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

        # Perform the prediction on the validation data
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

        accuracy = correct / N_test
        accuracy_list.append(accuracy)

        logger.info(
            f"--> Epoch Number : {epoch + 1}",
            f" | Training Loss : {round(train_cost,4)}",
            f" | Validation Loss : {round(val_cost,4)}",
            f" | Validation Accuracy : {round(accuracy * 100, 2)}%",
        )

    return accuracy_list, train_cost_list, val_cost_list

# ruff: noqa
"""Implementation of the train_model function."""

import pickle
from pathlib import Path

import click
import mlflow
import torch
from loguru import logger

# from sklearn.model_selection import StratifiedKFold
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

    logger.debug(f"Model loaded on {device}")
    return model


def train(model, optimizer, criterion, data_loader):
    """Train the model."""
    train_cost = 0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        model.train()
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z, y)
        loss.backward()
        optimizer.step()
        train_cost += loss.item()

    return train_cost


def evaluate(model, criterion, data_loader):
    """Evaluate the model."""
    correct = 0
    val_cost = 0

    for x_test, y_test in data_loader:
        model.eval()
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        z = model(x_test)
        val_loss = criterion(z, y_test)
        _, yhat = torch.max(z.data, 1)
        correct += (yhat == y_test).sum().item()
        val_cost += val_loss.item()

    return val_cost, correct


@click.command("train-model")
@click.option("--experiment-name", help="Name of the experiment")
@click.option("--run-reason", help="Reason for running the training")
@click.option("--team", help="Team responsible for the training")
@click.option("--n_splits", default=3, help="Number of splits for the cross-validation")
def train_model(
    experiment_name: str,
    run_reason: str,
    team: str,
    n_splits: int = 3,
) -> None:
    """Train the CIFAR model."""
    mlflow.set_experiment(experiment_name)

    config = load_config()

    model = load_model(config)

    train_dataset, validation_dataset = load_datasets()

    # y_train = [label for _, label in train_dataset]

    # skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # for fold, (train_index, val_index) in enumerate(skf.split(train_dataset, y_train)):

    #     logger.info(f"Training fold {fold + 1}/{n_splits}")

    #     train_dataset_fold = torch.utils.data.Subset(train_dataset, train_index)
    #     validation_dataset_fold = torch.utils.data.Subset(train_dataset, val_index)

    #     train_fold_loader = torch.utils.data.DataLoader(
    #         dataset=train_dataset_fold,
    #         batch_size=config["training"]["batch_size"],
    #     )

    #     validation_fold_loader = torch.utils.data.DataLoader(
    #         dataset=validation_dataset_fold,
    #         batch_size=config["training"]["batch_size"],
    #     )

    #     n_fold = len(validation_dataset_fold)

    #     accuracy_list = []
    #     train_cost_list = []
    #     val_cost_list = []

    #     len_train_fold_loader = len(train_fold_loader)
    #     len_val_fold_loader = len(validation_fold_loader)

    #     criterion = nn.CrossEntropyLoss()
    #     n_epochs = config["training"]["epochs"]
    #     learning_rate = config["training"]["learning_rate"]
    #     optimizer = torch.optim.SGD(
    #         model.parameters(),
    #         lr=learning_rate,
    #         momentum=config["training"]["momentum"],
    #     )

    #     for epoch in range(n_epochs):
    #         train_cost = 0
    #         train(model, optimizer, criterion, train_fold_loader)

    #         train_cost = train_cost / len_train_fold_loader
    #         train_cost_list.append(train_cost)

    #         val_cost, correct = evaluate(model, criterion, validation_fold_loader)

    #         val_cost = val_cost / len_val_fold_loader
    #         val_cost_list.append(val_cost)

    #         accuracy = correct / n_fold
    #         accuracy_list.append(accuracy)

    #         logger.info(
    #             f"Epoch Results {epoch + 1}, train loss: {round(train_cost,4)}, "
    #             f"val loss: {round(val_cost,4)}, val accuracy: {round(accuracy * 100, 2)}",
    #         )

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

    len_train_loader = len(train_loader)
    len_val_loader = len(validation_loader)

    criterion = nn.CrossEntropyLoss()
    n_epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=config["training"]["momentum"],
    )

    for epoch in range(n_epochs):
        train_cost = train(model, optimizer, criterion, train_loader)

        train_cost = train_cost / len_train_loader
        train_cost_list.append(train_cost)

        val_cost, correct = evaluate(model, criterion, validation_loader)

        val_cost = val_cost / len_val_loader
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

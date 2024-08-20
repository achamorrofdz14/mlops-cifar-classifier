# ruff: noqa: TRY003, FBT001, FBT003
"""Process data for CIFAR-10 classifier."""

import sys
from pathlib import Path

import click
import torchvision.datasets as dsets
import yaml
from loguru import logger
from torchvision.transforms import transforms

from src.cifar_classifier import CONFIG_DIR, DATA_DIR, TEST, TRAIN


def _get_transformations(
    part: str, config_path: Path,
) -> transforms.Compose | tuple[transforms.Compose, transforms.Compose]:
    """Get transformations for CIFAR-10 dataset.

    :param part: part of the dataset to transform
    :param config_path: Path where configuration file is stored

    :return: composed_train, composed_test: transformations for training and test datasets
    """
    logger.debug(f"Getting transformations for {part} dataset.")

    try:

        # Check if the part is valid
        if part not in [TRAIN, TEST]:
            logger.error(f"Invalid part {part}.")
            sys.exit(1)

        # Load the configuration file
        with config_path.open() as file:
            config = yaml.safe_load(file)

        logger.info(f"Configuration file loaded from {config_path}")

        # Use config data values
        config = config["data"]

        image_size = config["image_size"]

        mean, std = config["mean"], config["std"]

        if part == TRAIN:
            logger.debug("Getting transformations for training dataset.")
            composed_train = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(config["random_rotation"]),
                transforms.RandomHorizontalFlip(config["random_horizontal_flip"]),
                transforms.ColorJitter(brightness = config["color_jitter"]["brightness"],
                                        contrast = config["color_jitter"]["contrast"],
                                        saturation = config["color_jitter"]["saturation"]),
                transforms.RandomAdjustSharpness(
                    sharpness_factor=config["random_sharpness"]["factor"],
                    p = config["random_sharpness"]["p"],
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=config["random_erasing"]["probability"],
                                        scale=tuple(config["random_erasing"]["scale"]),
                                        value=config["random_erasing"]["value"],
                                        inplace=config["random_erasing"]["inplace"])])
            logger.info("Transformations for training dataset loaded.")

            return composed_train
        if part == TEST:
            logger.debug("Getting transformations for test dataset.")
            composed_test = transforms.Compose([transforms.Resize((image_size,image_size)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])
            logger.info("Transformations for test dataset loaded.")

            return composed_test

    except FileNotFoundError:
        logger.error(f"Configuration file not found in {config_path}.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAMLS file: {e.with_traceback()}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred getting trainsformations: {e.with_traceback()}")
        sys.exit(1)

def _load_transform(
    is_train: bool,
    out_dir: Path,
    composed: transforms.Compose,
    ) -> dsets.CIFAR10:
    """Transform CIFAR-10 dataset.

    :param is_train: flag to download and process training data
    :param out_dir: Path where output data will be saved
    :param composed: transformations to be applied to the dataset

    :return: CIFAR-10 dataset transformed
    """
    logger.debug("Downloading and transforming CIFAR-10 dataset.")

    try:
        # Load and transform the dataset
        trans_dataset = dsets.CIFAR10(
            root=out_dir,
            train=is_train,
            download=True,
            transform=composed,
        )
    except Exception as e:
        logger.error(f"An error occurred loading and transforming the dataset:{e.with_traceback()}")
        sys.exit(1)

    return trans_dataset

@click.command("process-data")
@click.option(
    "--train",
    default=False,
    is_flag=True,
    help="Enables download and process training data.",
)
@click.option(
    "--validation",
    default=False,
    is_flag=True,
    help="Enables download and process validation data.",
)
@click.option(
    "--out-train-dir",
    default=DATA_DIR,
    type=click.Path(True, path_type=Path),
    help="Output directory for training data.",
)
@click.option(
    "--out-validation-dir",
    default=DATA_DIR,
    type=click.Path(True, path_type=Path),
    help="Output directory for validation data.",
)
@click.option(
    "--config-path",
    default=CONFIG_DIR / "config.yml",
    type=click.Path(True, path_type=Path),
    help="Path to configuration file.",
)
def process_data(
    train: bool,
    validation: bool,
    out_train_dir: Path,
    out_validation_dir: Path,
    config_path: Path,
    ) -> dsets.CIFAR10 | tuple[dsets.CIFAR10, dsets.CIFAR10]:
    """Process CIFAR-10 dataset.

    :param train: flag to download and process training data
    :param validation: flag to download and process validation data
    :param out_train_dir: Path where output train data will be saved
    :param out_validation_dir: Path where output validation data will be saved
    :param config_path: Path where configuration file is stored

    :return train_dataset, validation_dataset: train and test datasets transformed
    """
    if not (train or validation):
        raise click.UsageError("At least one of --train or --validation must be provided.")

    logger.debug("Processing CIFAR-10 dataset.")

    if train:
        logger.debug("Processing training data.")
        composed_train = _get_transformations(TRAIN, config_path)
        train_dataset = _load_transform(is_train=True,
                                        out_dir=out_train_dir,
                                        composed=composed_train)
        logger.info("CIFAR-10 training dataset loaded and transformed.")

    if validation:
        logger.debug("Processing validation data.")
        composed_test = _get_transformations(TEST, config_path)
        validation_dataset = _load_transform(is_train=False,
                                             out_dir=out_validation_dir,
                                             composed=composed_test)
        logger.info("CIFAR-10 validation dataset loaded and transformed.")

    if train and validation:
        return train_dataset, validation_dataset
    if train:
        return train_dataset
    if validation:
        return validation_dataset


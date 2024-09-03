# ruff: noqa: D401
"""Main module for the CLI."""
import click

from .data import process_data
from .model import train_model


def _main() -> None:
    """Provide the main entry point for the CLI."""

    @click.group(chain=True)
    def entry_point() -> None:
        """MLOps cifar classifier test command line application."""

    entry_point.add_command(
        process_data.process_data,
    )

    entry_point.add_command(
        train_model.train_model,
    )

    entry_point()


if __name__ == "__main__":
    _main()

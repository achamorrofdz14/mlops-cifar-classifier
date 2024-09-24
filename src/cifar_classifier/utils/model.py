"""Contains the functions to manage the model."""

import mlflow
from mlflow.tracking import MlflowClient


def save_model_to_registry(run_id: str, new_stage: str = "Production") -> None:
    """Save the model to the MLFlow registry."""
    # Create connection to MLFlow
    client = MlflowClient()

    # Create the model URI based on the run_id
    model_uri = f"runs:/{run_id}/cifar_classifier_model"
    model_details = mlflow.register_model(
        model_uri=model_uri,
        name="torch_cifar_classifier_cnn",
    )

    # Transition the model to the new stage
    client.transition_model_version_stage(
        name=model_details.name,
        version=model_details.version,
        stage=new_stage,
        archive_existing_versions=True,
    )


def get_best_and_last_experiment_run_id(
    experiment_id: str,
    threshold: float,
) -> str:
    """Get the run_id of the best experiment based on the threshold."""
    experiment = mlflow.get_experiment(experiment_id)
    query = f"metrics.mean_val_accuracy > {threshold}"
    search_results = mlflow.search_runs(
        experiment.experiment_id,
        filter_string=query,
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
    )
    last_run = search_results[
        search_results["start_time"] == search_results["start_time"].max()
    ]
    return last_run["run_id"][0]

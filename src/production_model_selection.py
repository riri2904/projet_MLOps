import argparse
import pathlib
import sys
from pprint import pprint

import joblib
import mlflow
from mlflow.tracking import MlflowClient

from train import read_params


def log_production_model(config_path):
    config = read_params(config_path)
    mlflow_config = config["mlflow_global_config"]
    model_name = mlflow_config["registered_model_name"]
    model_dir = (
        pathlib.Path(__file__).parent.parent.resolve() / config["model_dir"]
    )  # Path where model will be stored
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    runs = mlflow.search_runs(experiment_ids=[1])

    max_accuracy = max(runs["metrics.accuracy"])
    max_accuracy_run_id = list(
        runs[runs["metrics.accuracy"] == max_accuracy]["run_id"]
    )[0]

    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
        if mv["run_id"] == max_accuracy_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]
            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=model_name, version=current_version, stage="Production"
            )
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=model_name, version=current_version, stage="Staging"
            )

    loaded_model = mlflow.pyfunc.load_model(logged_model)
    joblib.dump(loaded_model, model_dir)


if __name__ == "__main__":
    path_yaml = pathlib.Path(__file__).parent.parent.resolve() / "params.yaml"
    sys.argv = [""]
    del sys
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=path_yaml)
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)

import argparse
import pathlib
import sys

import pandas as pd
import yaml
from evidently.dashboard import Dashboard
from evidently.tabs import CatTargetDriftTab, DataDriftTab


def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def category_to_numeric(x):
    if x == "b'Hernia'":
        return 0
    if x == "b'Normal'":
        return 1
    if x == "b'Spondylolisthesis'":
        return 2
    else:
        return x


def model_monitoring(config_path):
    config = read_params(config_path)
    train_data_path = (
        pathlib.Path(__file__).parent.parent.resolve()
        / config["raw_data_config"]["raw_data_csv"]
    )
    new_train_data_path = (
        pathlib.Path(__file__).parent.parent.resolve()
        / config["raw_data_config"]["new_train_data_csv"]
    )
    target = config["raw_data_config"]["target"]
    monitor_dashboard_path = (
        pathlib.Path(__file__).parent.parent.resolve()
        / config["model_monitor"]["monitor_dashboard_html"]
    )
    monitor_target = config["model_monitor"]["target_col_name"]

    ref = pd.read_csv(train_data_path)
    cur = pd.read_csv(new_train_data_path)

    ref = ref.rename(columns={target: monitor_target}, inplace=False)
    cur = cur.rename(columns={target: monitor_target}, inplace=False)

    ref[target] = ref[target].apply(category_to_numeric)
    cur[monitor_target] = cur[monitor_target].apply(category_to_numeric)

    ref.rename(columns={target: "target"}, inplace=True)
    cur.rename(columns={monitor_target: "target"}, inplace=True)

    data_and_target_drift_dashboard = Dashboard(
        tabs=[DataDriftTab, CatTargetDriftTab]
    )
    data_and_target_drift_dashboard.calculate(ref, cur, column_mapping=None)
    data_and_target_drift_dashboard.save(monitor_dashboard_path)


if __name__ == "__main__":
    path_yaml = pathlib.Path(__file__).parent.parent.resolve() / "params.yaml"
    sys.argv = [""]
    del sys
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=path_yaml)
    parsed_args = args.parse_args()
    model_monitoring(config_path=parsed_args.config)

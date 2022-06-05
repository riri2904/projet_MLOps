import argparse
import json
import pathlib
import sys
from urllib.parse import urlparse

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier


def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def accuracymeasures(y_test, predictions, avg_method):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average=avg_method)
    recall = recall_score(y_test, predictions, average=avg_method)
    f1score = f1_score(y_test, predictions, average=avg_method)
    target_names = ["0", "1", "2"]
    print("Classification report")
    print("---------------------", "\n")
    print(
        classification_report(y_test, predictions, target_names=target_names),
        "\n",
    )
    print("Confusion Matrix")
    print("---------------------", "\n")
    print(confusion_matrix(y_test, predictions), "\n")

    print("Accuracy Measures")
    print("---------------------", "\n")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1score)

    return accuracy, precision, recall, f1score


def get_feat_and_target(df, target):
    """
    Get features and target variables seperately from given dataframe and target
    input: dataframe and target column
    output: two dataframes for x and y
    """
    x = df.drop(target, axis=1)
    y = df[[target]]
    return x, y


def train_and_evaluate(config_path):
    """
    Train & evaluate model. Performances tracking with mlflow : http://localhost:1234/
    output: two dataframes for x and y
    """
    config = read_params(config_path)
    train_data_path = (
        pathlib.Path(__file__).parent.parent.resolve()
        / config["processed_data_config"]["train_data_csv"]
    )
    test_data_path = (
        pathlib.Path(__file__).parent.parent.resolve()
        / config["processed_data_config"]["test_data_csv"]
    )

    model_choice = config["model_choice"]["model"]

    if model_choice == "random_forest":
        max_depth = config["random_forest"]["max_depth"]
        n_estimators = config["random_forest"]["n_estimators"]
        run_name = config["mlflow_random_forest_config"]["run_name"]

    if model_choice == "knn":
        n_neighbors = config["knn"]["n_neighbors"]
        run_name = config["mlflow_knn_config"]["run_name"]

    target = config["raw_data_config"]["target"]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")
    train_x, train_y = get_feat_and_target(train, target)
    test_x, test_y = get_feat_and_target(test, target)

    ################### MLFLOW ###############################
    mlflow_config = config["mlflow_global_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=run_name) as mlops_run:
        if model_choice == "random_forest":
            model = RandomForestClassifier(
                max_depth=max_depth, n_estimators=n_estimators
            )
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("n_estimators", n_estimators)

        if model_choice == "knn":
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            mlflow.log_param("n_neighbors", n_neighbors)

        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        accuracy, precision, recall, f1score = accuracymeasures(
            test_y, y_pred, "weighted"
        )

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1score)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=mlflow_config["registered_model_name"],
            )
        else:
            mlflow.sklearn.load_model(model, "model")


if __name__ == "__main__":
    path_yaml = pathlib.Path(__file__).parent.parent.resolve() / "params.yaml"
    sys.argv = [""]
    del sys
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=path_yaml)
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)

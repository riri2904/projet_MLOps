import argparse
import os
import pathlib
import sys
from random import shuffle

import pandas as pd
from sklearn.model_selection import train_test_split

from src.load_data import read_params


def processing_data(df, target):
    """
    remove the 2 first caracters of the column class and the last caracter
    the data of 'class' have the form : b'...' where ... are the response
    """
    df[target] = df[target].str[2:]
    df[target] = df[target].str[:-1]

    return df


def split_data(df, train_data_path, test_data_path, split_ratio, random_state):

    train, test = train_test_split(
        df, test_size=split_ratio, random_state=random_state, shuffle=True
    )
    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")


def split_and_saved_data(config_path):
    """
    split the train dataset(data/raw) and save it in the data/processed folder
    input: config path
    output: save splitted files in output folder
    """
    config = read_params(config_path)

    target = config["raw_data_config"]["target"]

    raw_data_csv = (
        pathlib.Path(__file__).parent.parent.resolve()
        / config["raw_data_config"]["raw_data_csv"]
    )

    test_data_path = (
        pathlib.Path(__file__).parent.parent.resolve()
        / config["processed_data_config"]["test_data_csv"]
    )
    train_data_path = (
        pathlib.Path(__file__).parent.parent.resolve()
        / config["processed_data_config"]["train_data_csv"]
    )
    split_ratio = config["raw_data_config"]["train_test_split_ratio"]
    random_state = config["raw_data_config"]["random_state"]

    raw_df = pd.read_csv(raw_data_csv)
    raw_df = processing_data(raw_df, target)
    split_data(
        raw_df, train_data_path, test_data_path, split_ratio, random_state
    )


if __name__ == "__main__":
    path_yaml = pathlib.Path(__file__).parent.parent.resolve() / "params.yaml"
    sys.argv = [""]
    del sys
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=path_yaml)
    parsed_args = args.parse_args()
    split_and_saved_data(config_path=parsed_args.config)

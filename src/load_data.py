import argparse
import pathlib
import sys
from io import BytesIO, TextIOWrapper
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yaml
from scipy.io import arff
from scipy.io.arff import loadarff


def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def load_raw_data(config_path):
    """
    load data from external location(data/external) to the raw folder(data/raw) with train and teting dataset
    input: config_path
    output: save train file in data/raw folder
    """
    config = read_params(config_path)
    raw_url_data = config["raw_data_config"]["url_data"]
    raw_filename_data = config["raw_data_config"]["filename_data"]
    raw_data_csv = (
        pathlib.Path(__file__).parent.parent.resolve()
        / config["raw_data_config"]["raw_data_csv"]
    )

    resp = urlopen(raw_url_data)
    zipfile = ZipFile(BytesIO(resp.read()))

    in_mem_fo = TextIOWrapper(
        zipfile.open(raw_filename_data), encoding="ascii"
    )
    data = loadarff(in_mem_fo)
    df = pd.DataFrame(data[0])

    df.to_csv(raw_data_csv, index=False)


if __name__ == "__main__":
    path_yaml = pathlib.Path(__file__).parent.parent.resolve() / "params.yaml"
    sys.argv = [""]
    del sys
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=path_yaml)
    parsed_args = args.parse_args()
    load_raw_data(config_path=parsed_args.config)

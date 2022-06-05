import pandas as pd

from src.data_processing import *
from src.train import get_feat_and_target


def test_get_feat_and_target():
    df = pd.DataFrame(columns=["A", "B", "C"])
    target = "B"
    x, y = get_feat_and_target(df, target)

    assert all([a == b for a, b in zip(x.columns, ["A", "C"])])
    assert all([a == b for a, b in zip(y.columns, ["B"])])


def test_processing_data():
    df = pd.DataFrame(
        {
            "A": ["b'Hernia'"],
            "B": ["b'Spondylolisthesis'"],
            "C": ["b'Normal'"],
        },
        columns=["A", "B", "C"],
    )

    target = "A"
    x = processing_data(df, target)

    target = "B"
    y = processing_data(df, target)

    target = "C"
    z = processing_data(df, target)

    assert all([a == b for a, b in zip(df["A"], x["A"])])
    assert all([a == b for a, b in zip(df["B"], x["B"])])
    assert all([a == b for a, b in zip(df["C"], x["C"])])

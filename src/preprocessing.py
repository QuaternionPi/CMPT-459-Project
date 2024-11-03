import pandas as pd
import numpy as np


def preprocess(path: str) -> pd.DataFrame:
    """
    Preprocess data.

    :param path: Path to data file (csv)
    :return: Pandas data frame of kept data
    """
    df: pd.DataFrame = pd.read_csv(path)
    return df

import pandas as pd
import numpy as np
from writer import Writer


def preprocess(path: str, writer: Writer) -> pd.DataFrame:
    """
    Preprocess data.

    :param path: Path to data file (csv).
    :param writer: Where to write outputs.
    :return: Pandas data frame of kept data.
    """
    df: pd.DataFrame = pd.read_csv(path)
    return df

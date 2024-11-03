import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.metrics import mutual_info_score
from writer import Writer


def recursive_feature_elimination(data: pd.DataFrame, writer: Writer) -> pd.DataFrame:
    """
    Performs recursive feature elimination.

    :return: feature-reduced data frame.
    """
    pass


def lasso_regression(data: pd.DataFrame, writer: Writer) -> pd.DataFrame:
    """
    Performs lasso regression.

    :return: feature-reduced data frame.
    """
    pass


def mutual_information(data: pd.DataFrame, writer: Writer) -> pd.DataFrame:
    """
    Removes columns above a certain mutual information score.

    :return: feature-reduced data frame.
    """
    pass

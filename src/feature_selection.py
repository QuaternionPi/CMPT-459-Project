import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.svm import SVR
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.metrics import mutual_info_score
from writer import Writer


def recursive_feature_elimination(data: pd.DataFrame, writer: Writer) -> pd.DataFrame:
    """
    Performs recursive feature elimination.

    :return: feature-reduced data frame.
    """
    y = data["Rainfall"]
    X = data.drop(columns=["Rainfall"], inplace=False)

    estimator = SVR(kernel="linear")

    selector = RFE(estimator, n_features_to_select=5, step=1)
    selector = selector.fit(X, y)

    dropped_columns = [
        col for col, keep in zip(data.columns, selector.get_support()) if keep
    ]
    return data.drop(columns=dropped_columns, inplace=False)


def lasso_regression(data: pd.DataFrame, writer: Writer) -> pd.DataFrame:
    """
    Performs lasso regression.

    :return: feature-reduced data frame.
    """
    y = data["Rainfall"]
    X = data.drop(columns=["Rainfall"], inplace=False)

    estimator = StandardScaler()
    estimator.fit(X)

    lasso = Lasso(alpha=0.2)

    selector = SelectFromModel(lasso)
    selector = selector.fit(estimator.transform(X), y)

    dropped_columns = [
        col for col, keep in zip(data.columns, selector.get_support()) if keep
    ]
    return data.drop(columns=dropped_columns, inplace=False)


def mutual_information(data: pd.DataFrame, writer: Writer) -> pd.DataFrame:
    """
    Removes columns above a certain mutual information score.

    :return: feature-reduced data frame.
    """
    pass

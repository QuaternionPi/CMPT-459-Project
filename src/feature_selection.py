import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.svm import SVR
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
)
from sklearn.feature_selection import mutual_info_classif as mutual_info_classifier
from sklearn.linear_model import Lasso
from writer import Writer


def recursive_feature_elimination(data: pd.DataFrame, writer: Writer) -> pd.DataFrame:
    """
    Performs recursive feature elimination.

    :return: feature-reduced data frame.
    """
    y = data["RainTomorrow"]
    X = data.drop(columns=["RainTomorrow"], inplace=False)

    estimator = SVR(kernel="linear")

    selector = RFE(estimator, n_features_to_select=5, step=1)
    selector = selector.fit(X, y)

    dropped_columns = [
        col for col, keep in zip(data.columns, selector.get_support()) if not keep
    ]
    result = data.drop(columns=dropped_columns, inplace=False)
    writer.write_line("Recursive Feature Elimination Columns:")
    writer.write_line(list(result.columns))
    writer.write_line("")
    return result


def lasso_regression(data: pd.DataFrame, writer: Writer) -> pd.DataFrame:
    """
    Performs lasso regression.

    :return: feature-reduced data frame.
    """
    y = data["RainTomorrow"]
    X = data.drop(columns=["RainTomorrow"], inplace=False)

    estimator = StandardScaler()
    estimator.fit(X)

    lasso = Lasso(alpha=0.01, max_iter=10000)

    selector = SelectFromModel(lasso)
    selector = selector.fit(estimator.transform(X), y)

    dropped_columns = [
        col for col, keep in zip(data.columns, selector.get_support()) if not keep
    ]
    result = data.drop(columns=dropped_columns, inplace=False)
    writer.write_line("Lasso Regression Columns:")
    writer.write_line(list(result.columns))
    writer.write_line("")
    return result


def mutual_information(data: pd.DataFrame, writer: Writer) -> pd.DataFrame:
    """
    Removes columns above a certain mutual information score.

    :return: feature-reduced data frame.
    """
    y = data["RainTomorrow"]
    X = data.drop(columns=["RainTomorrow"], inplace=False)

    selector = SelectKBest(mutual_info_classifier, k=5)
    selector = selector.fit(X, y)

    dropped_columns = [
        col for col, keep in zip(data.columns, selector.get_support()) if not keep
    ]
    result = data.drop(columns=dropped_columns, inplace=False)
    writer.write_line("Mutual Information Columns:")
    writer.write_line(list(result.columns))
    writer.write_line("")
    return result

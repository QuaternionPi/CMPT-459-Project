import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin as SKLearnClassifier
from sklearn.model_selection import KFold, cross_validate
from writer import Writer
from collections import namedtuple

Result = namedtuple("Result", "accuracy precision recall f1 dataset_name classifier")


def test(
    X: pd.DataFrame, y: pd.DataFrame, classifier: SKLearnClassifier, writer: Writer
) -> dict[str, np.ndarray]:
    """
    :param X: DataFrame of input features.
    :param y: DataFrame of target feature.
    :param classifier: Classifier to test.
    :param writer: Where to write the outputs.
    :return: Dict of scores.
    """
    cv = KFold(n_splits=5, random_state=1, shuffle=True)

    scores = cross_validate(
        classifier,
        X,
        y,
        scoring=["accuracy", "precision", "recall", "f1"],
        cv=cv,
        n_jobs=-1,
    )
    return scores


def batch_test(
    runs: list[tuple[str, SKLearnClassifier]],
    datasets: dict[str, pd.DataFrame],
    writer: Writer,
) -> list[Result]:
    """
    Perform batch testing of classifiers over many datasets.

    :param runs: List of dataset names and classifiers.
    :param datasets: Datasets and their names.
    :param writer: Where to write the outputs.
    :return: List of results.
    """
    results: list[Result] = []
    n_test = 0

    test_train: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for dataset_name in datasets.keys():
        dataset = datasets[dataset_name]
        y = dataset["RainTomorrow"]
        X = dataset.drop(columns=["RainTomorrow"], inplace=False)
        test_train[dataset_name] = (X, y)

    for dataset_name, classifier in runs:
        writer.write_line_verbose(f"Running test number {n_test}")
        n_test += 1

        X, y = test_train[dataset_name]
        scores = test(X, y, classifier, writer)

        accuracy = scores["test_accuracy"].mean()
        precision = scores["test_precision"].mean()
        recall = scores["test_recall"].mean()
        f1 = scores["test_f1"].mean()

        result = Result(accuracy, precision, recall, f1, dataset_name, classifier)
        results.append(result)

    return results

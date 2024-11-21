import pandas as pd
from sklearn.base import ClassifierMixin as SKLearnClassifier


def classify(
    train: pd.DataFrame, test: pd.DataFrame, classifier: SKLearnClassifier
) -> float:
    """
    Trains and tests classifier

    :param train: data to train classifier
    :param test: data to test classifier accuracy
    :param classifier: classifier to evaluate
    :return: how well the classifier did
    """
    return 0

import pandas as pd
from sklearn.base import ClassifierMixin as SKLearnClassifier
from sklearn.model_selection import KFold, cross_validate
from writer import Writer
from collections import namedtuple

Result = namedtuple("Result", "accuracy precision recall f1 dataset_name classifier")


def test(data: pd.DataFrame, classifier: SKLearnClassifier, writer: Writer) -> float:
    cv = KFold(n_splits=5, random_state=1, shuffle=True)

    y = data["RainTomorrow"]
    X = data.drop(columns=["RainTomorrow"], inplace=False)

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
    results: list[Result] = []
    n_test = 0

    for dataset_name, classifier in runs:
        writer.write_line_verbose(f"Running test number {n_test}")
        n_test += 1

        dataset = datasets[dataset_name]
        scores = test(dataset, classifier, writer)

        accuracy = scores["test_accuracy"].mean()
        precision = scores["test_precision"].mean()
        recall = scores["test_recall"].mean()
        f1 = scores["test_f1"].mean()

        result = Result(accuracy, precision, recall, f1, dataset_name, classifier)
        results.append(result)

    return results

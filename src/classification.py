import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin as SKLearnClassifier
from sklearn.model_selection import KFold, cross_val_score
from writer import Writer


class Test:
    test_number = 0


def test(data: pd.DataFrame, classifier: SKLearnClassifier, writer: Writer) -> float:
    cv = KFold(n_splits=5, random_state=1, shuffle=True)

    Test.test_number += 1
    writer.write_line_verbose(f"Running test number {Test.test_number}")

    y = data["RainTomorrow"]
    X = data.drop(columns=["RainTomorrow"], inplace=False)

    scores: np.ndarray = cross_val_score(
        classifier,
        X,
        y,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )
    return scores.mean()

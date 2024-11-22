import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin as SKLearnClassifier
from sklearn.model_selection import KFold, cross_val_score
from writer import Writer


def test(data: pd.DataFrame, classifier: SKLearnClassifier, writer: Writer) -> float:
    cv = KFold(n_splits=5, random_state=1, shuffle=True)

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

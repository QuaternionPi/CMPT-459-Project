import pandas as pd
from sklearn.base import ClusterMixin as SKLearnClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from writer import Writer
from exploratory_analysis import Analyzer
import time


class ClusterAnalyzer:
    def __init__(
        self, clusterings: list[SKLearnClustering], data: pd.DataFrame, writer: Writer
    ):
        """
        Runs various clustering algorithms and computes their properties

        :param clusterings: Clustering algorithms to analyze.
        :param data: Data frame to analyze.
        :param writer: Where to write outputs.
        """
        self.clusterings: list[SKLearnClustering] = clusterings
        self.data: pd.DataFrame = data
        self.writer: Writer = writer

    def perform_clusterings(self) -> list[float]:
        """
        Perform clustering on the clustering models.

        :return: List of runtimes.
        """
        runtimes: list[float] = []
        for clustering in self.clusterings:
            start: float = time.perf_counter()
            clustering.fit(self.data, self.data["Rainfall"])
            end: float = time.perf_counter()
            delta: float = end - start
            runtimes.append(delta)
        return runtimes

    def silhouette_score(self) -> list[float]:
        """
        Compute the silhouette scores of clustering models.

        :return: List of silhouette score.
        """
        scores: list[float] = []
        X: pd.DataFrame = self.data
        for clustering in self.clusterings:
            y = clustering.labels_
            score: float = silhouette_score(X, y)
            scores.append(float(score))
        return scores

    def visualize(self, writer: Writer) -> list[Analyzer]:
        """
        Return pairs of visualizers trained on PCA data.

        :return: 2d analyzers trained regular data, drawn on PCA data.
        """
        pca = PCA(n_components=2)
        reduced_data = pd.DataFrame(pca.fit_transform(self.data)).rename(
            columns={0: "0", 1: "1"}
        )

        analyzers: list[Analyzer] = []
        for clustering in self.clusterings:
            y = clustering.fit_predict(self.data)
            X = reduced_data.copy(deep=True)
            X["rains"] = y
            analyzer = Analyzer(X, writer)
            analyzers.append(analyzer)
        print(self.data.to_numpy().shape)
        return analyzers

import pandas as pd
import numpy as np
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
            clustering.fit(self.data)
            end: float = time.perf_counter()
            delta: float = end - start
            runtimes.append(delta)
        return runtimes

    def silhouette_score(self) -> list[float]:
        """
        Compute the silhouette scores of clustering models.

        :return: List of runtimes.
        """
        pass

    def visualize(self) -> list[tuple[Analyzer, Analyzer]]:
        """
        Return pairs of visualizers trained on PCA data.

        :return: Pairs of 2d and 3d analyzers trained on PCA data.
        """
        pass

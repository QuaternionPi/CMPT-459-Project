import pandas as pd
import numpy as np
from sklearn.base import ClusterMixin as SKLearnClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from writer import Writer
from exploratory_analysis import Analyzer


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

    def perform_clusterings() -> list[float]:
        """
        Perform clustering on the clustering models.

        :return: List of runtimes.
        """
        pass

    def silhouette_score() -> list[float]:
        """
        Compute the silhouette scores of clustering models.

        :return: List of runtimes.
        """
        pass

    def visualize() -> list[tuple[Analyzer, Analyzer]]:
        """
        Return pairs of visualizers trained on PCA data.

        :return: Pairs of 2d and 3d analyzers trained on PCA data.
        """
        pass

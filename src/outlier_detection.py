import pandas as pd
from writer import Writer
from exploratory_analysis import Analyzer
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import LocalOutlierFactor


class OutlierDetection:
    def __init__(
        self, lof: float, bandwidth: float, data: pd.DataFrame, writer: Writer
    ):
        """
        Determines if data points are outliers by lof

        :param lof: Threshold to be considered an outlier.
        :param data: Data frame to analyze.
        :param writer: Where to write outputs.
        """
        self.lof: float = lof
        self.bandwidth: float = bandwidth
        self.data: pd.DataFrame = data
        self.writer: Writer = writer

    def find_outliers(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return a data frame with a column identifying outliers.

        :return: Data frame with outliers column.
        """
        pass

    def data_without_outliers(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return a data frame without outliers.

        :return: Data frame without outliers.
        """

    def visualize(self) -> tuple[Analyzer, Analyzer]:
        """
        Return pair of visualizers trained on PCA data showing outliers.

        :return: Pair of 2d and analyzers trained on PCA data showing outliers.
        """
        pass

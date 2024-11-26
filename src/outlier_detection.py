import pandas as pd
from writer import Writer
from exploratory_analysis import Analyzer
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import LocalOutlierFactor


class OutlierDetection:
    def __init__(self, lof: int, bandwidth: float, data: pd.DataFrame, writer: Writer):
        """
        Determines if data points are outliers by lof

        :param lof: Threshold to be considered an outlier by LOF.
        :param bandwidth: Threshold to be considered an outlier by kernel density.
        :param data: Data frame to analyze.
        :param writer: Where to write outputs.
        """
        self.lof: int = lof
        self.bandwidth: float = bandwidth
        self.data: pd.DataFrame = data
        self.writer: Writer = writer
        self.lof_data: pd.DataFrame | None = None
        self.kd_data: pd.DataFrame | None = None

    def find_outliers(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return a data frame with a column identifying outliers.

        :return: Data frame with outliers column.
        """
        self.lof_data: pd.DataFrame = self.data.copy(deep=True)
        self.kd_data: pd.DataFrame = self.data.copy(deep=True)

        self.lof_data["Outlier"] = LocalOutlierFactor(n_neighbors=self.lof).fit_predict(
            self.data
        )
        self.kd_data["Outlier"] = pd.Series(
            KernelDensity(kernel="gaussian", bandwidth=self.bandwidth)
            .fit(self.data)
            .score_samples()
        ).apply(lambda x: -1 if x < 0 else 1)

        return (self.lof_data, self.kd_data)

    def data_without_outliers(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return a data frame without outliers.

        :return: Data frame without outliers.
        """
        pass

    def visualize(self) -> tuple[Analyzer, Analyzer]:
        """
        Return pair of visualizers trained on PCA data showing outliers.

        :return: Pair of 2d and analyzers trained on PCA data showing outliers.
        """
        pass

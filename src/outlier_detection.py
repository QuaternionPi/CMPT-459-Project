import pandas as pd
from sklearn.decomposition import PCA
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

        pca = PCA(n_components=2)
        reduced_data = pd.DataFrame(pca.fit_transform(self.data)).rename(
            columns={0: "0", 1: "1"}
        )

        self.lof_data: pd.DataFrame = reduced_data.copy(deep=True)
        self.kd_data: pd.DataFrame = reduced_data

        self.lof_data["Outlier"] = LocalOutlierFactor(n_neighbors=self.lof).fit_predict(
            self.data
        )
        self.kd_data["Outlier"] = pd.Series(
            KernelDensity(kernel="gaussian", bandwidth=self.bandwidth)
            .fit(self.data)
            .score_samples(self.data)
        ).apply(lambda x: -1 if x > -20.0246 else 1)

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

        lof_analyzer = Analyzer(self.lof_data, self.writer)
        kd_analyzer = Analyzer(self.kd_data, self.writer)
        return lof_analyzer, kd_analyzer

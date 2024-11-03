import pandas as pd
import numpy as np
from writer import Writer


class Analyzer:
    def __init__(self, data: pd.DataFrame, writer: Writer):
        """
        Exploratory Data Analysis tool.

        :param data: data frame to analyze.
        :param writer: where to write outputs.
        """
        self.data: pd.DataFrame = data
        self.writer: Writer = writer

    def box_plot(self, col: str, path: str | None = None) -> None:
        """
        Draws a box plot of a column.

        :param col: Column to be plotted.
        :param path: Path to save drawn image. None for no save.
        """
        pass

    def scatter_plot(self, x_col: str, y_col: str, path: str | None = None) -> None:
        """
        Draws a scatter plot of two columns.

        :param x_col: Column to be plotted on x-axis.
        :param y_col: Column to be plotted on y-axis.
        :param path: Path to save drawn image. None for no save.
        """
        pass

    def scatter_plot_3d(
        self, x_col: str, y_col: str, z_col: str, path: str | None = None
    ) -> None:
        """
        Draws a scatter plot of three columns.

        :param x_col: Column to be plotted on x-axis.
        :param y_col: Column to be plotted on y-axis.
        :param z_col: Column to be plotted on z-axis.
        :param path: Path to save drawn image. None for no save.
        """
        pass

    def histogram(self, x_col: str, y_col: str, path: str | None = None) -> None:
        """
        Draws a histogram of one column against one another.

        :param x_col: Column to be plotted on x-axis.
        :param y_col: Column to be counted on y-axis.
        :param path: Path to save drawn image. None for no save.
        """
        pass

    def histogram_3d(
        self, x_col: str, y_col: str, z_col: str, path: str | None = None
    ) -> None:
        """
        Draws a histogram of one column against one another.

        :param x_col: Column to be plotted on x-axis.
        :param y_col: Column to be plotted on y-axis.
        :param z_col: Column to be counted on z-axis.
        :param path: Path to save drawn image. None for no save.
        """
        pass

    def variance(self, col: str) -> float:
        """
        Returns the variance of a column.

        :param col: The column to be computed.
        :return: The variance of the column.
        """
        pass

    def correlation(self, x_col: str, y_col: str) -> float:
        """
        Returns the correlation of two columns.

        :param x_col: A column to be computed.
        :param y_col: A column to be computed.
        :return: The correlation of the columns.
        """
        pass
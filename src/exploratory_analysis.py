import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
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
        plt.figure()

        x: pd.Series = self.data[col]
        plt.title("Box Plot")
        plt.boxplot(x)
        func_name = "box"
        self._save(path, func_name, col)

        plt.show()

    def scatter_plot(self, x_col: str, y_col: str, path: str | None = None) -> None:
        """
        Draws a scatter plot of two columns.

        :param x_col: Column to be plotted on x-axis.
        :param y_col: Column to be plotted on y-axis.
        :param path: Path to save drawn image. None for no save.
        """
        x: pd.Series = self.data[x_col]
        y: pd.Series = self.data[y_col]
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title("Scatter Plot")
        plt.scatter(x, y)
        func_name = "scatter"
        self._save(path, func_name, x_col, y_col)
        plt.show()

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
        x: pd.Series = self.data[x_col]
        y: pd.Series = self.data[y_col]
        z: pd.Series = self.data[z_col]
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.zlabel(z_col)
        plt.title("Scatter Plot 3D")
        plt.scatter(x, y, z)
        func_name = "scatter_3d"
        self._save(path, func_name, x_col, y_col, z_col)
        plt.show()

    def histogram(self, x_col: str, y_col: str, path: str | None = None) -> None:
        """
        Draws a histogram of one column against one another.

        :param x_col: Column to be plotted on x-axis.
        :param y_col: Column to be counted on y-axis.
        :param path: Path to save drawn image. None for no save.
        """
        x: pd.Series = self.data[x_col]
        y: pd.Series = self.data[y_col]
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title("Histogram")
        plt.hist(x, y)
        func_name = "hist"
        self._save(path, func_name, x_col, y_col)
        plt.show()

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
        x: pd.Series = self.data[x_col]
        y: pd.Series = self.data[y_col]
        z: pd.Series = self.data[z_col]
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.zlabel(z_col)
        plt.title("Histogram 3D")
        plt.hist2d(x, y, z)
        func_name = "hist_3d"
        self._save(path, func_name, x_col, y_col, z_col)
        plt.show()

    def variance(self, col: str) -> float:
        """
        Returns the variance of a column.

        :param col: The column to be computed.
        :return: The variance of the column.
        """
        return self.data[col].var()

    def correlation(self, x_col: str, y_col: str) -> float:
        """
        Returns the correlation of two columns.

        :param x_col: A column to be computed.
        :param y_col: A column to be computed.
        :return: The correlation of the columns.
        """
        x = self.data[x_col]
        y = self.data[y_col]
        return x.corr(y)

    def _save(self, path: str | None, func: str, *cols: str) -> None:
        """
        Save a plot to file if path != null.

        :param path: Path to save location.
        :param func: Abbreviated name of calling function.
        :param cols: Columns used in the function.
        """
        if path == None:
            return
        terms: list[str] = list(cols)
        terms.insert(0, func)
        filename = "-".join(terms).replace(" ", "-")
        dest: str = os.path.join(path, filename)
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(dest)

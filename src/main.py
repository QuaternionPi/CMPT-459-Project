import pandas as pd
import numpy as np
import argparse
from writer import Writer
from exploratory_analysis import Analyzer
from preprocessing import preprocess
from feature_selection import (
    recursive_feature_elimination,
    lasso_regression,
    mutual_information,
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans, OPTICS, DBSCAN
from clustering import ClusterAnalyzer


def parse_args() -> tuple[bool, str]:
    """
    Parses command line arguments.

    :return: tuple of verbose and data
    """
    parser = argparse.ArgumentParser(description="number of clusters to find")
    parser.add_argument(
        "--verbose", "-v", type=bool, help="print verbose", default=False
    )
    parser.add_argument(
        "--data", "-d", type=str, help="path to data", default="weatherAUS.csv"
    )

    args = parser.parse_args()
    return (args.verbose, args.data)


def normalize_column(col: pd.Series) -> pd.Series:
    return (col - col.mean()) / col.std()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Modified from https://stackoverflow.com/questions/26414913/normalize-columns-of-a-dataframe
    return df.iloc[:].apply(normalize_column, axis=0)


def split(data: pd.DataFrame, test_ratio: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_size = 1 / test_ratio
    return train_test_split(data, test_size=test_size)


def eda(data: pd.DataFrame, writer: Writer) -> None:
    """
    Perform exploratory data analysis

    :param data: Data to analyze.
    :param writer: where to write outputs.
    """
    numeric_types = ["int16", "int32", "int64", "float16", "float32", "float64"]

    data = data.select_dtypes(include=numeric_types)
    numeric_columns: list[str] = list(data.columns)
    numeric_columns.remove("RainTomorrow")
    columns_count = len(numeric_columns)
    column_pairs = [
        (numeric_columns[x], numeric_columns[y])
        for x in range(columns_count)
        for y in range(columns_count)
        if x < y
    ]
    column_triples = [
        (numeric_columns[x], numeric_columns[y], numeric_columns[z])
        for x in range(columns_count)
        for y in range(columns_count)
        for z in range(columns_count)
        if x < y and y < z
    ]

    analyzer: Analyzer = Analyzer(data, writer)
    variances: list[tuple[float, str]] = [
        (analyzer.variance(col), col) for col in numeric_columns
    ]
    variances.sort(key=lambda x: x[0])
    correlations: list[tuple[float, str, str]] = [
        (analyzer.correlation(x, y), x, y) for x, y in column_pairs
    ]
    correlations.sort(key=lambda x: x[0])

    for variance in variances[:3]:
        writer.write_line(variance)

    for variance in variances[-3:]:
        writer.write_line(variance)

    for x_col, y_col in column_pairs:
        path = "./eda"
        analyzer.scatter_plot(
            x_col, y_col, ("RainTomorrow", ["Dry", "Rain"]), path=path
        )


def clustering(data: pd.DataFrame, writer: Writer) -> None:
    numeric_types = ["int16", "int32", "int64", "float16", "float32", "float64"]

    numerics = data.select_dtypes(include=numeric_types)
    numerics = normalize(numerics)
    writer.write_line(f"Total entires: {len(numerics.index)}")
    numerics = numerics.drop(numerics.sample(frac=0.975).index)
    writer.write_line(f"Entries kept for clustering: {len(numerics.index)}")

    kmeans = KMeans(n_clusters=2)
    optics = OPTICS()
    dbscan = DBSCAN(eps=4, min_samples=2)

    cluster_analyzer = ClusterAnalyzer([kmeans, optics, dbscan], numerics, writer)
    times: list[float] = cluster_analyzer.perform_clusterings()
    silhouettes: list[float] = cluster_analyzer.silhouette_score()
    visualizers: list[Analyzer] = cluster_analyzer.visualize(writer)

    writer.write_line("Clustering Runtimes:")
    writer.write_line(times)
    writer.write_line("Clustering Silhouette Scores")
    writer.write_line(silhouettes)

    paths = ["./kmeans", "./optics", "./dbscan"]
    for visualizer, path in zip(visualizers, paths):
        visualizer.scatter_plot(
            "0",
            "1",
            ("rains", []),
            path=path,
        )


def feature_selection(data: pd.DataFrame, writer: Writer) -> None:
    numeric_types = ["int16", "int32", "int64", "float16", "float32", "float64"]

    data = data.copy(deep=True)
    data = data.select_dtypes(include=numeric_types)
    rain_tomorrow = data["RainTomorrow"]
    data = data.drop(columns=["RainTomorrow"])
    data = normalize(data)
    data["RainTomorrow"] = rain_tomorrow
    writer.write_line(f"Total entires: {len(data.index)}")
    data = data.drop(data.sample(frac=0.975).index)
    writer.write_line(f"Entries kept for feature selection: {len(data.index)}")

    rfe = recursive_feature_elimination(data, writer)
    lasso = lasso_regression(data, writer)
    mutual = mutual_information(data, writer)


def main() -> None:
    """
    Main function of the program
    """
    (verbose, path) = parse_args()
    writer: Writer = Writer(verbose, None)
    data = preprocess(path, writer)
    # eda(data, writer)
    clustering(data, writer)
    # feature_selection(data, writer)

    test_ratio = 5
    train, test = split(data, test_ratio=test_ratio)


if __name__ == "__main__":
    main()

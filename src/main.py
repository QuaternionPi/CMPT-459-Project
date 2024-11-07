import pandas as pd
import numpy as np
import argparse
from writer import Writer
from exploratory_analysis import Analyzer
from preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


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

    numerics = data.select_dtypes(include=numeric_types)
    numeric_columns: list[str] = numerics.columns
    column_pairs = [(x, y) for x in numeric_columns for y in numeric_columns if x != y]
    column_triples = [
        (x, y, z)
        for x in numeric_columns
        for y in numeric_columns
        for z in numeric_columns
        if x != y and x != z and y != z
    ]

    analyzer: Analyzer = Analyzer(numerics, writer)
    variances: list[tuple[float, str]] = [
        (analyzer.variance(col), col) for col in numeric_columns
    ]
    variances.sort(key=lambda x: x[0])
    correlations: list[tuple[float, str, str]] = [
        (analyzer.correlation(x, y), x, y) for x, y in column_pairs
    ]
    correlations.sort(key=lambda x: x[0])

    for variance in variances[:3]:
        print(variance)

    for variance in variances[-3:]:
        print(variance)


def main() -> None:
    """
    Main function of the program
    """
    (verbose, path) = parse_args()
    writer: Writer = Writer(verbose, None)
    data = preprocess(path, writer)
    eda(data, writer)

    test_ratio = 5
    train, test = split(data, test_ratio=test_ratio)


if __name__ == "__main__":
    main()

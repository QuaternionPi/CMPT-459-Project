import pandas as pd
import numpy as np
import argparse
from writer import Writer
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


def main() -> None:
    """
    Main function of the program
    """
    (verbose, path) = parse_args()
    writer: Writer = Writer(verbose, None)

    data = preprocess(path, writer)
    test_ratio = 5
    train, test = split(data, test_ratio=test_ratio)


if __name__ == "__main__":
    main()

import argparse


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


def main() -> None:
    """
    Main function of the program
    """
    (verbose, data) = parse_args()


if __name__ == "__main_":
    main()

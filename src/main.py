import pandas as pd
import argparse
from writer import Writer
from preprocessing import preprocess
from exploratory_analysis import Analyzer
from outlier_detection import OutlierDetection
from clustering import ClusterAnalyzer
from feature_selection import (
    recursive_feature_elimination,
    lasso_regression,
    mutual_information,
)
from classification import batch_test, Result
from sklearn.cluster import KMeans, OPTICS, DBSCAN
from sklearn.base import ClassifierMixin as SKLearnClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC as SupportVectorClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.exceptions import UndefinedMetricWarning
import warnings


def parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--data-reduction",
        type=float,
        help="amount of data to exclude i.e. =10 keeps only 1 in 10 data points",
        default=10,
    )
    parser.add_argument(
        "--exploratory-data-analysis",
        "-eda",
        type=bool,
        help="run exploratory data analysis?",
        default=True,
    )
    parser.add_argument(
        "--clustering",
        type=bool,
        help="run clustering?",
        default=True,
    )
    parser.add_argument(
        "--outlier-detection",
        "-od",
        type=bool,
        help="run outlier detection?",
        default=True,
    )
    parser.add_argument(
        "--feature-selection",
        "-fs",
        type=bool,
        help="run feature selection?",
        default=True,
    )
    parser.add_argument(
        "--classification",
        type=bool,
        help="run classification?",
        default=True,
    )

    args = parser.parse_args()
    return args


def normalize_column(col: pd.Series) -> pd.Series:
    return (col - col.mean()) / col.std()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Modified from https://stackoverflow.com/questions/26414913/normalize-columns-of-a-dataframe
    return df.iloc[:].apply(normalize_column, axis=0)


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


def outlier_detection(data: pd.DataFrame, writer: Writer) -> None:
    numeric_types = ["int16", "int32", "int64", "float16", "float32", "float64"]

    data = data.select_dtypes(include=numeric_types)

    lof = 20
    bandwidth = 0.8
    outliers = OutlierDetection(lof, bandwidth, data, writer)

    outliers.find_outliers()
    lof_analyzer, kd_analyzer = outliers.visualize()

    lof_analyzer.scatter_plot("0", "1", ("Outlier", ["Out", "In"]), path="./lof")
    kd_analyzer.scatter_plot("0", "1", ("Outlier", ["Out", "In"]), path="./kd")
    print("line")


def clustering(data: pd.DataFrame, writer: Writer) -> None:
    numeric_types = ["int16", "int32", "int64", "float16", "float32", "float64"]

    numerics = data.select_dtypes(include=numeric_types)
    numerics = normalize(numerics)

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


def feature_selection(
    data: pd.DataFrame, writer: Writer
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Selects features based on several methods

    :return: (RFE DataFrame, Lasso DataFrame, Mutual Info DataFrame)
    """
    numeric_types = ["int16", "int32", "int64", "float16", "float32", "float64"]

    data = data.copy(deep=True)
    data = data.select_dtypes(include=numeric_types)
    rain_tomorrow = data["RainTomorrow"]
    data = data.drop(columns=["RainTomorrow"])
    data = normalize(data)
    data["RainTomorrow"] = rain_tomorrow

    rfe = recursive_feature_elimination(data, writer)
    lasso = lasso_regression(data, writer)
    mutual = mutual_information(data, writer)

    return rfe, lasso, mutual


def classification(datasets: dict[str, pd.DataFrame], writer: Writer) -> None:
    k_nearest_neighbours = [KNeighborsClassifier(k + 1) for k in range(0, 30, 2)]
    support_vectors = [
        SupportVectorClassifier(C=C, kernel=kernel)
        for C in range(1, 10)
        for kernel in ["linear", "poly", "rbf", "sigmoid"]
    ]
    random_forests = [RandomForestClassifier(n_trees) for n_trees in range(1, 20, 2)]

    classifier_sets = [k_nearest_neighbours, support_vectors, random_forests]
    classifiers: list[SKLearnClassifier] = [
        classifier
        for classifier_set in classifier_sets
        for classifier in classifier_set
    ]

    runs = [
        (dataset_name, classifier)
        for dataset_name in datasets.keys()
        for classifier in classifiers
    ]
    results: list[Result] = batch_test(runs, datasets, writer)

    results = sorted(results, key=lambda x: x.accuracy)

    ensembles = [
        VotingClassifier(
            [(str(k - i), result.classifier) for i, result in enumerate(results[-k:])]
        )
        for k in range(3, 9, 2)
    ]

    runs = [("all", classifier) for classifier in ensembles]
    results += batch_test(runs, datasets, writer)

    results = sorted(results, key=lambda x: x.accuracy)

    writer.write_line("| Accuracy | Precision | Recall | F1 | Dataset | Classifier |")
    writer.write_line("| -------- | --------- | ------ | -- | ------- | ---------- |")
    for result in results:
        accuracy = round(float(result.accuracy), 4)
        precision = round(float(result.precision), 4)
        recall = round(float(result.recall), 4)
        f1 = round(float(result.f1), 4)

        dataset = str.ljust(result.dataset_name, 6)
        classifier = result.classifier

        line = (
            f"| {accuracy} | {precision} | {recall} | {f1} | {dataset} | {classifier} |"
        )
        writer.write_line(line)


def main() -> None:
    """
    Main function of the program
    """
    args = parse_args()
    verbose = args.verbose
    path = args.data
    data_reduction = args.data_reduction
    run_exploratory_data_analysis = args.exploratory_data_analysis
    run_clustering = args.clustering
    run_outlier_detection = args.outlier_detection
    run_feature_selection = args.feature_selection
    run_classification = args.classification

    if data_reduction < 1:
        data_reduction = 1

    if not verbose:
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    writer: Writer = Writer(verbose, None)
    data = preprocess(path, data_reduction, writer)

    numeric_types = ["int16", "int32", "int64", "float16", "float32", "float64"]
    numerics = data.select_dtypes(include=numeric_types)

    datasets = {"all": numerics}

    if run_exploratory_data_analysis:
        eda(data, writer)
    if run_clustering:
        clustering(data, writer)
    if run_outlier_detection:
        outlier_detection(data, writer)
    if run_feature_selection:
        rfe, lasso, mutual = feature_selection(data, writer)
        datasets["rfe"] = numerics[rfe.columns]
        datasets["lasso"] = numerics[lasso.columns]
        datasets["mutual"] = numerics[mutual.columns]

    if run_classification:
        classification(datasets, writer)

    writer.write_line("---- Done! ----")


if __name__ == "__main__":
    main()

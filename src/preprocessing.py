import pandas as pd
import numpy as np
from writer import Writer
from sklearn import preprocessing


def preprocess(path: str, data_reduction: float, writer: Writer) -> pd.DataFrame:
    """
    Preprocess data.

    :param path: Path to data df (csv).
    :param writer: Where to write outputs.
    :return: Pandas data frame of kept data.
    """
    df: pd.DataFrame = pd.read_csv(path)
    df = df.drop(columns=["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"])

    frac = 1 - 1 / data_reduction
    df = df.drop(df.sample(frac=frac, random_state=1).index)

    int_dtypes = [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
    ]

    # impute remaining NA values with mean. reference https://saturncloud.io/blog/how-to-replace-nan-values-with-the-average-of-columns-in-pandas-dataframe/
    for col in df.columns:
        if df[col].dtype in int_dtypes:  # numerical, impute with mean
            mean = df[col].mean()
            df[col] = df[col].fillna(mean)

        else:  # categorical, impute with mode
            mode = df[col].mode()[0]  # return most frequent mode, the 0th index
            df[col] = df[col].fillna(mode)

    # Label Encoding for Location
    # reference for label encoding: https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/
    label_encoder = preprocessing.LabelEncoder()

    # Encode labels in column 'species'.
    df["Location_enc"] = label_encoder.fit_transform(df["Location"])

    # Change rain tomorrow from strings to numerics
    df["RainTomorrow"] = df["RainTomorrow"].apply(lambda x: 0 if str(x) == "No" else 1)

    # Year and month https://stackoverflow.com/questions/25146121/extracting-just-month-and-year-separately-from-pandas-datetime-column
    df["Year"] = pd.DatetimeIndex(df["Date"]).year
    df["Month"] = pd.DatetimeIndex(df["Date"]).month

    # Remove redundant columns
    df = df.drop(columns=["Location", "Date"])
    return df

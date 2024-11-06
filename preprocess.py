import pandas as pd
import numpy as np

# python preprocess.py

file = pd.read_csv("weatherAUS.csv")

# drop all columns that have >= 30% NA values
file = file.drop(columns=['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'])
print(file)
print(file.dtypes)

int_dtypes = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64]

# impute remaining NA values with mean. reference https://saturncloud.io/blog/how-to-replace-nan-values-with-the-average-of-columns-in-pandas-dataframe/
for col in file.columns:
    if file[col].dtype in int_dtypes: # numerical, impute with mean
        mean = file[col].mean()
        file[col].fillna(mean, inplace=True)

    else:   # categorical, impute with mode
        mean = file[col].mode()[0] # return most frequent mode, the 0th index
        file[col].fillna(mean, inplace=True)

print(file)
file.to_csv("preprocessed_weatherAUS.csv")
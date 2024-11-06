import pandas as pd
import numpy as np
from sklearn import preprocessing

# python preprocess.py

file = pd.read_csv("weatherAUS.csv")

# drop all columns that have >= 30% NA values
file = file.drop(columns=['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'])

int_dtypes = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64]

# impute remaining NA values with mean. reference https://saturncloud.io/blog/how-to-replace-nan-values-with-the-average-of-columns-in-pandas-dataframe/
for col in file.columns:
    if file[col].dtype in int_dtypes: # numerical, impute with mean
        mean = file[col].mean()
        file[col].fillna(mean, inplace=True)

    else:   # categorical, impute with mode
        mean = file[col].mode()[0] # return most frequent mode, the 0th index
        file[col].fillna(mean, inplace=True)

# Label Encoding for Location
# reference for label encoding: https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
encoded_col = label_encoder.fit_transform(file['Location'])

# reference for inserting a column at index: https://stackoverflow.com/questions/18674064/how-do-i-insert-a-column-at-a-specific-column-index-in-pandas
file.insert(loc=1, column='Location_enc', value=encoded_col)

print(file)
file.to_csv("preprocessed_weatherAUS.csv")
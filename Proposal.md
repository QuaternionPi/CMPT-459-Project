# CMPT 459 Project Proposal
David Wiebe | 301470104 | dlw13@sfu.ca\
Rachel Lagasse | 301395285 | rlagasse@sfu.ca
## Data
We've selected climatology [data from Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) titled "Rain in Australia". This real-world dataset contains 10 years of rainfall data collected from various Australian weather stations. It has 23 columns including a mix of numeric and categorical data. It has over 140,000 entries and a total size of 140MB.

The data includes a rich set of features to analyze. Numeric features include humidity, temperature and wind speed. Categorical features include location and wind direction. Most columns are numeric, which may affect our choice of classifiers. Our target variable for classification will be RainTomorrow which predicts if there is rainfall the next day (>= 1mm).

Cleaning needs to be done on the data. This is expected to be a straight forward task, primarily replacing N/A values with medians and modes of numerical and categorical values, respectively. Upon surface inspection there do not appear to be any obvious issues i.e. no data has massive amounts of missing information. 3267 of the target RainTomorrow columns are N/A which we will likely drop to ensure we don't introduce bias into our data.

For data preprocessing ideas, we may extract the Month as another column to see if the month-to-month weather conditions vary. We may group the MinTemp and MaxTemp columns into various temperature regions. For the Location column, we may perform one-hot encoding to convert each weather station location into a unique numerical value the applied data mining techniques can better understand.

## Problem Definition
Out task will be to predict if rain will fall or not (RainTomorrow) based on the data available. The data has a Rainfall (mm) category per day, however it will be one hot encoded with the predicate of rainfall being equal to 0. 

## Justification
Rainfall is important to predict not just to decide weather to grab an umbrella. It affects everything from air traffic to skiing to agriculture. It's essential for weather stations to provide accurate predictions on weather patterns for Australian citizens based on location-specific data. Therefore, the ability to predict future weather patterns based on current data is a valuable feature to predict and consider.

The data set has many attributes important to climate prediction as mentioned above. Based on the discussions present on the Kaggle page it appears that achieving >90% accuracy is possible. While we may not be able to achieve such accuracy it is never the less possible.

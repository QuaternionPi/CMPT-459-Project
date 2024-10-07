# CMPT 459 Project Proposal
David Wiebe | 301470104 | dlw13@sfu.ca\
Rachel Lagasse | Student Number | SFU Email
## Data
We've selected climatology [data from Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) titled "Rain in Australia". It has 23 columns including a mix of numeric and categorical data. It has over 140,000 entries and a total size of 140MB. 

The data includes a rich set of features to analyze. Numeric features include humidity, temperature and wind speed. Categorical features include location and wind direction. Most columns are numeric, which may affect our choice of classifiers.

Cleaning needs to be done on the data. This is expected to be a straight forward task, primarily replacing N/A values with medians and modes. Upon surface inspection there do not appear to be any obvious issues i.e. no data has massive amounts of missing information.

## Problem Definition
Out task will be to predict if rain will fall based on the data available. The data has a rainfall category, however it will be one hot encoded with the predicate of rainfall being equal to 0. 

## Justification
Rainfall is important to predict not just to decide weather to grab an umbrella. It affects everything from air traffic to skiing to agriculture. 

The data set has many attributes important to climate prediction as mentioned above. 
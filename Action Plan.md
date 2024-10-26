# CMPT 459 Project Proposal
David Wiebe | 301470104 | dlw13@sfu.ca\
Rachel Lagasse | 301395285 | rlagasse@sfu.ca

## 1. Dataset Selection
We've selected climatology [data from Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) titled "Rain in Australia". This real-world dataset contains 10 years of rainfall data collected from various Australian weather stations. It has 23 columns including a mix of numeric and categorical data. It has over 140,000 entries and a total size of 140MB. [Read our proposal](./Proposal.md) for more details.

## 2. Data Preprocessing
- Categories with $>30\%$ missing variables remove, otherwise impute
  - Impute by location
- Normalize/standardize numerical features, and perform encoding for categorical variables
  - Encode wind based on direction
  - Half of the compass rose
- We have sufficient examples to justify skipping data augmentation
  - No tiny minority class
- Our data has insufficient dimensionality to require PCA

## 3. Exploratory Data Analysis (EDA)
- Perform basic EDA to understand the structure and distribution of the dataset
  - Create 2, 3d plots (histograms, box plots, heat maps, etc.)
  - Note important correlations
  - Find high variance categories
- Discuss insights and challenges discovered (e.g., class imbalance, highly correlated features)

## 4. Clustering
- Apply at least two different clustering algorithms 
  - We will use K-Means, OPTICS and DBSCAN
  - Maybe more at a later date
- Visualize clustering results using PCA and make 2, 3d scatter plots
- Evaluate the clustering using Silhouette Score
  - Can use code from assignment 2
- Discuss the clustering algorithms and compare their performance

## 5. Outlier Detection
- Perform outlier detection using Local Outlier Factor (LOF)
- Visualize the outliers using plots like in part 4
- Analyze the outliers
  - Are they noise, or do they contain important information? 
  - Decide whether to keep or remove them for further analysis

## 6. Feature Selection
- Apply feature selection techniques to reduce dimensionality:
  - Recursive Feature Elimination (RFE)
  - Lasso Regression
  - Mutual information
- Discuss the selected features and their importance
- Evaluate the model with and without feature selection
  - Compare performance and computational efficiency

## 7. Classification
- Use at least three classification algorithms, such as Random Forest, Support Vector Machines (SVM), k-NN...
- Split the dataset and perform cross validation
  - 90 | 10 split
  - 5-fold cross validation
- Evaluate the models using various metrics, including:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - AUC-ROC
- Visualize results using a confusion matrix and ROC curves

## 8. Hyperparameter Tuning:
- Perform hyperparameter tuning using grid search
  - To be determined based on our use of classifiers
- Discuss the impact of tuning on model performance with before and after comparison

## 9. Conclusion
- Discuss the insights that you learned about the domain of the dataset
- Discuss the lessons learned about data mining methodology
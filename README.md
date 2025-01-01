# CMPT 459 Final Project
David Wiebe | 301470104 | dlw13@sfu.ca\
Rachel Lagasse | 301395285 | rlagasse@sfu.ca

# Running the Program
The program can be called with the following.
```console
cd ./CMPT-459-Project
python src/main.py
```
It will take several minutes to run completely. Be patient.

## Flags
The program has the following flags:\
`--verbose` to toggle verbose mode.\
`--data` to input the data path\
`--data-reduction` to set the data reduction factor. Must be at least 1.\
`--exploratory-data-analysis` to toggle exploratory data analysis.\
`--clustering` to toggle clustering.\
`--outlier-detection` to toggle outlier detection.\
`--feature-selection` to toggle feature selection.\
`--classification` to toggle classification.

## Help
For help run:
```
python src/main.py --help
```

# Report
## 1 Dataset - Rain in Australia
[View Data](./weatherAUS.csv)

We selected climatology [data from Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) titled "Rain in Australia". 
This real-world dataset contains 10 years of rainfall data collected from various Australian weather stations. 
It has 23 features including a mix of numeric and categorical data. 
It has over 140,000 entries and a total size of 140MB.

The data includes a rich set of features to analyze. 
Numeric features include humidity, temperature and wind speed. 
Categorical features include location and wind direction. 
Most features are numeric, which affected our choice of classifiers. 
Our target variable for classification will be the feature `RainTomorrow` which predicts if there is rainfall the next day ($\geq$ 1mm).

## 2 Data Preprocessing
[View Code](./src/preprocessing.py)

The majority of data preprocessing was a straightforward exercise without. 
Our data had several features with $\gt 30\%$ missing data. 
It seemed impractical to impute these features, so we dropped them. 
We then imputed the mean on our numerical features. 
Likewise we imputed the mode for categorical features.
The feature `RainTomorrow` is converted from a string "yes" or "no" to a boolean $1$ or $0$

We extracted the year and month and stored them both as integer features. 
The date was originally stored as a string. 
Thankfully, Pandas has built in tools to do this efficiently.

Our data has considerable size. 
This allows us to skip creating synthetic data. 
We did need to perform data reduction.
All our tests were done at 20 times data reduction.  
Data reduction is controlled by the command line argument `--data-reduction=20`.
The minority class (it rains) is about one quarter of the data, so we decided against augmentation.
At the end of preprocessing our data had $20$ numerical features - not enough to require PCA.

Exploratory Data Analysis requires non-normalized data. 
Normalization is held off until it is needed in later stages of the project.


## 3 Exploratory Data Analysis
[View Code](./src/exploratory_analysis.py)

Exploratory data analysis helped us understand the data we are working with. 
We started by looking at variances and correlations. After that we created plots. 
Finally we highlight some of our findings here

Variances differ greatly between the features.
The lowest variance feature is `Year`. 
This makes sense as it has only a small handful of values.
The highest variance feature is `Humidity at 3 pm`. 
Note this is far more varied than `Humidity at 9 am` for unknown reasons.

Correlations were a mystery to us at first. 
The feature pairs with the lowest recorded variance weren't the features who appeared to have the lowest variance to our eyes.
This is because of the imputation of missing data.
Imputations place a lot of values on the mean of that particular feature.
Pairs with many imputations on either features get high variance.
Those without get incorrectly seen as lower in comparison.
As such correlations were not recorded.

Many data were collected and are shown below. 
We highlight the top and bottom three variances. 
We've also selected several plots.

### 3.1 Top Three Lowest Variances
| Feature  | Variance |
| -------- | -------- |
| Wind Direction 9 am East West   | 0.779 |
| Wind Gust Direction North South | 0.796 |
| Wind Direction 9 am North South | 0.866 |

### 3.2 Bottom Three Highest Variances
| Feature           | Variance |
| ----------------- | -------- |
| Location Encoding | 202.733  |
| Humidity at 9 am  | 356.825  |
| Humidity at 3 pm  | 421.468  |

### 3.3 Interesting Plots
<figure>
    <img src='./eda/scatter-MaxTemp-Temp3pm.png' alt='Scatter plot of Max Temp vs Temp at 3 pm' />
    <figcaption>Scatter plot of Max Temp vs Temp at 3 pm</figcaption>
</figure>
<br>
<figure>
    <img src='./eda/scatter-MaxTemp-Rainfall.png' alt='Scatter plot of Max Temp vs Rainfall' />
    <figcaption>Scatter plot of Max Temp vs Rainfall</figcaption>
</figure>
<br>
<figure>
    <img src='./eda/scatter-MaxTemp-Pressure9am.png' alt='Scatter plot of Max Temp vs Pressure at 9 am' />
    <figcaption>Scatter plot of Max Temp vs Pressure at 9 am</figcaption>
</figure>
<br>
<figure>
    <img src='./eda/scatter-Pressure9am-Pressure3pm.png' alt='Scatter plot of Pressure at 9 am vs Pressure at 3 pm' />
    <figcaption>Scatter plot of Pressure at 9 am vs Pressure at 3 pm</figcaption>
</figure>
<br>
<figure>
    <img src='./eda/scatter-Pressure9am-Location_enc.png' alt='Scatter plot of Location Encoding vs Pressure at 9 am' />
    <figcaption>Scatter plot of Location Encoding vs Pressure at 9 am</figcaption>
</figure>
<br>
<figure>
    <img src='./eda/scatter-WindGustSpeed-Pressure9am.png' alt='Scatter plot of Wind Gust Speed vs Pressure at 9 am' />
    <figcaption>Scatter plot of Wind Gust Speed vs Pressure at 9 am</figcaption>
</figure>
<br>
<figure>
    <img src='./eda/scatter-WindGustSpeed-Location_enc.png' alt='Scatter plot of Location Encoding vs Wind Gust Speed' />
    <figcaption>Scatter plot of Location Encoding vs Wind Gust Speed</figcaption>
</figure>

## 4 Clustering
[View Code](./src/clustering.py)

For clustering we chose to use KMeans, OPTICS, and DBSCAN. 
KMeans was chosen for its reliability. 
DBSCAN was chosen because we believed it would not get "distracted" by the high variances present in the data.
OPTICS was chosen to compliment DBSCAN.
We then visualized the clusterings.

We tuned hyperparameters to maximize Silhouette score. 
KMeans failed to produce any good results.
DBSCAN was the most fickle, but it gave the best result. 
DBSCAN assigned almost every point to a single cluster.
OPTICS despite our tuning assigned almost every point as an outlier.

### 4.1 Clustering Performance
| Metric           | KMeans | OPTICS | DBSCAN |
| ---------------- | ------ | ------ | ------ |
| Runtime          | 0.187  | 11.191 | 0.097  |
| Silhouette Score | 0.172  | -0.392 | 0.581  |

### 4.2 Plots
<figure>
    <img src='./kmeans/scatter-0-1.png' alt='KMeans Clustering' />
    <figcaption>KMeans Clustering</figcaption>
</figure>
<br>
<figure>
    <img src='./optics/scatter-0-1.png' alt='OPTICS Clustering' />
    <figcaption>OPTICS Clustering</figcaption>
</figure>
<br>
<figure>
    <img src='./dbscan/scatter-0-1.png' alt='DBSCAN Clustering' />
    <figcaption>DBSCAN Clustering</figcaption>
</figure>

## 5 Outlier Detection
[View Code](./src/outlier_detection.py)

Our two outlier detection methods are Local Outlier Factor and Kernel Density.
Local Outlier Factor was chosen as it is quick and simple to implement.
Kernel Density was chosen as we expected it to find patterns of odd behavior.

They both failed to find any real number of outliers ($\lt 1\%$). 
This is likely because we have no real outliers.
Our data is professionally collected.
There are no extreme values discovered by EDA.
Furthermore upon inspecting what values were classified as outliers by Local Outlier Factor are mostly days of high rain.
This is a good indication of the quality of our data.

### 5.1 Plots
<figure>
    <img src='./lof/scatter-0-1.png' alt='Local Outlier Factor Outliers' />
    <figcaption>Local Outlier Factor Outliers</figcaption>
</figure>
<br>
<figure>
    <img src='./kd/scatter-0-1.png' alt='Kernel Density Outliers' />
    <figcaption>Kernel Density Outliers</figcaption>
</figure>

## 6 Feature Selection
[View Code](./src/feature_selection.py)

We ran three different feature selection algorithms. 
They're all chosen because we didn't know what to pick.
Thankfully implementing all three gave us a good spread.
We selected Recursive Feature Elimination, Lasso Regression, and Mutual Information.
We ran these clustering algorithms and tuned them to produce 5 features each for comparison.
You can see the results in the table below. 
Our discussion of the results is after the table.

### 6.1 Selected Features
| Features Kept    | Recursive Feature | Lasso Regression | Mutual Information | Fraction |
| --------------- | ----------------- | ---------------- | ------------------ | -------- |
| Rainfall        | X                 | X                | X                  | $3/3$    |
| Wind Gust Speed | X                 | X                |                    | $2/3$    |
| Wind Speed 3 pm |                   | X                |                    | $1/3$    |
| Humidity 9 am   |                   |                  | X                  | $1/3$    |
| Humidity 3 pm   | X                 | X                | X                  | $3/3$    |
| Pressure 9 am   | X                 |                  | X                  | $2/3$    |
| Pressure 3 pm   | X                 | X                | X                  | $3/3$    |

### 6.2 Discussion of Selected Features
Three features are selected by all of the three models. 
These features are Rainfall, Humidity at 3 pm, and Pressure at 3 pm.
- Rainfall today is an obvious predictor of rainfall tomorrow
- Humidity and pressure both make sense for weather prediction

Humidity and pressure were both more commonly referenced at 3 pm vs 9 am.
This implies weather conditions later in the day have stronger predictive powers than those later in the day.
Of course this is obvious to us humans, but its cool the feature selection figured it out.

## 7 Classification
[View Code](./src/classification.py)

We performed our classification with 4 kinds of classifiers, over 4 metrics, and on 4 subsets of our data.
Analysis was done with 5 fold cross validation.
We originally planned to do 10 but for performance we cut it down.

1. We chose the following classifiers
   - Support Vector Machines
   - K Nearest Neighbours
   - Random Forest Classifiers
   - Ensemble Voting Classifiers of the best 3, 5, and 7 base classifiers
2. For metrics we picked 4 and sorted by accuracy
   - Accuracy
   - Precision
   - Recall
   - F1 score
3. Tests were run over 4 different datasets
   - All of the data 
   - Features picked by RFE
   - Features picked by Mutual Information
   - Features picked by Lasso

## 8 Hyperparameter Tuning
[View Code](./src/main.py)

### 8.1 Support Vector Machines
Support Vector Machines had two tuned hyper parameters. 
`C` The regularization strength and `kernel` the kernel function used.
The best value was `C=6`, these has better accuracy than other SVMs.
Why is likely just a function of the data set.
The worst kernel was `kernel = 'sigmoid'`. 
This is because our data is linear. 
The best was `kernel='linear'`.
Likewise because our data is linear.

### 8.2 K Nearest Neighbours
K Nearest Neighbours have one tuned hyper parameter. 
`n_neighbours` is how many nearest neighbours should this classifier check? 
The best 5 base classifiers were all KNNs with `n_neighbours > 15`.
This is quite impressive given the data set and how high the variances are.

### 8.3 Random Forest Classifiers
Random Forests have only one tuned hyper parameter.
`n_estimators` The number of decision trees to be created.
Higher numbers of decision trees do better overall.
KNNs with `n_estimators=17, 19` were in the top 15 of all classifiers.
It may in fact be possible to achieve higher accuracies with an increase in the number of decision trees.

### 8.4 Ensemble Classifiers
We tested Ensemble Voting Classifiers of sizes 3, 5, and 7.
The used the best k out of all the base classifiers.
Size 3 did the best of all classifiers.
Next was the size 5 EVC.
Size 7 interestingly did worse than the best 3 base classifiers.

## 9 Results and Conclusion
### 9.1 Good Results
The best performing classifiers were the Ensemble Voting Classifiers of size 3 and 5.
This goes to show combining classifiers was a good idea. 
Interestingly the top 7 ensemble voting classifier of 7 does worse than some of the base classifiers.
This is likely caused by under-representation of the minority class.
The data is close enough to balanced for the base classifiers. 
Unfortunately combining these classifiers boosts the under-representation enough to prevent accurate classification.

The 5 best base classifiers were all KNNs. 
This is a little unexpected but it is okay.

### 9.2 Bad Results
Support Vector machines did terribly. 
They are unable to handle the complex relationships between features.

Random forests with `n_neighbours = 1` were also quite bad.
This is because they're essentially just over-fit decision trees.
Their performance does increase quite quickly with more trees.

Mutual Information tended to produce the worst classification results.
I don't know why this would be.
My best guess is that wind speed is really important in a non-obvious way and Mutual Information removes all wind speed features.

### 9.3 Best Scores
| Accuracy | Precision | Recall | F1     |
| -------- | --------- | ------ | ------ |
| 0.8406   | 0.7657    | 0.4098 | 0.5329 |

### Challenges
The biggest challenge with this project has been balancing performance with results.
We have a lot of data, which is good however, long runtimes we can't iterate as quickly.
Currently the runtime sits at several minutes for the whole project.

### Takeaways
We learned the hard way how messy and difficult to work with weather data is. 
EDA revealed many interesting relationships, like the triangle formed by `Max Temp` and `Pressure at 9 am`.
Clustering showed our data is just a single blob.
Outlier detection showed how clean/usable our data is.
Feature selection taught us there are no overwhelmingly strong predictors of rain. 
Clustering and hyperparameter tuning provided many good classifiers, showing that there are no magic settings out there.

### Next Steps
Next steps will be to try out more classifiers and incorporated more data.
Probabilistic classifiers like Bayes Nets may do well in these situations. 
Similarly having larger Random Forests may prove fruitful.
We planned, but were unable to, encode wind directions as pairs of (N/S, E/W) values. 
This may be important information to our classifiers.
It may also be fruitful to include the previous day's weather conditions.

### 10 References
Our Data Source - https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package

SKLearn Python Library - https://scikit-learn.org/stable/

Pandas Python Library - https://pandas.pydata.org/

Matplotlib Python Library - https://matplotlib.org/

## Appendix A - Classification Results
| Accuracy | Precision | Recall | F1 | Dataset | Classifier |
| -------- | --------- | ------ | -- | ------- | ---------- |
| 0.7575 | 0.4555 | 0.4591 | 0.4564 | mutual | RandomForestClassifier(n_estimators=1) |        
| 0.7579 | 0.4587 | 0.4775 | 0.4669 | rfe    | RandomForestClassifier(n_estimators=1) |        
| 0.7587 | 0.4574 | 0.4591 | 0.4577 | all    | RandomForestClassifier(n_estimators=1) |        
| 0.7624 | 0.4649 | 0.4559 | 0.4594 | lasso  | RandomForestClassifier(n_estimators=1) |        
| 0.7683 | 0.4794 | 0.4715 | 0.4743 | rfe    | KNeighborsClassifier(n_neighbors=1) |
| 0.7696 | 0.4817 | 0.4754 | 0.4779 | mutual | KNeighborsClassifier(n_neighbors=1) |
| 0.7751 | 0.4937 | 0.4811 | 0.4867 | lasso  | KNeighborsClassifier(n_neighbors=1) |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | all    | SVC(C=1, kernel='poly') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | all    | SVC(C=1) |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | all    | SVC(C=1, kernel='sigmoid') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | all    | SVC(C=2) |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | all    | SVC(C=2, kernel='sigmoid') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | all    | SVC(C=3, kernel='sigmoid') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | all    | SVC(C=4, kernel='sigmoid') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | all    | SVC(C=5, kernel='sigmoid') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | all    | SVC(C=6, kernel='sigmoid') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | all    | SVC(C=7, kernel='sigmoid') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | all    | SVC(C=8, kernel='sigmoid') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | all    | SVC(C=9, kernel='sigmoid') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | rfe    | SVC(C=1, kernel='sigmoid') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | rfe    | SVC(C=2, kernel='sigmoid') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | rfe    | SVC(C=3, kernel='sigmoid') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | rfe    | SVC(C=4, kernel='sigmoid') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | rfe    | SVC(C=5, kernel='sigmoid') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | rfe    | SVC(C=6, kernel='sigmoid') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | rfe    | SVC(C=7, kernel='sigmoid') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | lasso  | SVC(C=1, kernel='sigmoid') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | lasso  | SVC(C=2, kernel='sigmoid') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | mutual | SVC(C=1, kernel='sigmoid') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | mutual | SVC(C=2, kernel='sigmoid') |
| 0.7775 | 0.0000 | 0.0000 | 0.0000 | mutual | SVC(C=3, kernel='sigmoid') |
| 0.7777 | 0.2000 | 0.0006 | 0.0013 | all    | SVC(C=3) |
| 0.7777 | 0.2000 | 0.0006 | 0.0013 | lasso  | SVC(C=3, kernel='sigmoid') |
| 0.7777 | 0.2000 | 0.0006 | 0.0013 | mutual | SVC(C=4, kernel='sigmoid') |
| 0.7777 | 0.2000 | 0.0006 | 0.0013 | mutual | SVC(C=5, kernel='sigmoid') |
| 0.7778 | 0.4000 | 0.0013 | 0.0026 | rfe    | SVC(C=8, kernel='sigmoid') |
| 0.7779 | 0.5333 | 0.0031 | 0.0062 | rfe    | SVC(C=9, kernel='sigmoid') |
| 0.7779 | 0.4000 | 0.0019 | 0.0038 | all    | SVC(C=2, kernel='poly') |
| 0.7779 | 0.4000 | 0.0019 | 0.0038 | all    | SVC(C=4) |
| 0.7779 | 0.4000 | 0.0019 | 0.0038 | lasso  | SVC(C=4, kernel='sigmoid') |
| 0.7781 | 0.6000 | 0.0026 | 0.0052 | lasso  | SVC(C=5, kernel='sigmoid') |
| 0.7782 | 0.6000 | 0.0033 | 0.0065 | rfe    | SVC(C=1) |
| 0.7782 | 0.8000 | 0.0032 | 0.0064 | mutual | SVC(C=6, kernel='sigmoid') |
| 0.7785 | 0.6000 | 0.0045 | 0.0090 | mutual | SVC(C=1) |
| 0.7788 | 0.8000 | 0.0058 | 0.0116 | all    | SVC(C=5) |
| 0.7789 | 1.0000 | 0.0063 | 0.0125 | mutual | SVC(C=7, kernel='sigmoid') |
| 0.7792 | 1.0000 | 0.0077 | 0.0152 | lasso  | SVC(C=6, kernel='sigmoid') |
| 0.7793 | 1.0000 | 0.0084 | 0.0166 | lasso  | SVC(C=1) |
| 0.7800 | 0.9200 | 0.0126 | 0.0248 | mutual | SVC(C=8, kernel='sigmoid') |
| 0.7803 | 1.0000 | 0.0129 | 0.0253 | all    | SVC(C=6) |
| 0.7810 | 0.8484 | 0.0191 | 0.0372 | mutual | SVC(C=9, kernel='sigmoid') |
| 0.7811 | 1.0000 | 0.0168 | 0.0328 | all    | SVC(C=3, kernel='poly') |
| 0.7817 | 0.9378 | 0.0204 | 0.0398 | lasso  | SVC(C=7, kernel='sigmoid') |
| 0.7825 | 0.9750 | 0.0239 | 0.0462 | all    | SVC(C=7) |
| 0.7834 | 0.9155 | 0.0307 | 0.0590 | mutual | SVC(C=2) |
| 0.7841 | 0.9067 | 0.0338 | 0.0648 | lasso  | SVC(C=8, kernel='sigmoid') |
| 0.7850 | 0.5174 | 0.4961 | 0.5057 | all    | KNeighborsClassifier(n_neighbors=1) |
| 0.7851 | 0.9411 | 0.0386 | 0.0730 | all    | SVC(C=8) |
| 0.7866 | 0.8963 | 0.0476 | 0.0897 | lasso  | SVC(C=9, kernel='sigmoid') |
| 0.7867 | 0.9388 | 0.0468 | 0.0879 | rfe    | SVC(C=2) |
| 0.7877 | 0.9174 | 0.0518 | 0.0968 | all    | SVC(C=4, kernel='poly') |
| 0.7894 | 0.9254 | 0.0600 | 0.1112 | all    | SVC(C=9) |
| 0.7929 | 0.8703 | 0.0851 | 0.1515 | mutual | SVC(C=3) |
| 0.7944 | 0.9042 | 0.0897 | 0.1602 | lasso  | SVC(C=2) |
| 0.7944 | 0.5468 | 0.4582 | 0.4971 | lasso  | RandomForestClassifier(n_estimators=3) |        
| 0.7954 | 0.5476 | 0.4652 | 0.5021 | mutual | RandomForestClassifier(n_estimators=3) |        
| 0.7983 | 0.9242 | 0.1065 | 0.1879 | all    | SVC(C=5, kernel='poly') |
| 0.7999 | 0.5600 | 0.4708 | 0.5108 | rfe    | RandomForestClassifier(n_estimators=3) |
| 0.8012 | 0.9091 | 0.1238 | 0.2142 | lasso  | SVC(C=1, kernel='poly') |
| 0.8017 | 0.8945 | 0.1298 | 0.2211 | mutual | SVC(C=1, kernel='poly') |
| 0.8032 | 0.5692 | 0.4732 | 0.5163 | all    | RandomForestClassifier(n_estimators=3) |        
| 0.8032 | 0.5745 | 0.4478 | 0.5022 | rfe    | RandomForestClassifier(n_estimators=5) |        
| 0.8034 | 0.8906 | 0.1378 | 0.2355 | rfe    | SVC(C=3) |
| 0.8035 | 0.5746 | 0.4512 | 0.5043 | rfe    | KNeighborsClassifier(n_neighbors=3) |
| 0.8042 | 0.5772 | 0.4532 | 0.5067 | mutual | KNeighborsClassifier(n_neighbors=3) |
| 0.8046 | 0.8732 | 0.1474 | 0.2487 | all    | SVC(C=6, kernel='poly') |
| 0.8054 | 0.5799 | 0.4696 | 0.5175 | lasso  | RandomForestClassifier(n_estimators=5) |        
| 0.8057 | 0.8729 | 0.1574 | 0.2599 | mutual | SVC(C=4) |
| 0.8063 | 0.5821 | 0.4670 | 0.5165 | mutual | RandomForestClassifier(n_estimators=5) |
| 0.8090 | 0.5904 | 0.4635 | 0.5183 | rfe    | RandomForestClassifier(n_estimators=7) |
| 0.8097 | 0.8373 | 0.1854 | 0.3002 | all    | SVC(C=7, kernel='poly') |
| 0.8109 | 0.6000 | 0.4520 | 0.5144 | lasso  | KNeighborsClassifier(n_neighbors=3) |
| 0.8119 | 0.8489 | 0.1935 | 0.3114 | lasso  | SVC(C=3) |
| 0.8130 | 0.6030 | 0.4651 | 0.5244 | mutual | RandomForestClassifier(n_estimators=7) |
| 0.8142 | 0.8300 | 0.2121 | 0.3350 | rfe    | SVC(C=4) |
| 0.8144 | 0.6095 | 0.4710 | 0.5294 | rfe    | RandomForestClassifier(n_estimators=9) |
| 0.8147 | 0.8517 | 0.2097 | 0.3317 | mutual | SVC(C=5) |
| 0.8149 | 0.6098 | 0.4698 | 0.5297 | all    | RandomForestClassifier(n_estimators=5) |        
| 0.8160 | 0.6230 | 0.4452 | 0.5176 | mutual | KNeighborsClassifier() |
| 0.8162 | 0.6223 | 0.4408 | 0.5152 | rfe    | KNeighborsClassifier() |
| 0.8173 | 0.8343 | 0.2276 | 0.3549 | all    | SVC(C=8, kernel='poly') |
| 0.8177 | 0.6212 | 0.4676 | 0.5322 | lasso  | RandomForestClassifier(n_estimators=7) |        
| 0.8189 | 0.8302 | 0.2418 | 0.3705 | mutual | SVC(C=6) |
| 0.8191 | 0.6289 | 0.4616 | 0.5308 | rfe    | RandomForestClassifier(n_estimators=13) |       
| 0.8195 | 0.6260 | 0.4694 | 0.5355 | lasso  | RandomForestClassifier(n_estimators=9) |
| 0.8195 | 0.6307 | 0.4592 | 0.5301 | rfe    | RandomForestClassifier(n_estimators=17) |       
| 0.8206 | 0.6255 | 0.4822 | 0.5437 | all    | KNeighborsClassifier(n_neighbors=3) |
| 0.8214 | 0.8308 | 0.2521 | 0.3843 | all    | SVC(C=9, kernel='poly') |
| 0.8221 | 0.6369 | 0.4671 | 0.5380 | rfe    | RandomForestClassifier(n_estimators=15) |        
| 0.8222 | 0.8194 | 0.2635 | 0.3955 | mutual | SVC(C=7) |
| 0.8225 | 0.6363 | 0.4761 | 0.5433 | rfe    | RandomForestClassifier(n_estimators=11) |       
| 0.8226 | 0.8259 | 0.2616 | 0.3946 | rfe    | SVC(C=5) |
| 0.8228 | 0.6411 | 0.4660 | 0.5380 | rfe    | RandomForestClassifier(n_estimators=19) |
| 0.8232 | 0.8204 | 0.2675 | 0.4010 | rfe    | SVC(C=1, kernel='poly') |
| 0.8235 | 0.6423 | 0.4689 | 0.5412 | mutual | RandomForestClassifier(n_estimators=9) |        
| 0.8237 | 0.8024 | 0.2815 | 0.4136 | mutual | SVC(C=8) |
| 0.8239 | 0.8053 | 0.2809 | 0.4136 | mutual | SVC(C=2, kernel='poly') |
| 0.8240 | 0.8396 | 0.2622 | 0.3974 | lasso  | SVC(C=4) |
| 0.8248 | 0.6482 | 0.4648 | 0.5407 | lasso  | RandomForestClassifier(n_estimators=11) |       
| 0.8248 | 0.6474 | 0.4695 | 0.5432 | all    | RandomForestClassifier(n_estimators=7) |        
| 0.8248 | 0.6636 | 0.4352 | 0.5243 | mutual | KNeighborsClassifier(n_neighbors=7) |
| 0.8252 | 0.7912 | 0.2957 | 0.4282 | mutual | SVC(C=9) |
| 0.8257 | 0.6650 | 0.4366 | 0.5262 | rfe    | KNeighborsClassifier(n_neighbors=7) |
| 0.8259 | 0.6565 | 0.4589 | 0.5392 | lasso  | RandomForestClassifier(n_estimators=15) |       
| 0.8262 | 0.6572 | 0.4642 | 0.5421 | lasso  | RandomForestClassifier(n_estimators=13) |       
| 0.8263 | 0.8163 | 0.2880 | 0.4234 | rfe    | SVC(C=6) |
| 0.8263 | 0.6511 | 0.4750 | 0.5483 | mutual | RandomForestClassifier(n_estimators=11) |        
| 0.8266 | 0.6569 | 0.4623 | 0.5419 | mutual | RandomForestClassifier(n_estimators=17) |       
| 0.8272 | 0.8126 | 0.2947 | 0.4303 | lasso  | SVC(C=5) |
| 0.8274 | 0.6584 | 0.4663 | 0.5453 | mutual | RandomForestClassifier(n_estimators=13) |       
| 0.8276 | 0.7784 | 0.3190 | 0.4504 | mutual | SVC(C=3, kernel='poly') |
| 0.8276 | 0.6665 | 0.4501 | 0.5364 | lasso  | KNeighborsClassifier() |
| 0.8277 | 0.8212 | 0.2930 | 0.4297 | lasso  | SVC(C=2, kernel='poly') |
| 0.8279 | 0.6633 | 0.4592 | 0.5420 | mutual | RandomForestClassifier(n_estimators=19) |        
| 0.8280 | 0.6617 | 0.4655 | 0.5457 | lasso  | RandomForestClassifier(n_estimators=17) |        
| 0.8281 | 0.8090 | 0.3032 | 0.4386 | rfe    | SVC(C=7) |
| 0.8283 | 0.6644 | 0.4617 | 0.5441 | all    | RandomForestClassifier(n_estimators=11) |       
| 0.8285 | 0.6605 | 0.4703 | 0.5488 | mutual | RandomForestClassifier(n_estimators=15) |       
| 0.8295 | 0.6860 | 0.4312 | 0.5287 | mutual | KNeighborsClassifier(n_neighbors=9) |
| 0.8296 | 0.6633 | 0.4753 | 0.5533 | all    | RandomForestClassifier(n_estimators=9) |
| 0.8299 | 0.6713 | 0.4651 | 0.5480 | lasso  | RandomForestClassifier(n_estimators=19) |        
| 0.8301 | 0.8041 | 0.3165 | 0.4522 | rfe    | SVC(C=8) |
| 0.8302 | 0.7951 | 0.3234 | 0.4578 | rfe    | SVC(C=9) |
| 0.8306 | 0.8042 | 0.3191 | 0.4551 | lasso  | SVC(C=6) |
| 0.8306 | 0.6883 | 0.4382 | 0.5342 | rfe    | KNeighborsClassifier(n_neighbors=9) |
| 0.8309 | 0.7788 | 0.3394 | 0.4708 | mutual | SVC(C=4, kernel='poly') |
| 0.8310 | 0.7803 | 0.3410 | 0.4720 | rfe    | SVC(C=2, kernel='poly') |
| 0.8310 | 0.6832 | 0.4506 | 0.5417 | lasso  | KNeighborsClassifier(n_neighbors=7) |
| 0.8316 | 0.7176 | 0.4062 | 0.5168 | mutual | SVC(C=3, kernel='linear') |
| 0.8317 | 0.7186 | 0.4072 | 0.5174 | mutual | SVC(C=5, kernel='linear') |
| 0.8318 | 0.7176 | 0.4074 | 0.5179 | mutual | SVC(C=4, kernel='linear') |
| 0.8318 | 0.6987 | 0.4328 | 0.5331 | rfe    | KNeighborsClassifier(n_neighbors=11) |
| 0.8320 | 0.7778 | 0.3466 | 0.4778 | mutual | SVC(C=5, kernel='poly') |
| 0.8320 | 0.7176 | 0.4078 | 0.5184 | mutual | SVC(C=6, kernel='linear') |
| 0.8321 | 0.7171 | 0.4107 | 0.5204 | mutual | SVC(C=7, kernel='linear') |
| 0.8323 | 0.7743 | 0.3509 | 0.4813 | mutual | SVC(C=6, kernel='poly') |
| 0.8323 | 0.7687 | 0.3544 | 0.4839 | mutual | SVC(C=7, kernel='poly') |
| 0.8323 | 0.7258 | 0.3976 | 0.5125 | mutual | SVC(C=1, kernel='linear') |
| 0.8324 | 0.6916 | 0.4473 | 0.5421 | all    | RandomForestClassifier(n_estimators=15) |       
| 0.8325 | 0.7680 | 0.3570 | 0.4860 | mutual | SVC(C=8, kernel='poly') |
| 0.8325 | 0.7146 | 0.4177 | 0.5251 | mutual | SVC(C=8, kernel='linear') |
| 0.8327 | 0.7194 | 0.4107 | 0.5211 | mutual | SVC(C=9, kernel='linear') |
| 0.8327 | 0.7985 | 0.3349 | 0.4702 | lasso  | SVC(C=7) |
| 0.8328 | 0.7055 | 0.4307 | 0.5334 | rfe    | KNeighborsClassifier(n_neighbors=13) |
| 0.8328 | 0.6835 | 0.4636 | 0.5514 | all    | KNeighborsClassifier() |
| 0.8329 | 0.7983 | 0.3374 | 0.4725 | lasso  | SVC(C=3, kernel='poly') |
| 0.8329 | 0.7662 | 0.3604 | 0.4891 | mutual | SVC(C=9, kernel='poly') |
| 0.8329 | 0.7218 | 0.4087 | 0.5206 | mutual | SVC(C=2, kernel='linear') |
| 0.8335 | 0.7063 | 0.4315 | 0.5348 | mutual | KNeighborsClassifier(n_neighbors=11) |
| 0.8335 | 0.7094 | 0.4280 | 0.5329 | rfe    | KNeighborsClassifier(n_neighbors=19) |
| 0.8335 | 0.7115 | 0.4250 | 0.5312 | rfe    | KNeighborsClassifier(n_neighbors=21) |
| 0.8338 | 0.7020 | 0.4411 | 0.5406 | all    | KNeighborsClassifier(n_neighbors=7) |
| 0.8339 | 0.7531 | 0.3802 | 0.5036 | lasso  | SVC(C=8, kernel='poly') |
| 0.8339 | 0.7571 | 0.3750 | 0.5002 | lasso  | SVC(C=7, kernel='poly') |
| 0.8340 | 0.7650 | 0.3689 | 0.4963 | lasso  | SVC(C=6, kernel='poly') |
| 0.8342 | 0.7137 | 0.4258 | 0.5326 | mutual | KNeighborsClassifier(n_neighbors=13) |
| 0.8342 | 0.7702 | 0.3683 | 0.4960 | rfe    | SVC(C=3, kernel='poly') |
| 0.8342 | 0.7224 | 0.4145 | 0.5259 | mutual | KNeighborsClassifier(n_neighbors=19) |
| 0.8342 | 0.7985 | 0.3433 | 0.4786 | lasso  | SVC(C=8) |
| 0.8343 | 0.7613 | 0.3761 | 0.5014 | rfe    | SVC(C=4, kernel='poly') |
| 0.8343 | 0.7034 | 0.4442 | 0.5433 | lasso  | KNeighborsClassifier(n_neighbors=9) |
| 0.8345 | 0.6952 | 0.4559 | 0.5503 | all    | RandomForestClassifier(n_estimators=13) |       
| 0.8345 | 0.7898 | 0.3503 | 0.4842 | lasso  | SVC(C=9) |
| 0.8345 | 0.7239 | 0.4158 | 0.5272 | mutual | KNeighborsClassifier(n_neighbors=21) |
| 0.8346 | 0.7517 | 0.3856 | 0.5083 | lasso  | SVC(C=9, kernel='poly') |
| 0.8346 | 0.7167 | 0.4256 | 0.5332 | rfe    | KNeighborsClassifier(n_neighbors=23) |
| 0.8347 | 0.7887 | 0.3535 | 0.4868 | lasso  | SVC(C=4, kernel='poly') |
| 0.8347 | 0.7739 | 0.3660 | 0.4954 | lasso  | SVC(C=5, kernel='poly') |
| 0.8349 | 0.7339 | 0.4048 | 0.5211 | mutual | KNeighborsClassifier(n_neighbors=25) |
| 0.8350 | 0.7525 | 0.3881 | 0.5105 | rfe    | SVC(C=6, kernel='poly') |
| 0.8350 | 0.7201 | 0.4233 | 0.5323 | rfe    | SVC(C=2, kernel='linear') |
| 0.8350 | 0.7353 | 0.4026 | 0.5198 | mutual | KNeighborsClassifier(n_neighbors=27) |
| 0.8351 | 0.7548 | 0.3862 | 0.5094 | rfe    | SVC(C=5, kernel='poly') |
| 0.8351 | 0.7303 | 0.4109 | 0.5252 | mutual | KNeighborsClassifier(n_neighbors=23) |
| 0.8351 | 0.7229 | 0.4199 | 0.5303 | all    | KNeighborsClassifier(n_neighbors=13) |
| 0.8353 | 0.7489 | 0.3947 | 0.5150 | rfe    | SVC(C=8, kernel='poly') |
| 0.8353 | 0.7231 | 0.4213 | 0.5316 | mutual | KNeighborsClassifier(n_neighbors=17) |
| 0.8353 | 0.7196 | 0.4258 | 0.5342 | rfe    | SVC(C=1, kernel='linear') |
| 0.8353 | 0.7223 | 0.4235 | 0.5329 | mutual | KNeighborsClassifier(n_neighbors=15) |
| 0.8356 | 0.7512 | 0.3940 | 0.5151 | rfe    | SVC(C=7, kernel='poly') |
| 0.8356 | 0.7169 | 0.4328 | 0.5387 | all    | KNeighborsClassifier(n_neighbors=9) |
| 0.8356 | 0.7415 | 0.3996 | 0.5188 | mutual | KNeighborsClassifier(n_neighbors=29) |
| 0.8356 | 0.7176 | 0.4297 | 0.5367 | all    | KNeighborsClassifier(n_neighbors=11) |
| 0.8356 | 0.7249 | 0.4224 | 0.5326 | rfe    | KNeighborsClassifier(n_neighbors=25) |
| 0.8356 | 0.7177 | 0.4327 | 0.5385 | rfe    | KNeighborsClassifier(n_neighbors=15) |
| 0.8357 | 0.7227 | 0.4260 | 0.5349 | rfe    | SVC(C=5, kernel='linear') |
| 0.8358 | 0.6933 | 0.4721 | 0.5605 | all    | RandomForestClassifier(n_estimators=19) |       
| 0.8358 | 0.7504 | 0.3966 | 0.5170 | rfe    | SVC(C=9, kernel='poly') |
| 0.8358 | 0.7220 | 0.4272 | 0.5358 | rfe    | SVC(C=6, kernel='linear') |
| 0.8360 | 0.7250 | 0.4247 | 0.5346 | rfe    | SVC(C=9, kernel='linear') |
| 0.8360 | 0.7254 | 0.4241 | 0.5342 | rfe    | SVC(C=8, kernel='linear') |
| 0.8360 | 0.7223 | 0.4278 | 0.5363 | rfe    | SVC(C=4, kernel='linear') |
| 0.8360 | 0.7184 | 0.4342 | 0.5402 | rfe    | KNeighborsClassifier(n_neighbors=17) |
| 0.8361 | 0.7236 | 0.4272 | 0.5363 | rfe    | SVC(C=7, kernel='linear') |
| 0.8361 | 0.7288 | 0.4198 | 0.5318 | rfe    | KNeighborsClassifier(n_neighbors=27) |
| 0.8365 | 0.7105 | 0.4514 | 0.5504 | all    | SVC(C=6, kernel='linear') |
| 0.8365 | 0.7232 | 0.4302 | 0.5386 | rfe    | SVC(C=3, kernel='linear') |
| 0.8365 | 0.7314 | 0.4195 | 0.5323 | rfe    | KNeighborsClassifier(n_neighbors=29) |
| 0.8367 | 0.7077 | 0.4555 | 0.5530 | lasso  | SVC(C=5, kernel='linear') |
| 0.8368 | 0.7091 | 0.4582 | 0.5543 | all    | SVC(C=3, kernel='linear') |
| 0.8368 | 0.7110 | 0.4528 | 0.5513 | all    | SVC(C=1, kernel='linear') |
| 0.8371 | 0.7115 | 0.4535 | 0.5521 | all    | SVC(C=8, kernel='linear') |
| 0.8371 | 0.7068 | 0.4615 | 0.5568 | all    | SVC(C=5, kernel='linear') |
| 0.8371 | 0.7086 | 0.4579 | 0.5549 | lasso  | SVC(C=3, kernel='linear') |
| 0.8372 | 0.7280 | 0.4302 | 0.5397 | lasso  | KNeighborsClassifier(n_neighbors=19) |
| 0.8372 | 0.7055 | 0.4618 | 0.5571 | all    | RandomForestClassifier(n_estimators=17) |       
| 0.8372 | 0.7138 | 0.4493 | 0.5499 | all    | SVC(C=2, kernel='linear') |
| 0.8372 | 0.7091 | 0.4571 | 0.5547 | lasso  | SVC(C=2, kernel='linear') |
| 0.8372 | 0.7176 | 0.4440 | 0.5477 | lasso  | KNeighborsClassifier(n_neighbors=11) |
| 0.8373 | 0.7147 | 0.4529 | 0.5525 | lasso  | SVC(C=1, kernel='linear') |
| 0.8373 | 0.7093 | 0.4577 | 0.5552 | lasso  | SVC(C=4, kernel='linear') |
| 0.8375 | 0.7229 | 0.4395 | 0.5455 | lasso  | KNeighborsClassifier(n_neighbors=13) |
| 0.8378 | 0.7354 | 0.4230 | 0.5363 | all    | KNeighborsClassifier(n_neighbors=15) |
| 0.8378 | 0.7103 | 0.4598 | 0.5569 | lasso  | SVC(C=6, kernel='linear') |
| 0.8379 | 0.7291 | 0.4342 | 0.5430 | lasso  | KNeighborsClassifier(n_neighbors=17) |
| 0.8379 | 0.7239 | 0.4394 | 0.5458 | lasso  | KNeighborsClassifier(n_neighbors=15) |
| 0.8379 | 0.7105 | 0.4595 | 0.5570 | lasso  | SVC(C=7, kernel='linear') |
| 0.8380 | 0.7341 | 0.4281 | 0.5399 | lasso  | KNeighborsClassifier(n_neighbors=25) |
| 0.8380 | 0.7180 | 0.4521 | 0.5528 | all    | SVC(C=7, kernel='linear') |
| 0.8380 | 0.7108 | 0.4610 | 0.5580 | lasso  | SVC(C=9, kernel='linear') |
| 0.8383 | 0.7365 | 0.4273 | 0.5398 | lasso  | KNeighborsClassifier(n_neighbors=23) |
| 0.8383 | 0.7136 | 0.4586 | 0.5570 | all    | SVC(C=9, kernel='linear') |
| 0.8383 | 0.7114 | 0.4622 | 0.5591 | lasso  | SVC(C=8, kernel='linear') |
| 0.8389 | 0.7377 | 0.4294 | 0.5418 | lasso  | KNeighborsClassifier(n_neighbors=21) |
| 0.8390 | 0.7152 | 0.4605 | 0.5591 | all    | SVC(C=4, kernel='linear') |
| 0.8391 | 0.7458 | 0.4214 | 0.5376 | all    | KNeighborsClassifier(n_neighbors=19) |
| 0.8393 | 0.7399 | 0.4293 | 0.5424 | lasso  | KNeighborsClassifier(n_neighbors=27) |
| 0.8393 | 0.7612 | 0.4045 | 0.5274 | all    | KNeighborsClassifier(n_neighbors=27) |
| 0.8393 | 0.7650 | 0.4012 | 0.5254 | all    | KNeighborsClassifier(n_neighbors=29) |
| 0.8395 | 0.7509 | 0.4174 | 0.5358 | all    | KNeighborsClassifier(n_neighbors=21) |
| 0.8400 | 0.7627 | 0.4086 | 0.5310 | all    | KNeighborsClassifier(n_neighbors=25) |
| 0.8402 | 0.7421 | 0.4329 | 0.5460 | lasso  | KNeighborsClassifier(n_neighbors=29) |
| 0.8402 | 0.7596 | 0.4125 | 0.5338 | all    | KNeighborsClassifier(n_neighbors=23) |
| 0.8402 | 0.7600 | 0.4126 | 0.5339 | all    | VotingClassifier(5 estimators) |
| 0.8404 | 0.7606 | 0.4125 | 0.5340 | all    | VotingClassifier(3 estimators) |
| 0.8404 | 0.7468 | 0.4284 | 0.5433 | all    | KNeighborsClassifier(n_neighbors=17) |
| 0.8406 | 0.7657 | 0.4098 | 0.5329 | all    | VotingClassifier(7 estimators) |
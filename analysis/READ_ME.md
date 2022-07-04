# Analysis Folder Documentation

This documentation presents an overview of the content of this folder. More details can be viewed at each specific subfolder and files. The analysis folder is organized similarly to the CRISP-DM framework proposes for organizing a data-driven project.

## 💼 1_business_problem

The folder contains a simplified version of the project's chart with the main information of business problem understandment and data problem defintion. It also contains the definition of performance and model selection metrics.

## 🔍 2_exploratory_data_analysis

The folder contains notebooks of exploratory data analysis. Also, the main data quality issues and explored and dealt with. The key findings are:

- General
    - No missing data
    - Some assets last much longer than the average and they are presented as candidates for outliers
    - A 99 % symmetric filter is applied - it was assumed that the underestimation of RUL for these assets would have less impact than unexpected stops due to failures
    - A feature engineering for the RUL calculation was performed (RUL regression problem). Similarly, it was done for creation of failure class (classification problem).
- Regression Analysis
    - Features with zero or vey low variance were removed
    - There are no duplicated rows or columns
    - The candidates for outliers looked as signal noise and the choice of this first attempt was not to deal with them
    - Features with very low correlation with RUL were removed
    - There is a very intense degree of features cross correlation, making it difficult to select only by looking to correlation. An automated feature selection was proposed.
    - The correlations with RUL presented non-linear behavior.
- Classification Analysis
    - The data quality issues were treated the same as in the regression analysis
    - The dataset classes are very imbalaced and thus, demand additional treatment in the modeling step
    - The classes seem to be fairly separable in a linear way. Linear models are suggested along with the ones proposed in the regression analysis
    - Hypotheses testing showed a behavior similar to correlation analysis results - i.e., except for `set1` and `set2` tags, all other variables showed statistically significant effect on separating both classes.

## 💻 3_modeling

This folder contains the two scripts used for training and monitoring the models' performance.

- **1_RLU_regression**: the feature selection was made using `RFECV` with LinearRegression for the speed of execution. The models tested were RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor and XGBoostRegressor. The validation strategy was the k-fold cross validation, using the $R^2$ as performance selection metric. The hyperparameter selection was performed using RandomizedSearch with the same cross-validation technique.
- **2_failure_classification**: the feature selection was made using `RFECV` with LogisticRegression for the speed of execution. The models tested were RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, XGBoostClassifier and LogisticRegression. The validation strategy was the stratified k-fold cross validation, using the $F1$ as performance selection metric. The hyperparameter selection was performed using RandomizedSearch with the same cross-validation technique.

For both models, the scaling techinque was the RobustScaler, in order to aboid statistic bias due to extreme values. In the classification model, the modeling Pipeline included SMOTE resampling to eliminate class imbalance bias. The SMOTE was applied with default parameters.

## 🎯 Model Testing

- **General**: both models presented high degree of overfitting, with poor generalization performance. This could be due to high variance of values of different assets for the same runtime.
- **1_RLU_regression**: presented an average error pf 38 cycles, which represent about 20 % of average failure runtime.
The feature importance analysis presented evidence that the feature selection technique was not effective, because not all features presented significant importance. Most important features were `runtime`, `tag21` and `tag4`.
- **2_failure_classification**: the most important features were `tag3`, `tag8` and `runtime`. However, this model presented more features with non significant effects. There's is an opportunity of applying the probability estimation by using a conservative approach, provided that the costs of early maintenances due to false negatives are assessed and accepted.

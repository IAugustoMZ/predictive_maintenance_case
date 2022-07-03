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

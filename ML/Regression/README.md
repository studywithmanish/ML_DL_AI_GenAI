# Boston Housing Regression Analysis

This project provides a comprehensive analysis of the Boston Housing dataset, focusing on predicting median home values using various regression techniques. The analysis is divided into multiple parts to make it more manageable and focused.

## Project Structure

1. **Data Preprocessing and EDA** (Jupyter Notebook)
   - `1_Data_Preprocessing_and_EDA.ipynb`: Exploratory data analysis, handling missing values, outlier detection, feature scaling, and feature engineering.

2. **Statistical Analysis** (Python Script)
   - `2_Statistical_Analysis_and_Hypothesis_Testing.py`: Normality testing, outlier analysis, chi-square tests, ANOVA, and correlation analysis.

3. **Basic Regression Models** (Python Script)
   - `3_Basic_Regression_Models.py`: Implementation of Linear Regression, Polynomial Regression, Ridge, Lasso, and ElasticNet with hyperparameter tuning.

4. **Advanced Models** (Python Script)
   - `4_Advanced_Models_Ensemble_and_Stacking.py`: Implementation of Random Forest, Gradient Boosting, XGBoost, LightGBM, and ensemble methods.

5. **Model Comparison and Final Analysis** (Python Script)
   - `5_Model_Comparison_and_Final_Analysis.py`: Statistical comparison of all models, final model selection, and comprehensive analysis.

## Dataset

The Boston Housing dataset contains information collected by the U.S. Census Service concerning housing in the area of Boston, MA. The dataset includes the following features:

- CRIM: Per capita crime rate by town
- ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS: Proportion of non-retail business acres per town
- CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- NOX: Nitric oxides concentration (parts per 10 million)
- RM: Average number of rooms per dwelling
- AGE: Proportion of owner-occupied units built prior to 1940
- DIS: Weighted distances to five Boston employment centers
- RAD: Index of accessibility to radial highways
- TAX: Full-value property-tax rate per $10,000
- PTRATIO: Pupil-teacher ratio by town
- B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT: % lower status of the population
- MEDV: Median value of owner-occupied homes in $1000's (target variable)

## Requirements

To run the scripts in this project, you need the following Python packages:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
statsmodels
xgboost
lightgbm
joblib
```

You can install these packages using pip:

```
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels xgboost lightgbm joblib
```

## How to Run

### 1. Jupyter Notebook

To run the Jupyter notebook:

```
jupyter notebook 1_Data_Preprocessing_and_EDA.ipynb
```

### 2. Python Scripts

Run the Python scripts in sequence:

```
python 2_Statistical_Analysis_and_Hypothesis_Testing.py
python 3_Basic_Regression_Models.py
python 4_Advanced_Models_Ensemble_and_Stacking.py
python 5_Model_Comparison_and_Final_Analysis.py
```

## Output

The scripts generate various outputs:

1. **Visualizations**: Plots and charts saved in the working directory and model_artifacts folder
2. **Model Artifacts**: Trained models saved in the model_artifacts folder
3. **Metrics**: Performance metrics saved in CSV files
4. **Final Analysis**: Comprehensive report in the final_analysis folder

## Key Findings

The analysis reveals:

1. The most important features for predicting median home values
2. The best performing regression models for this dataset
3. Statistical significance of model performance differences
4. Insights into the relationships between features and housing prices

## Final Report

A comprehensive final report is generated in `final_analysis/final_report.md`, which summarizes all findings and provides recommendations for further analysis and model deployment.

## Author

This project was created as part of a regression analysis assignment.

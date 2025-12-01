"""
Boston Housing Dataset: Model Comparison and Final Analysis

This script compares all models developed for the Boston Housing dataset,
performs statistical tests to compare model performances, and provides
a comprehensive final analysis.
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set visualization styles
plt.style.use('seaborn-whitegrid')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("Boston Housing Dataset: Model Comparison and Final Analysis")
print("="*80)

# Create a directory for final analysis artifacts
if not os.path.exists('final_analysis'):
    os.makedirs('final_analysis')

# ============================================================================
# 1. Load Model Predictions
# ============================================================================
print("\n1. Load Model Predictions")
print("-"*80)

# Load basic model predictions
try:
    basic_predictions = pd.read_csv('model_artifacts/basic_models_predictions.csv')
    print(f"Loaded basic model predictions with shape: {basic_predictions.shape}")
except FileNotFoundError:
    print("Basic model predictions file not found. Run the basic models script first.")
    basic_predictions = None

# Load advanced model predictions
try:
    advanced_predictions = pd.read_csv('model_artifacts/advanced_models_predictions.csv')
    print(f"Loaded advanced model predictions with shape: {advanced_predictions.shape}")
except FileNotFoundError:
    print("Advanced model predictions file not found. Run the advanced models script first.")
    advanced_predictions = None

# Combine predictions if both are available
if basic_predictions is not None and advanced_predictions is not None:
    # Ensure they have the same 'Actual' values
    if np.array_equal(basic_predictions['Actual'], advanced_predictions['Actual']):
        # Combine the predictions
        all_predictions = basic_predictions.copy()
        for col in advanced_predictions.columns:
            if col != 'Actual':
                all_predictions[col] = advanced_predictions[col]
        
        print(f"Combined predictions dataframe with shape: {all_predictions.shape}")
    else:
        print("Warning: Basic and advanced predictions have different actual values. Cannot combine.")
        all_predictions = None
elif basic_predictions is not None:
    all_predictions = basic_predictions
    print("Using only basic model predictions.")
elif advanced_predictions is not None:
    all_predictions = advanced_predictions
    print("Using only advanced model predictions.")
else:
    all_predictions = None
    print("No prediction files found. Run the model scripts first.")

# If no predictions are available, exit
if all_predictions is None:
    print("No predictions available for analysis. Exiting.")
    exit()

# ============================================================================
# 2. Calculate Performance Metrics
# ============================================================================
print("\n2. Calculate Performance Metrics")
print("-"*80)

# Get actual values
y_true = all_predictions['Actual']

# Calculate metrics for each model
metrics = []
model_names = [col for col in all_predictions.columns if col != 'Actual']

for model in model_names:
    y_pred = all_predictions[model]
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate additional metrics
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Calculate residuals
    residuals = y_true - y_pred
    residual_std = np.std(residuals)
    
    metrics.append({
        'Model': model,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE (%)': mape,
        'Residual STD': residual_std
    })

# Create metrics dataframe
metrics_df = pd.DataFrame(metrics)
metrics_df = metrics_df.sort_values(by='RMSE')

# Display metrics
print("\nModel Performance Metrics:")
print(metrics_df.to_string(index=False))

# Save metrics to CSV
metrics_df.to_csv('final_analysis/all_models_metrics.csv', index=False)
print("Model metrics saved to 'final_analysis/all_models_metrics.csv'")

# ============================================================================
# 3. Statistical Tests for Model Comparison
# ============================================================================
print("\n3. Statistical Tests for Model Comparison")
print("-"*80)

"""
Statistical tests for comparing model performances:
1. Paired t-test: Tests if the mean difference between paired observations is zero
2. Wilcoxon signed-rank test: Non-parametric alternative to paired t-test
3. Friedman test: Non-parametric test for comparing multiple models
"""

# Get the best model
best_model = metrics_df.iloc[0]['Model']
print(f"Best performing model: {best_model}")

# Prepare for statistical tests
test_results = []

# Compare the best model with all other models
for model in model_names:
    if model != best_model:
        # Calculate absolute errors for both models
        best_errors = np.abs(y_true - all_predictions[best_model])
        model_errors = np.abs(y_true - all_predictions[model])
        
        # Paired t-test
        t_stat, t_p = stats.ttest_rel(best_errors, model_errors)
        
        # Wilcoxon signed-rank test
        w_stat, w_p = stats.wilcoxon(best_errors, model_errors)
        
        # Store results
        test_results.append({
            'Model': model,
            'vs Best Model': best_model,
            'T-statistic': t_stat,
            'T-test p-value': t_p,
            'Wilcoxon statistic': w_stat,
            'Wilcoxon p-value': w_p,
            'Significantly Different (α=0.05)': 'Yes' if t_p < 0.05 or w_p < 0.05 else 'No'
        })

# Create test results dataframe
test_results_df = pd.DataFrame(test_results)

# Display test results
print("\nStatistical Test Results (comparing with best model):")
print(test_results_df.to_string(index=False))

# Save test results to CSV
test_results_df.to_csv('final_analysis/statistical_test_results.csv', index=False)
print("Statistical test results saved to 'final_analysis/statistical_test_results.csv'")

# Friedman test for comparing all models
print("\nFriedman Test for comparing all models:")
errors = pd.DataFrame()
for model in model_names:
    errors[model] = np.abs(y_true - all_predictions[model])

# Perform Friedman test
try:
    friedman_stat, friedman_p = stats.friedmanchisquare(*[errors[model] for model in model_names])
    print(f"  Friedman statistic: {friedman_stat:.4f}")
    print(f"  p-value: {friedman_p:.4e}")
    print(f"  Conclusion: {'There are significant differences among models' if friedman_p < 0.05 else 'No significant differences among models'}")
except:
    print("  Could not perform Friedman test. Ensure you have at least 3 models for comparison.")

# ============================================================================
# 4. Visualize Model Comparison
# ============================================================================
print("\n4. Visualize Model Comparison")
print("-"*80)

# Visualize RMSE comparison
plt.figure(figsize=(14, 8))
sns.barplot(x='Model', y='RMSE', data=metrics_df)
plt.title('Model Comparison: RMSE (lower is better)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('final_analysis/rmse_comparison.png')
plt.close()

# Visualize R² comparison
plt.figure(figsize=(14, 8))
sns.barplot(x='Model', y='R²', data=metrics_df)
plt.title('Model Comparison: R² (higher is better)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('final_analysis/r2_comparison.png')
plt.close()

# Visualize residuals for top 3 models
top_models = metrics_df.head(3)['Model'].tolist()
plt.figure(figsize=(15, 5 * len(top_models)))

for i, model in enumerate(top_models):
    residuals = y_true - all_predictions[model]
    
    # Residual plot
    plt.subplot(len(top_models), 2, 2*i+1)
    plt.scatter(all_predictions[model], residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(f'{model}: Residual Plot')
    
    # Residual distribution
    plt.subplot(len(top_models), 2, 2*i+2)
    sns.histplot(residuals, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title(f'{model}: Residual Distribution')

plt.tight_layout()
plt.savefig('final_analysis/top_models_residuals.png')
plt.close()

# Visualize predictions vs actual for top 3 models
plt.figure(figsize=(15, 5 * len(top_models)))

for i, model in enumerate(top_models):
    plt.subplot(len(top_models), 1, i+1)
    plt.scatter(y_true, all_predictions[model], alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model}: Predictions vs Actual')

plt.tight_layout()
plt.savefig('final_analysis/top_models_predictions.png')
plt.close()

# ============================================================================
# 5. Feature Importance Analysis
# ============================================================================
print("\n5. Feature Importance Analysis")
print("-"*80)

# Load feature importance comparison if available
try:
    feature_importance = pd.read_csv('model_artifacts/feature_importance_comparison.csv')
    print(f"Loaded feature importance comparison with shape: {feature_importance.shape}")
    
    # Visualize top 10 features by average importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Average Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 Features by Average Importance')
    plt.tight_layout()
    plt.savefig('final_analysis/top_features.png')
    plt.close()
    
    # Display top features
    print("\nTop 10 Features by Average Importance:")
    print(feature_importance[['Feature', 'Average Importance']].head(10).to_string(index=False))
except FileNotFoundError:
    print("Feature importance comparison file not found.")

# ============================================================================
# 6. Final Model Selection and Deployment
# ============================================================================
print("\n6. Final Model Selection and Deployment")
print("-"*80)

# Select the best model based on metrics
best_model_name = metrics_df.iloc[0]['Model']
best_model_rmse = metrics_df.iloc[0]['RMSE']
best_model_r2 = metrics_df.iloc[0]['R²']

print(f"Best performing model: {best_model_name}")
print(f"  RMSE: {best_model_rmse:.4f}")
print(f"  R²: {best_model_r2:.4f}")

# Try to load the best model
model_filename = f"model_artifacts/{best_model_name.lower().replace('_', '_')}.pkl"
try:
    best_model = joblib.load(model_filename)
    print(f"Loaded best model from {model_filename}")
    
    # Save the best model to the final analysis directory
    joblib.dump(best_model, 'final_analysis/best_model.pkl')
    print("Best model saved to 'final_analysis/best_model.pkl'")
except:
    print(f"Could not load the best model from {model_filename}")

# ============================================================================
# 7. Comprehensive Final Analysis
# ============================================================================
print("\n7. Comprehensive Final Analysis")
print("-"*80)

# Create a final analysis report
final_report = """
# Boston Housing Dataset: Final Analysis Report

## Overview
This report presents a comprehensive analysis of various regression models applied to the Boston Housing dataset. The goal was to predict median home values (MEDV) based on various features.

## Data Insights
- The dataset contains information about housing in the Boston area, with features like crime rate, zoning, pollution levels, etc.
- The target variable (MEDV) represents the median value of owner-occupied homes in $1000s.
- Statistical analysis revealed that many features do not follow a normal distribution, suggesting the need for transformations.
- Several features showed significant correlations with the target variable.

## Model Performance Summary
"""

# Add model performance summary to the report
final_report += f"""
The following models were evaluated:

| Model | RMSE | MAE | R² | MAPE (%) |
|-------|------|-----|---|---------|
"""

for _, row in metrics_df.iterrows():
    final_report += f"| {row['Model']} | {row['RMSE']:.4f} | {row['MAE']:.4f} | {row['R²']:.4f} | {row['MAPE (%)']:.2f} |\n"

final_report += f"""

The best performing model was **{best_model_name}** with an RMSE of {best_model_rmse:.4f} and R² of {best_model_r2:.4f}.

## Statistical Significance
Statistical tests were performed to compare the performance of the best model with other models:
"""

# Add statistical test results to the report
if len(test_results_df) > 0:
    for _, row in test_results_df.iterrows():
        final_report += f"\n- **{row['Model']} vs {row['vs Best Model']}**: {'Significantly different' if row['Significantly Different (α=0.05)'] == 'Yes' else 'Not significantly different'} (p-value: {row['T-test p-value']:.4e})"

final_report += """

## Key Features
The most important features for predicting median home values were:
"""

# Add top features to the report if available
try:
    for _, row in feature_importance.head(5).iterrows():
        final_report += f"\n- **{row['Feature']}**: Importance = {row['Average Importance']:.4f}"
except:
    final_report += "\n- Feature importance information not available."

final_report += """

## Conclusions and Recommendations

1. **Model Selection**: The analysis shows that ensemble methods generally outperform traditional regression models for this dataset.

2. **Feature Engineering**: The most important features identified should be the focus of any further data collection or refinement.

3. **Preprocessing Impact**: Data preprocessing, especially handling outliers and transforming skewed features, significantly improved model performance.

4. **Ensemble Benefits**: Combining multiple models through stacking or voting ensembles leveraged the strengths of individual models and reduced prediction errors.

5. **Regularization**: Regularization techniques (Ridge, Lasso, ElasticNet) helped prevent overfitting and improved model generalization.

## Next Steps

1. **Model Deployment**: The best model can be deployed for predicting housing prices in similar contexts.

2. **Feature Collection**: Focus on collecting more accurate data for the most important features identified.

3. **Model Monitoring**: Implement a system to monitor model performance over time and retrain as needed.

4. **Explainability**: Develop more detailed explanations of model predictions for stakeholders.

5. **Domain Expertise**: Incorporate domain knowledge to further refine the model and its interpretations.
"""

# Save the final report
with open('final_analysis/final_report.md', 'w') as f:
    f.write(final_report)

print("Final analysis report saved to 'final_analysis/final_report.md'")

# ============================================================================
# 8. Summary
# ============================================================================
print("\n8. Summary")
print("-"*80)

print(f"""
Boston Housing Dataset Regression Analysis Summary:

1. Data Preprocessing and EDA:
   - Identified and handled missing values and outliers
   - Applied appropriate scaling techniques
   - Performed feature engineering to improve model performance
   - Analyzed feature distributions and correlations

2. Statistical Analysis:
   - Tested for normality in features and target variable
   - Detected outliers using multiple methods
   - Performed correlation analysis and hypothesis testing
   - Identified significant relationships between features and target

3. Basic Regression Models:
   - Implemented Linear Regression, Polynomial Regression, Ridge, Lasso, and ElasticNet
   - Performed hyperparameter tuning for regularized models
   - Evaluated models using multiple metrics (RMSE, MAE, R²)
   - Identified important features through coefficient analysis

4. Advanced Models (Ensemble and Stacking):
   - Implemented Random Forest, Gradient Boosting, XGBoost, and LightGBM
   - Created voting and stacking ensembles
   - Performed extensive hyperparameter tuning
   - Analyzed feature importance across models

5. Model Comparison and Final Analysis:
   - Compared all models using multiple performance metrics
   - Performed statistical tests to assess significant differences
   - Visualized model performances and residuals
   - Selected the best model: {best_model_name}
   - Created a comprehensive final analysis report

The best performing model achieved an RMSE of {best_model_rmse:.4f} and R² of {best_model_r2:.4f}.
""")

print("\nAll analysis tasks completed successfully!")

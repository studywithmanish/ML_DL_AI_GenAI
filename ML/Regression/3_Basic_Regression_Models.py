"""
Boston Housing Dataset: Basic Regression Models

This script implements and evaluates basic regression models for the Boston Housing dataset.
It includes Linear Regression, Ridge, Lasso, and ElasticNet with hyperparameter tuning.
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import os

# Set visualization styles
plt.style.use('seaborn-whitegrid')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Create a directory for model artifacts
if not os.path.exists('model_artifacts'):
    os.makedirs('model_artifacts')

print("Boston Housing Dataset: Basic Regression Models")
print("="*80)

# Load the processed dataset
try:
    df = pd.read_csv('boston_housing_processed.csv')
    print("Loaded processed dataset")
except FileNotFoundError:
    df = pd.read_csv('bostonHousing.csv')
    print("Loaded original dataset")

# ============================================================================
# 1. Data Preparation
# ============================================================================
print("\n1. Data Preparation")
print("-"*80)

# Separate features and target
X = df.drop(['MEDV', 'CAT. MEDV'], axis=1)
y = df['MEDV']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for future use
joblib.dump(scaler, 'model_artifacts/scaler.pkl')
print("Scaler saved to 'model_artifacts/scaler.pkl'")

# ============================================================================
# 2. Model Evaluation Function
# ============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Train and evaluate a regression model
    
    Parameters:
    model: The regression model to evaluate
    X_train, X_test, y_train, y_test: Training and testing data
    model_name: Name of the model for reporting
    
    Returns:
    dict: Dictionary of evaluation metrics
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmse = np.sqrt(-cross_val_score(model, X_train, y_train, 
                                      scoring='neg_mean_squared_error', 
                                      cv=cv))
    cv_r2 = cross_val_score(model, X_train, y_train, 
                           scoring='r2', 
                           cv=cv)
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"  Training RMSE: {train_rmse:.4f}")
    print(f"  Testing RMSE: {test_rmse:.4f}")
    print(f"  Training MAE: {train_mae:.4f}")
    print(f"  Testing MAE: {test_mae:.4f}")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Testing R²: {test_r2:.4f}")
    print(f"  Cross-Validation RMSE: {cv_rmse.mean():.4f} (±{cv_rmse.std():.4f})")
    print(f"  Cross-Validation R²: {cv_r2.mean():.4f} (±{cv_r2.std():.4f})")
    
    # Visualize predictions vs actual
    plt.figure(figsize=(12, 5))
    
    # Training set
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name}: Training Set')
    
    # Testing set
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name}: Testing Set')
    
    plt.tight_layout()
    plt.savefig(f'model_artifacts/{model_name.lower().replace(" ", "_")}_predictions.png')
    plt.close()
    
    # Visualize residuals
    plt.figure(figsize=(12, 5))
    
    # Training set residuals
    plt.subplot(1, 2, 1)
    residuals_train = y_train - y_train_pred
    plt.scatter(y_train_pred, residuals_train, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(f'{model_name}: Training Residuals')
    
    # Testing set residuals
    plt.subplot(1, 2, 2)
    residuals_test = y_test - y_test_pred
    plt.scatter(y_test_pred, residuals_test, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(f'{model_name}: Testing Residuals')
    
    plt.tight_layout()
    plt.savefig(f'model_artifacts/{model_name.lower().replace(" ", "_")}_residuals.png')
    plt.close()
    
    # Return metrics
    return {
        'model': model,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_rmse_mean': cv_rmse.mean(),
        'cv_rmse_std': cv_rmse.std(),
        'cv_r2_mean': cv_r2.mean(),
        'cv_r2_std': cv_r2.std(),
        'y_test_pred': y_test_pred
    }

# ============================================================================
# 3. Linear Regression
# ============================================================================
print("\n3. Linear Regression")
print("-"*80)

# Simple Linear Regression
lr = LinearRegression()
lr_results = evaluate_model(lr, X_train_scaled, X_test_scaled, y_train, y_test, "Linear Regression")

# Save the model
joblib.dump(lr, 'model_artifacts/linear_regression.pkl')
print("Linear Regression model saved to 'model_artifacts/linear_regression.pkl'")

# Feature importance for Linear Regression
lr_coef = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_
})
lr_coef = lr_coef.sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Coefficient', y='Feature', data=lr_coef)
plt.title('Linear Regression: Feature Coefficients')
plt.tight_layout()
plt.savefig('model_artifacts/linear_regression_coefficients.png')
plt.close()

print("\nLinear Regression Feature Importance:")
for index, row in lr_coef.iterrows():
    print(f"  {row['Feature']}: {row['Coefficient']:.4f}")

# ============================================================================
# 4. Polynomial Regression
# ============================================================================
print("\n4. Polynomial Regression")
print("-"*80)

# Create a pipeline for polynomial regression
poly_degrees = [2, 3]
poly_results = {}

for degree in poly_degrees:
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('scaler', StandardScaler()),
        ('regression', LinearRegression())
    ])
    
    model_name = f"Polynomial Regression (degree={degree})"
    poly_results[degree] = evaluate_model(
        poly_pipeline, X_train, X_test, y_train, y_test, model_name
    )
    
    # Save the model
    joblib.dump(poly_pipeline, f'model_artifacts/polynomial_regression_degree_{degree}.pkl')
    print(f"Polynomial Regression (degree={degree}) model saved")

# ============================================================================
# 5. Ridge Regression with Hyperparameter Tuning
# ============================================================================
print("\n5. Ridge Regression")
print("-"*80)

# Define the parameter grid
ridge_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

# Create a grid search
ridge = Ridge(random_state=42)
ridge_grid = GridSearchCV(
    ridge, ridge_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)

# Fit the grid search
ridge_grid.fit(X_train_scaled, y_train)

# Get the best model
best_ridge = ridge_grid.best_estimator_
best_ridge_alpha = ridge_grid.best_params_['alpha']

print(f"Best Ridge alpha: {best_ridge_alpha}")

# Evaluate the best Ridge model
ridge_results = evaluate_model(
    best_ridge, X_train_scaled, X_test_scaled, y_train, y_test, "Ridge Regression"
)

# Save the model
joblib.dump(best_ridge, 'model_artifacts/ridge_regression.pkl')
print("Ridge Regression model saved to 'model_artifacts/ridge_regression.pkl'")

# Feature importance for Ridge Regression
ridge_coef = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': best_ridge.coef_
})
ridge_coef = ridge_coef.sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Coefficient', y='Feature', data=ridge_coef)
plt.title('Ridge Regression: Feature Coefficients')
plt.tight_layout()
plt.savefig('model_artifacts/ridge_regression_coefficients.png')
plt.close()

# ============================================================================
# 6. Lasso Regression with Hyperparameter Tuning
# ============================================================================
print("\n6. Lasso Regression")
print("-"*80)

# Define the parameter grid
lasso_params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}

# Create a grid search
lasso = Lasso(random_state=42, max_iter=10000)
lasso_grid = GridSearchCV(
    lasso, lasso_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)

# Fit the grid search
lasso_grid.fit(X_train_scaled, y_train)

# Get the best model
best_lasso = lasso_grid.best_estimator_
best_lasso_alpha = lasso_grid.best_params_['alpha']

print(f"Best Lasso alpha: {best_lasso_alpha}")

# Evaluate the best Lasso model
lasso_results = evaluate_model(
    best_lasso, X_train_scaled, X_test_scaled, y_train, y_test, "Lasso Regression"
)

# Save the model
joblib.dump(best_lasso, 'model_artifacts/lasso_regression.pkl')
print("Lasso Regression model saved to 'model_artifacts/lasso_regression.pkl'")

# Feature importance for Lasso Regression
lasso_coef = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': best_lasso.coef_
})
lasso_coef = lasso_coef.sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Coefficient', y='Feature', data=lasso_coef)
plt.title('Lasso Regression: Feature Coefficients')
plt.tight_layout()
plt.savefig('model_artifacts/lasso_regression_coefficients.png')
plt.close()

# ============================================================================
# 7. ElasticNet with Hyperparameter Tuning
# ============================================================================
print("\n7. ElasticNet Regression")
print("-"*80)

# Define the parameter grid
elasticnet_params = {
    'alpha': [0.001, 0.01, 0.1, 1.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

# Create a grid search
elasticnet = ElasticNet(random_state=42, max_iter=10000)
elasticnet_grid = GridSearchCV(
    elasticnet, elasticnet_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)

# Fit the grid search
elasticnet_grid.fit(X_train_scaled, y_train)

# Get the best model
best_elasticnet = elasticnet_grid.best_estimator_
best_elasticnet_params = elasticnet_grid.best_params_

print(f"Best ElasticNet parameters: alpha={best_elasticnet_params['alpha']}, l1_ratio={best_elasticnet_params['l1_ratio']}")

# Evaluate the best ElasticNet model
elasticnet_results = evaluate_model(
    best_elasticnet, X_train_scaled, X_test_scaled, y_train, y_test, "ElasticNet Regression"
)

# Save the model
joblib.dump(best_elasticnet, 'model_artifacts/elasticnet_regression.pkl')
print("ElasticNet Regression model saved to 'model_artifacts/elasticnet_regression.pkl'")

# Feature importance for ElasticNet Regression
elasticnet_coef = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': best_elasticnet.coef_
})
elasticnet_coef = elasticnet_coef.sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Coefficient', y='Feature', data=elasticnet_coef)
plt.title('ElasticNet Regression: Feature Coefficients')
plt.tight_layout()
plt.savefig('model_artifacts/elasticnet_regression_coefficients.png')
plt.close()

# ============================================================================
# 8. Model Comparison
# ============================================================================
print("\n8. Model Comparison")
print("-"*80)

# Collect results
models = {
    'Linear Regression': lr_results,
    'Polynomial Regression (degree=2)': poly_results[2],
    'Ridge Regression': ridge_results,
    'Lasso Regression': lasso_results,
    'ElasticNet Regression': elasticnet_results
}

# Create comparison dataframes
comparison_df = pd.DataFrame({
    'Model': list(models.keys()),
    'Training RMSE': [models[m]['train_rmse'] for m in models],
    'Testing RMSE': [models[m]['test_rmse'] for m in models],
    'Training MAE': [models[m]['train_mae'] for m in models],
    'Testing MAE': [models[m]['test_mae'] for m in models],
    'Training R²': [models[m]['train_r2'] for m in models],
    'Testing R²': [models[m]['test_r2'] for m in models],
    'CV RMSE': [models[m]['cv_rmse_mean'] for m in models],
    'CV R²': [models[m]['cv_r2_mean'] for m in models]
})

# Sort by testing RMSE
comparison_df = comparison_df.sort_values(by='Testing RMSE')

# Display comparison
print("\nModel Comparison:")
print(comparison_df.to_string(index=False))

# Save comparison to CSV
comparison_df.to_csv('model_artifacts/basic_models_comparison.csv', index=False)
print("Model comparison saved to 'model_artifacts/basic_models_comparison.csv'")

# Visualize model comparison
plt.figure(figsize=(14, 10))

# RMSE comparison
plt.subplot(2, 1, 1)
sns.barplot(x='Model', y='Testing RMSE', data=comparison_df)
plt.title('Model Comparison: Testing RMSE (lower is better)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# R² comparison
plt.subplot(2, 1, 2)
sns.barplot(x='Model', y='Testing R²', data=comparison_df)
plt.title('Model Comparison: Testing R² (higher is better)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig('model_artifacts/basic_models_comparison.png')
plt.close()

# ============================================================================
# 9. Summary and Conclusions
# ============================================================================
print("\n9. Summary and Conclusions")
print("-"*80)

# Find the best model based on testing RMSE
best_model_name = comparison_df.iloc[0]['Model']
best_model_rmse = comparison_df.iloc[0]['Testing RMSE']
best_model_r2 = comparison_df.iloc[0]['Testing R²']

print(f"\nBest performing model: {best_model_name}")
print(f"  Testing RMSE: {best_model_rmse:.4f}")
print(f"  Testing R²: {best_model_r2:.4f}")

# Analyze feature importance across models
print("\nConsistent important features across models:")
important_features = set()

# Get top 3 features from each model
for model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet Regression']:
    if model_name == 'Linear Regression':
        coef_df = lr_coef
    elif model_name == 'Ridge Regression':
        coef_df = ridge_coef
    elif model_name == 'Lasso Regression':
        coef_df = lasso_coef
    else:  # ElasticNet
        coef_df = elasticnet_coef
    
    # Get top 3 features by absolute coefficient value
    top_features = coef_df.iloc[:3]['Feature'].tolist()
    important_features.update(top_features)
    
    print(f"  {model_name} top features: {', '.join(top_features)}")

print(f"\nUnion of top features: {', '.join(important_features)}")

print("\nConclusions:")
print("  1. The best performing basic regression model is " + best_model_name)
print("  2. Regularization techniques (Ridge, Lasso, ElasticNet) help prevent overfitting")
print("  3. Polynomial features can capture non-linear relationships")
print("  4. Feature importance analysis reveals consistent important predictors")
print("  5. These models provide a strong baseline for comparison with advanced models")

# Save predictions for ensemble methods
predictions_df = pd.DataFrame({
    'Actual': y_test.values,
    'Linear_Regression': lr_results['y_test_pred'],
    'Ridge_Regression': ridge_results['y_test_pred'],
    'Lasso_Regression': lasso_results['y_test_pred'],
    'ElasticNet_Regression': elasticnet_results['y_test_pred'],
    'Polynomial_Regression_2': poly_results[2]['y_test_pred']
})

predictions_df.to_csv('model_artifacts/basic_models_predictions.csv', index=False)
print("Model predictions saved to 'model_artifacts/basic_models_predictions.csv'")

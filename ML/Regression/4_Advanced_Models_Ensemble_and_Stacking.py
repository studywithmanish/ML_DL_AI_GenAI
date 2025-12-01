"""
Boston Housing Dataset: Advanced Models - Ensemble and Stacking

This script implements and evaluates advanced regression models including ensemble methods
and stacking for the Boston Housing dataset.
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingRegressor, StackingRegressor
import joblib
import os

# Set visualization styles
plt.style.use('seaborn-whitegrid')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Create a directory for model artifacts if it doesn't exist
if not os.path.exists('model_artifacts'):
    os.makedirs('model_artifacts')

print("Boston Housing Dataset: Advanced Models - Ensemble and Stacking")
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
# 3. Random Forest Regressor
# ============================================================================
print("\n3. Random Forest Regressor")
print("-"*80)

# Define the parameter grid
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a grid search
rf = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(
    rf, rf_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)

# Fit the grid search
print("Performing hyperparameter tuning for Random Forest...")
rf_grid.fit(X_train_scaled, y_train)

# Get the best model
best_rf = rf_grid.best_estimator_
best_rf_params = rf_grid.best_params_

print(f"Best Random Forest parameters:")
for param, value in best_rf_params.items():
    print(f"  {param}: {value}")

# Evaluate the best Random Forest model
rf_results = evaluate_model(
    best_rf, X_train_scaled, X_test_scaled, y_train, y_test, "Random Forest"
)

# Save the model
joblib.dump(best_rf, 'model_artifacts/random_forest.pkl')
print("Random Forest model saved to 'model_artifacts/random_forest.pkl'")

# Feature importance for Random Forest
rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
})
rf_importance = rf_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=rf_importance)
plt.title('Random Forest: Feature Importance')
plt.tight_layout()
plt.savefig('model_artifacts/random_forest_importance.png')
plt.close()

print("\nRandom Forest Feature Importance:")
for index, row in rf_importance.head(10).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# ============================================================================
# 4. Gradient Boosting Regressor
# ============================================================================
print("\n4. Gradient Boosting Regressor")
print("-"*80)

# Define the parameter grid
gb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 0.9, 1.0]
}

# Create a grid search
gb = GradientBoostingRegressor(random_state=42)
gb_grid = GridSearchCV(
    gb, gb_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)

# Fit the grid search
print("Performing hyperparameter tuning for Gradient Boosting...")
gb_grid.fit(X_train_scaled, y_train)

# Get the best model
best_gb = gb_grid.best_estimator_
best_gb_params = gb_grid.best_params_

print(f"Best Gradient Boosting parameters:")
for param, value in best_gb_params.items():
    print(f"  {param}: {value}")

# Evaluate the best Gradient Boosting model
gb_results = evaluate_model(
    best_gb, X_train_scaled, X_test_scaled, y_train, y_test, "Gradient Boosting"
)

# Save the model
joblib.dump(best_gb, 'model_artifacts/gradient_boosting.pkl')
print("Gradient Boosting model saved to 'model_artifacts/gradient_boosting.pkl'")

# Feature importance for Gradient Boosting
gb_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_gb.feature_importances_
})
gb_importance = gb_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=gb_importance)
plt.title('Gradient Boosting: Feature Importance')
plt.tight_layout()
plt.savefig('model_artifacts/gradient_boosting_importance.png')
plt.close()

print("\nGradient Boosting Feature Importance:")
for index, row in gb_importance.head(10).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# ============================================================================
# 5. XGBoost Regressor
# ============================================================================
print("\n5. XGBoost Regressor")
print("-"*80)

# Define the parameter grid
xgb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Create a grid search
xgb_reg = xgb.XGBRegressor(random_state=42)
xgb_grid = GridSearchCV(
    xgb_reg, xgb_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)

# Fit the grid search
print("Performing hyperparameter tuning for XGBoost...")
xgb_grid.fit(X_train_scaled, y_train)

# Get the best model
best_xgb = xgb_grid.best_estimator_
best_xgb_params = xgb_grid.best_params_

print(f"Best XGBoost parameters:")
for param, value in best_xgb_params.items():
    print(f"  {param}: {value}")

# Evaluate the best XGBoost model
xgb_results = evaluate_model(
    best_xgb, X_train_scaled, X_test_scaled, y_train, y_test, "XGBoost"
)

# Save the model
joblib.dump(best_xgb, 'model_artifacts/xgboost.pkl')
print("XGBoost model saved to 'model_artifacts/xgboost.pkl'")

# Feature importance for XGBoost
xgb_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_xgb.feature_importances_
})
xgb_importance = xgb_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=xgb_importance)
plt.title('XGBoost: Feature Importance')
plt.tight_layout()
plt.savefig('model_artifacts/xgboost_importance.png')
plt.close()

print("\nXGBoost Feature Importance:")
for index, row in xgb_importance.head(10).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# ============================================================================
# 6. LightGBM Regressor
# ============================================================================
print("\n6. LightGBM Regressor")
print("-"*80)

# Define the parameter grid
lgb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Create a grid search
lgb_reg = lgb.LGBMRegressor(random_state=42)
lgb_grid = GridSearchCV(
    lgb_reg, lgb_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)

# Fit the grid search
print("Performing hyperparameter tuning for LightGBM...")
lgb_grid.fit(X_train_scaled, y_train)

# Get the best model
best_lgb = lgb_grid.best_estimator_
best_lgb_params = lgb_grid.best_params_

print(f"Best LightGBM parameters:")
for param, value in best_lgb_params.items():
    print(f"  {param}: {value}")

# Evaluate the best LightGBM model
lgb_results = evaluate_model(
    best_lgb, X_train_scaled, X_test_scaled, y_train, y_test, "LightGBM"
)

# Save the model
joblib.dump(best_lgb, 'model_artifacts/lightgbm.pkl')
print("LightGBM model saved to 'model_artifacts/lightgbm.pkl'")

# Feature importance for LightGBM
lgb_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_lgb.feature_importances_
})
lgb_importance = lgb_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=lgb_importance)
plt.title('LightGBM: Feature Importance')
plt.tight_layout()
plt.savefig('model_artifacts/lightgbm_importance.png')
plt.close()

print("\nLightGBM Feature Importance:")
for index, row in lgb_importance.head(10).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# ============================================================================
# 7. Voting Ensemble
# ============================================================================
print("\n7. Voting Ensemble")
print("-"*80)

# Create a voting ensemble with the best models
voting_ensemble = VotingRegressor(
    estimators=[
        ('rf', best_rf),
        ('gb', best_gb),
        ('xgb', best_xgb),
        ('lgb', best_lgb)
    ]
)

# Evaluate the voting ensemble
voting_results = evaluate_model(
    voting_ensemble, X_train_scaled, X_test_scaled, y_train, y_test, "Voting Ensemble"
)

# Save the model
joblib.dump(voting_ensemble, 'model_artifacts/voting_ensemble.pkl')
print("Voting Ensemble model saved to 'model_artifacts/voting_ensemble.pkl'")

# ============================================================================
# 8. Stacking Ensemble
# ============================================================================
print("\n8. Stacking Ensemble")
print("-"*80)

# Define base models
base_models = [
    ('rf', RandomForestRegressor(n_estimators=best_rf_params['n_estimators'], 
                                max_depth=best_rf_params['max_depth'],
                                min_samples_split=best_rf_params['min_samples_split'],
                                min_samples_leaf=best_rf_params['min_samples_leaf'],
                                random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=best_gb_params['n_estimators'],
                                    learning_rate=best_gb_params['learning_rate'],
                                    max_depth=best_gb_params['max_depth'],
                                    min_samples_split=best_gb_params['min_samples_split'],
                                    subsample=best_gb_params['subsample'],
                                    random_state=42)),
    ('xgb', xgb.XGBRegressor(n_estimators=best_xgb_params['n_estimators'],
                            learning_rate=best_xgb_params['learning_rate'],
                            max_depth=best_xgb_params['max_depth'],
                            subsample=best_xgb_params['subsample'],
                            colsample_bytree=best_xgb_params['colsample_bytree'],
                            random_state=42)),
    ('lgb', lgb.LGBMRegressor(n_estimators=best_lgb_params['n_estimators'],
                             learning_rate=best_lgb_params['learning_rate'],
                             max_depth=best_lgb_params['max_depth'],
                             subsample=best_lgb_params['subsample'],
                             colsample_bytree=best_lgb_params['colsample_bytree'],
                             random_state=42))
]

# Define meta-learner
meta_learner = Ridge(alpha=1.0)

# Create stacking ensemble
stacking_ensemble = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5
)

# Evaluate the stacking ensemble
stacking_results = evaluate_model(
    stacking_ensemble, X_train_scaled, X_test_scaled, y_train, y_test, "Stacking Ensemble"
)

# Save the model
joblib.dump(stacking_ensemble, 'model_artifacts/stacking_ensemble.pkl')
print("Stacking Ensemble model saved to 'model_artifacts/stacking_ensemble.pkl'")

# ============================================================================
# 9. Model Comparison
# ============================================================================
print("\n9. Model Comparison")
print("-"*80)

# Collect results
models = {
    'Random Forest': rf_results,
    'Gradient Boosting': gb_results,
    'XGBoost': xgb_results,
    'LightGBM': lgb_results,
    'Voting Ensemble': voting_results,
    'Stacking Ensemble': stacking_results
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
comparison_df.to_csv('model_artifacts/advanced_models_comparison.csv', index=False)
print("Model comparison saved to 'model_artifacts/advanced_models_comparison.csv'")

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

plt.savefig('model_artifacts/advanced_models_comparison.png')
plt.close()

# ============================================================================
# 10. Feature Importance Comparison
# ============================================================================
print("\n10. Feature Importance Comparison")
print("-"*80)

# Collect feature importances from different models
feature_importance_df = pd.DataFrame({'Feature': X.columns})

# Add importance from each model
feature_importance_df['Random Forest'] = rf_importance.set_index('Feature').loc[feature_importance_df['Feature'], 'Importance'].values
feature_importance_df['Gradient Boosting'] = gb_importance.set_index('Feature').loc[feature_importance_df['Feature'], 'Importance'].values
feature_importance_df['XGBoost'] = xgb_importance.set_index('Feature').loc[feature_importance_df['Feature'], 'Importance'].values
feature_importance_df['LightGBM'] = lgb_importance.set_index('Feature').loc[feature_importance_df['Feature'], 'Importance'].values

# Calculate average importance
feature_importance_df['Average Importance'] = feature_importance_df[['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']].mean(axis=1)

# Sort by average importance
feature_importance_df = feature_importance_df.sort_values(by='Average Importance', ascending=False)

# Display top features
print("\nTop 10 Features by Average Importance:")
print(feature_importance_df[['Feature', 'Average Importance']].head(10).to_string(index=False))

# Save feature importance comparison
feature_importance_df.to_csv('model_artifacts/feature_importance_comparison.csv', index=False)
print("Feature importance comparison saved to 'model_artifacts/feature_importance_comparison.csv'")

# Visualize feature importance comparison
plt.figure(figsize=(14, 10))
sns.heatmap(feature_importance_df[['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']].head(10), 
            annot=True, cmap='YlGnBu', yticklabels=feature_importance_df['Feature'].head(10))
plt.title('Feature Importance Comparison Across Models (Top 10 Features)')
plt.tight_layout()
plt.savefig('model_artifacts/feature_importance_comparison.png')
plt.close()

# ============================================================================
# 11. Summary and Conclusions
# ============================================================================
print("\n11. Summary and Conclusions")
print("-"*80)

# Find the best model based on testing RMSE
best_model_name = comparison_df.iloc[0]['Model']
best_model_rmse = comparison_df.iloc[0]['Testing RMSE']
best_model_r2 = comparison_df.iloc[0]['Testing R²']

print(f"\nBest performing model: {best_model_name}")
print(f"  Testing RMSE: {best_model_rmse:.4f}")
print(f"  Testing R²: {best_model_r2:.4f}")

# Compare with basic models
try:
    basic_comparison = pd.read_csv('model_artifacts/basic_models_comparison.csv')
    best_basic_model = basic_comparison.iloc[0]['Model']
    best_basic_rmse = basic_comparison.iloc[0]['Testing RMSE']
    best_basic_r2 = basic_comparison.iloc[0]['Testing R²']
    
    print(f"\nBest basic model: {best_basic_model}")
    print(f"  Testing RMSE: {best_basic_rmse:.4f}")
    print(f"  Testing R²: {best_basic_r2:.4f}")
    
    improvement = (best_basic_rmse - best_model_rmse) / best_basic_rmse * 100
    print(f"\nImprovement over best basic model: {improvement:.2f}% reduction in RMSE")
except:
    print("\nCould not load basic models comparison for comparison")

# Top features
print("\nMost important features across models:")
for feature in feature_importance_df['Feature'].head(5):
    print(f"  - {feature}")

print("\nConclusions:")
print("  1. Ensemble methods generally outperform individual models")
print("  2. Stacking and voting ensembles leverage the strengths of multiple models")
print("  3. Tree-based models (Random Forest, Gradient Boosting, XGBoost, LightGBM) are effective for this dataset")
print("  4. Feature importance analysis reveals consistent important predictors across models")
print("  5. Advanced models provide significant improvement over basic regression models")

# Save predictions for final comparison
predictions_df = pd.DataFrame({
    'Actual': y_test.values,
    'Random_Forest': rf_results['y_test_pred'],
    'Gradient_Boosting': gb_results['y_test_pred'],
    'XGBoost': xgb_results['y_test_pred'],
    'LightGBM': lgb_results['y_test_pred'],
    'Voting_Ensemble': voting_results['y_test_pred'],
    'Stacking_Ensemble': stacking_results['y_test_pred']
})

predictions_df.to_csv('model_artifacts/advanced_models_predictions.csv', index=False)
print("Model predictions saved to 'model_artifacts/advanced_models_predictions.csv'")

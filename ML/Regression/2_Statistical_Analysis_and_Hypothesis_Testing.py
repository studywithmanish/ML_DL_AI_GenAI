"""
Boston Housing Dataset: Statistical Analysis and Hypothesis Testing

This script focuses on statistical analysis and hypothesis testing for the Boston Housing dataset.
It includes normality tests, outlier detection, and various statistical tests.
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.graphics.gofplots import qqplot

# Set visualization styles
plt.style.use('seaborn-whitegrid')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Display all columns
pd.set_option('display.max_columns', None)

print("Boston Housing Dataset: Statistical Analysis and Hypothesis Testing")
print("="*80)

# Load the dataset
df = pd.read_csv('bostonHousing.csv')
print(f"Dataset loaded with shape: {df.shape}")

# ============================================================================
# 1. Normality Testing
# ============================================================================
print("\n1. Normality Testing")
print("-"*80)

"""
Hypothesis for Normality Testing:
- H0 (Null Hypothesis): The data follows a normal distribution
- H1 (Alternative Hypothesis): The data does not follow a normal distribution

We'll use multiple tests to check for normality:
1. Shapiro-Wilk Test: Best for small samples (n < 50)
2. D'Agostino-Pearson Test: Tests skewness and kurtosis
3. Kolmogorov-Smirnov Test: Compares with a reference distribution
4. Anderson-Darling Test: More sensitive at the tails than K-S
"""

# Function to perform normality tests
def test_normality(data, column, alpha=0.05):
    """
    Perform multiple normality tests on a given column of data
    
    Parameters:
    data (DataFrame): The dataframe containing the data
    column (str): The column name to test
    alpha (float): Significance level
    
    Returns:
    dict: Results of normality tests
    """
    results = {}
    
    # Extract the data
    x = data[column].dropna()
    
    # 1. Shapiro-Wilk Test
    shapiro_test = stats.shapiro(x)
    results['shapiro'] = {
        'statistic': shapiro_test[0],
        'p-value': shapiro_test[1],
        'normal': shapiro_test[1] > alpha
    }
    
    # 2. D'Agostino-Pearson Test
    k2, p_k2 = stats.normaltest(x)
    results['dagostino'] = {
        'statistic': k2,
        'p-value': p_k2,
        'normal': p_k2 > alpha
    }
    
    # 3. Kolmogorov-Smirnov Test
    ks_test = stats.kstest(x, 'norm', args=(np.mean(x), np.std(x, ddof=1)))
    results['ks'] = {
        'statistic': ks_test[0],
        'p-value': ks_test[1],
        'normal': ks_test[1] > alpha
    }
    
    # 4. Lilliefors Test (variant of K-S)
    try:
        lillie_test = lilliefors(x)
        results['lilliefors'] = {
            'statistic': lillie_test[0],
            'p-value': lillie_test[1],
            'normal': lillie_test[1] > alpha
        }
    except:
        results['lilliefors'] = {
            'statistic': None,
            'p-value': None,
            'normal': None
        }
    
    # 5. Anderson-Darling Test
    ad_test = stats.anderson(x, 'norm')
    results['anderson'] = {
        'statistic': ad_test[0],
        'critical_values': ad_test[1],
        'significance_levels': ad_test[2],
        'normal': ad_test[0] < ad_test[1][2]  # Compare with 5% critical value
    }
    
    return results

# Test normality for each feature
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
normality_results = {}

for column in numeric_columns:
    print(f"\nTesting normality for {column}:")
    results = test_normality(df, column)
    normality_results[column] = results
    
    # Print results in a readable format
    print(f"  Shapiro-Wilk Test: statistic={results['shapiro']['statistic']:.4f}, p-value={results['shapiro']['p-value']:.4e}, normal={results['shapiro']['normal']}")
    print(f"  D'Agostino Test: statistic={results['dagostino']['statistic']:.4f}, p-value={results['dagostino']['p-value']:.4e}, normal={results['dagostino']['normal']}")
    print(f"  Kolmogorov-Smirnov Test: statistic={results['ks']['statistic']:.4f}, p-value={results['ks']['p-value']:.4e}, normal={results['ks']['normal']}")
    
    if results['lilliefors']['statistic'] is not None:
        print(f"  Lilliefors Test: statistic={results['lilliefors']['statistic']:.4f}, p-value={results['lilliefors']['p-value']:.4e}, normal={results['lilliefors']['normal']}")
    
    print(f"  Anderson-Darling Test: statistic={results['anderson']['statistic']:.4f}, normal={results['anderson']['normal']}")
    
    # Visualize the distribution and QQ plot
    plt.figure(figsize=(15, 6))
    
    # Histogram with KDE
    plt.subplot(1, 2, 1)
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    
    # QQ Plot
    plt.subplot(1, 2, 2)
    qqplot(df[column], line='s', ax=plt.gca())
    plt.title(f'QQ Plot of {column}')
    
    plt.tight_layout()
    plt.savefig(f'normality_test_{column}.png')
    plt.close()
    
    # Conclusion for this feature
    is_normal = sum([results[test]['normal'] for test in ['shapiro', 'dagostino', 'ks'] if results[test]['normal'] is not None]) >= 2
    print(f"  Conclusion: {column} {'follows' if is_normal else 'does not follow'} a normal distribution.")

# ============================================================================
# 2. Outlier Detection and Analysis
# ============================================================================
print("\n2. Outlier Detection and Analysis")
print("-"*80)

"""
Methods for outlier detection:
1. Z-Score Method: Identifies values that are far from the mean
2. IQR Method: Identifies values outside the interquartile range
3. Modified Z-Score: More robust to outliers in the data itself
"""

# Function to detect outliers using Z-Score
def detect_outliers_zscore(data, column, threshold=3):
    """
    Detect outliers using Z-Score method
    
    Parameters:
    data (DataFrame): The dataframe containing the data
    column (str): The column name to check
    threshold (float): Z-Score threshold for outliers
    
    Returns:
    DataFrame: Subset of data containing outliers
    """
    z_scores = np.abs(stats.zscore(data[column]))
    outliers = data[z_scores > threshold]
    return outliers, z_scores > threshold

# Function to detect outliers using IQR
def detect_outliers_iqr(data, column, factor=1.5):
    """
    Detect outliers using IQR method
    
    Parameters:
    data (DataFrame): The dataframe containing the data
    column (str): The column name to check
    factor (float): Multiplier for IQR
    
    Returns:
    DataFrame: Subset of data containing outliers
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    is_outlier = (data[column] < lower_bound) | (data[column] > upper_bound)
    return outliers, is_outlier, lower_bound, upper_bound

# Function to detect outliers using Modified Z-Score
def detect_outliers_modified_zscore(data, column, threshold=3.5):
    """
    Detect outliers using Modified Z-Score method
    
    Parameters:
    data (DataFrame): The dataframe containing the data
    column (str): The column name to check
    threshold (float): Modified Z-Score threshold for outliers
    
    Returns:
    DataFrame: Subset of data containing outliers
    """
    median = data[column].median()
    mad = np.median(np.abs(data[column] - median))
    modified_z_scores = 0.6745 * np.abs(data[column] - median) / mad
    outliers = data[modified_z_scores > threshold]
    return outliers, modified_z_scores > threshold

# Detect outliers for each numeric column
for column in numeric_columns:
    print(f"\nOutlier detection for {column}:")
    
    # Z-Score method
    z_outliers, z_is_outlier = detect_outliers_zscore(df, column)
    print(f"  Z-Score method: {len(z_outliers)} outliers detected ({len(z_outliers)/len(df)*100:.2f}%)")
    
    # IQR method
    iqr_outliers, iqr_is_outlier, lower, upper = detect_outliers_iqr(df, column)
    print(f"  IQR method: {len(iqr_outliers)} outliers detected ({len(iqr_outliers)/len(df)*100:.2f}%)")
    print(f"    Lower bound: {lower:.4f}, Upper bound: {upper:.4f}")
    
    # Modified Z-Score method
    mz_outliers, mz_is_outlier = detect_outliers_modified_zscore(df, column)
    print(f"  Modified Z-Score method: {len(mz_outliers)} outliers detected ({len(mz_outliers)/len(df)*100:.2f}%)")
    
    # Visualize outliers
    plt.figure(figsize=(15, 6))
    
    # Box plot
    plt.subplot(1, 3, 1)
    sns.boxplot(y=df[column])
    plt.title(f'Box Plot of {column}')
    
    # Scatter plot with Z-Score outliers
    plt.subplot(1, 3, 2)
    plt.scatter(range(len(df)), df[column], c=['red' if x else 'blue' for x in z_is_outlier], alpha=0.5)
    plt.title(f'Z-Score Outliers in {column}')
    plt.xlabel('Index')
    plt.ylabel(column)
    
    # Scatter plot with IQR outliers
    plt.subplot(1, 3, 3)
    plt.scatter(range(len(df)), df[column], c=['red' if x else 'blue' for x in iqr_is_outlier], alpha=0.5)
    plt.axhline(y=lower, color='green', linestyle='--', label=f'Lower bound: {lower:.2f}')
    plt.axhline(y=upper, color='green', linestyle='--', label=f'Upper bound: {upper:.2f}')
    plt.title(f'IQR Outliers in {column}')
    plt.xlabel('Index')
    plt.ylabel(column)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'outliers_{column}.png')
    plt.close()

# ============================================================================
# 3. Chi-Square Tests and Other Statistical Tests
# ============================================================================
print("\n3. Chi-Square Tests and Other Statistical Tests")
print("-"*80)

"""
Chi-Square Tests:
1. Chi-Square Goodness of Fit: Tests if sample data fits a distribution
2. Chi-Square Test of Independence: Tests if categorical variables are related

Other Statistical Tests:
1. ANOVA: Tests differences among group means
2. Kruskal-Wallis H-test: Non-parametric alternative to ANOVA
"""

# Convert CHAS to categorical for chi-square test
df['CHAS_cat'] = df['CHAS'].astype('category')

# Create bins for MEDV to perform chi-square test of independence
df['MEDV_category'] = pd.qcut(df['MEDV'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

# Chi-Square Test of Independence between CHAS and MEDV_category
contingency_table = pd.crosstab(df['CHAS_cat'], df['MEDV_category'])
print("\nContingency Table (CHAS vs MEDV_category):")
print(contingency_table)

chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Test of Independence Results:")
print(f"  Chi2 statistic: {chi2:.4f}")
print(f"  p-value: {p:.4e}")
print(f"  Degrees of freedom: {dof}")
print(f"  Conclusion: Variables are {'dependent' if p < 0.05 else 'independent'}")

# Visualize the relationship
plt.figure(figsize=(10, 6))
sns.countplot(x='CHAS_cat', hue='MEDV_category', data=df)
plt.title('CHAS vs MEDV Categories')
plt.xlabel('Charles River Dummy (1 if tract bounds river; 0 otherwise)')
plt.ylabel('Count')
plt.legend(title='MEDV Category')
plt.savefig('chi_square_chas_medv.png')
plt.close()

# Create RAD categories for ANOVA
df['RAD_category'] = pd.cut(df['RAD'], bins=3, labels=['Low', 'Medium', 'High'])

# ANOVA: Test if MEDV means differ across RAD categories
rad_groups = [df[df['RAD_category'] == cat]['MEDV'] for cat in df['RAD_category'].unique()]
f_stat, p_value = stats.f_oneway(*rad_groups)

print("\nANOVA Test Results (MEDV across RAD categories):")
print(f"  F-statistic: {f_stat:.4f}")
print(f"  p-value: {p_value:.4e}")
print(f"  Conclusion: {'Reject' if p_value < 0.05 else 'Fail to reject'} null hypothesis")
print(f"  Interpretation: {'There is' if p_value < 0.05 else 'There is no'} significant difference in MEDV means across RAD categories")

# Visualize ANOVA results
plt.figure(figsize=(10, 6))
sns.boxplot(x='RAD_category', y='MEDV', data=df)
plt.title('MEDV across RAD Categories')
plt.xlabel('RAD Category')
plt.ylabel('MEDV')
plt.savefig('anova_rad_medv.png')
plt.close()

# Kruskal-Wallis H-test (non-parametric alternative to ANOVA)
h_stat, h_pvalue = stats.kruskal(*rad_groups)

print("\nKruskal-Wallis H-test Results (MEDV across RAD categories):")
print(f"  H-statistic: {h_stat:.4f}")
print(f"  p-value: {h_pvalue:.4e}")
print(f"  Conclusion: {'Reject' if h_pvalue < 0.05 else 'Fail to reject'} null hypothesis")
print(f"  Interpretation: {'There is' if h_pvalue < 0.05 else 'There is no'} significant difference in MEDV distributions across RAD categories")

# ============================================================================
# 4. Correlation Analysis and Hypothesis Testing
# ============================================================================
print("\n4. Correlation Analysis and Hypothesis Testing")
print("-"*80)

"""
Correlation Tests:
1. Pearson's r: Tests linear correlation (parametric)
2. Spearman's rho: Tests monotonic correlation (non-parametric)
3. Kendall's tau: Another non-parametric correlation test
"""

# Calculate correlation matrix
correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr()

# Test significance of correlations with MEDV
for column in correlation_matrix.index:
    if column != 'MEDV':
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(df[column], df['MEDV'])
        
        # Spearman correlation
        spearman_rho, spearman_p = stats.spearmanr(df[column], df['MEDV'])
        
        # Kendall's tau
        kendall_tau, kendall_p = stats.kendalltau(df[column], df['MEDV'])
        
        print(f"\nCorrelation between {column} and MEDV:")
        print(f"  Pearson's r: {pearson_r:.4f}, p-value: {pearson_p:.4e}, {'significant' if pearson_p < 0.05 else 'not significant'}")
        print(f"  Spearman's rho: {spearman_rho:.4f}, p-value: {spearman_p:.4e}, {'significant' if spearman_p < 0.05 else 'not significant'}")
        print(f"  Kendall's tau: {kendall_tau:.4f}, p-value: {kendall_p:.4e}, {'significant' if kendall_p < 0.05 else 'not significant'}")

# ============================================================================
# 5. Summary of Statistical Analysis
# ============================================================================
print("\n5. Summary of Statistical Analysis")
print("-"*80)

# Count non-normal features
non_normal_features = []
for column in normality_results:
    is_normal = sum([normality_results[column][test]['normal'] for test in ['shapiro', 'dagostino', 'ks'] 
                    if normality_results[column][test]['normal'] is not None]) >= 2
    if not is_normal:
        non_normal_features.append(column)

print(f"\nNormality Testing Summary:")
print(f"  {len(non_normal_features)} out of {len(normality_results)} features do not follow a normal distribution:")
for feature in non_normal_features:
    print(f"    - {feature}")

print("\nOutlier Detection Summary:")
print("  Features with significant outliers (>5% of data points):")
for column in numeric_columns:
    _, iqr_is_outlier, _, _ = detect_outliers_iqr(df, column)
    outlier_percentage = sum(iqr_is_outlier) / len(df) * 100
    if outlier_percentage > 5:
        print(f"    - {column}: {outlier_percentage:.2f}% outliers")

print("\nCorrelation Analysis Summary:")
print("  Features with strong correlation to MEDV (|r| > 0.5):")
for column in correlation_matrix.index:
    if column != 'MEDV' and abs(correlation_matrix.loc[column, 'MEDV']) > 0.5:
        print(f"    - {column}: r = {correlation_matrix.loc[column, 'MEDV']:.4f}")

print("\nStatistical Tests Summary:")
print("  - Chi-Square Test: CHAS and MEDV categories are " + 
      ("dependent" if p < 0.05 else "independent"))
print("  - ANOVA Test: MEDV means across RAD categories are " + 
      ("significantly different" if p_value < 0.05 else "not significantly different"))
print("  - Kruskal-Wallis Test: MEDV distributions across RAD categories are " + 
      ("significantly different" if h_pvalue < 0.05 else "not significantly different"))

print("\nConclusions and Recommendations:")
print("  1. Many features do not follow a normal distribution, suggesting the need for transformations")
print("  2. Several features have significant outliers that may need special handling")
print("  3. Strong correlations exist between certain features and the target variable")
print("  4. Both parametric and non-parametric tests should be considered for further analysis")
print("  5. Feature engineering and transformation may improve model performance")

# Save the processed data for the next steps
df.to_csv('boston_housing_processed.csv', index=False)
print("\nProcessed data saved to 'boston_housing_processed.csv'")

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 22:16:50 2024

@author: piman
"""

#%% Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import statsmodels.api as sm


#%% Load the CSV file
df = pd.read_csv("EC_df_Series.csv")

#%% Group by 'Diag_Year' and calculate the mean for 'Y5_survival'
df_grouped = df.groupby('Diag_Year')['Y5_survival'].agg(['mean', 'count']).reset_index()
df_grouped.rename(columns={'mean': 'percentage'}, inplace=True)
df_grouped['percentage'] = df_grouped['percentage'] * 100

#%% Plot the percentage of Y5_survival by year using regplot with confidence interval
# Prepare the data for the regression model
X_reg = sm.add_constant(df_grouped['Diag_Year'])  # Adds a constant term to the predictor
y_reg = df_grouped['percentage']

# Fit the regression model
model = sm.OLS(y_reg, X_reg).fit()

# Calculate R-squared
r_squared = model.rsquared

# Plot the percentage of Y5_survival by year using regplot with confidence interval
plt.figure(figsize=(10, 6))
sns.regplot(x='Diag_Year', y='percentage', data=df_grouped, marker='o', 
            scatter_kws={'s': 100, 'color': 'green'}, 
            line_kws={"color": "green"})

# Add R-squared to the plot
plt.text(x=2014, y=df_grouped['percentage'].min(), 
         s=f'$R^2 = {r_squared:.2f}$', 
         fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))


plt.xlabel('Diagnosis Year')
plt.ylabel('5 year survival %')
plt.legend()
plt.grid(False)
plt.show()

#%% Define continuous and categorical variables
continuous_vars = ['Age', 'TumorSizeCS', 'InSituMalignantTumorsTotal', 'BenignTumorsTotal', 'SurvivalMonths']
categorical_vars = [col for col in df.columns if col not in continuous_vars and col != 'Diag_Year']

# Group by 'Diag_Year' and calculate mean for continuous variables
continuous_median = df.groupby('Diag_Year')[continuous_vars].median().reset_index()

# Calculate percentage of value == 1 for categorical variables
categorical_percentages = df[categorical_vars].eq(1).groupby(df['Diag_Year']).mean().reset_index()

# Merge the continuous means and categorical percentages
merge_table = pd.merge(continuous_median, categorical_percentages, on='Diag_Year')

#%% Standard scaling
scaler = StandardScaler()
scaled_values = scaler.fit_transform(merge_table.drop(columns=['Diag_Year']))

# Create a new DataFrame with the scaled values
scaled_df = pd.DataFrame(scaled_values, columns=merge_table.columns.drop('Diag_Year'))
scaled_df.insert(0, 'Diag_Year', merge_table['Diag_Year'])

#%% Feature selection using variance threshold
# Set a variance threshold 0.1
selector = VarianceThreshold(threshold=0.1)
selected_features = selector.fit_transform(scaled_df.drop(columns=['Diag_Year', 'Y5_survival']))

# Get the names of the selected features
support = selector.get_support()
selected_feature_names = scaled_df.drop(columns=['Diag_Year', 'Y5_survival']).columns[support]

# Create a new DataFrame with the selected features
final_df = pd.DataFrame(selected_features, columns=selected_feature_names)
final_df.insert(0, 'Y5_survival', scaled_df['Y5_survival'])

#%% Correlation analysis
correlation_matrix = final_df.corr()

def custom_color_map(value):
    if value < -0.8:
        return '#4682B4'
    elif value > 0.8:
        return '#B22222'
    else:
        return '#C0C0C0'

# Apply the custom color map function to the correlation matrix
custom_cmap = mcolors.ListedColormap(['#4682B4', '#C0C0C0', '#B22222'])
bounds = [-1, -0.8, 0.8, 1]
norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Plot the correlation matrix with the custom color map
plt.figure(figsize=(15, 13))
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap=custom_cmap, norm=norm, cbar_kws={"shrink": .5}, linewidths=.6, annot_kws={"size": 10})
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.title('Correlation Matrix')
plt.show()

#%% RFE to address multicollinearity
# Define the features and the target variable
X = final_df.drop(columns=['Y5_survival', 'SurvivalMonths'])
y = final_df['Y5_survival']

model = Ridge(alpha=0.1) 

# Apply RFE
n_features_to_select = 15  # Number of features to select, adjust based on your needs
rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
X_rfe = rfe.fit_transform(X, y)

# Get the names of the selected features
selected_rfe_features = X.columns[rfe.support_]

# Create a new DataFrame with the selected features based on RFE
final_df_rfe = pd.DataFrame(X_rfe, columns=selected_rfe_features)
final_df_rfe.insert(0, 'Y5_survival', y)

#%% Correlation analysis
correlation_matrix_rfe = final_df_rfe.drop(columns=['Y5_survival']).corr()

# Apply the custom color map function to the correlation matrix
custom_cmap = mcolors.ListedColormap(['#4682B4', '#C0C0C0', '#B22222'])
bounds = [-1, -0.8, 0.8, 1]
norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)

mask = np.triu(np.ones_like(correlation_matrix_rfe, dtype=bool))

# Plot the correlation matrix with the custom color map
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix_rfe, mask=mask, annot=False, cmap=custom_cmap, norm=norm, cbar_kws={"shrink": .5}, linewidths=.6, annot_kws={"size": 10})
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('Correlation Matrix')
plt.show()

#%% Correlation analysis
correlation_matrix_rfe_cut = final_df_rfe.drop(columns=['Y5_survival', 'SurgicalTreatment_Local_excision', 'ExtensionCS_200', 'LymphNodesCS_610']).corr()

# Apply the custom color map function to the correlation matrix
custom_cmap = mcolors.ListedColormap(['#4682B4', '#C0C0C0', '#B22222'])
bounds = [-1, -0.8, 0.8, 1]
norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)

mask = np.triu(np.ones_like(correlation_matrix_rfe_cut, dtype=bool))

# Plot the correlation matrix with the custom color map
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix_rfe_cut, mask=mask, annot=False, cmap=custom_cmap, norm=norm, cbar_kws={"shrink": .5}, linewidths=.6, annot_kws={"size": 10})
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('Correlation Matrix')
plt.show()

#%% Regression analysis using cross-validation
# Define the features and the target variable
X = final_df_rfe.drop(columns=['Y5_survival', 'SurgicalTreatment_Local_excision', 'ExtensionCS_200', 'LymphNodesCS_610'])
y = final_df_rfe['Y5_survival']

# Initialize the linear regression model
model = LinearRegression()

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
y_pred = cross_val_predict(model, X, y, cv=5)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print(f'Cross-Validated R^2 Scores: {scores}')
print(f'Mean Cross-Validated R^2 Score: {scores.mean()}')

# Plot predicted vs actual values
plt.figure(figsize=(7, 6))
plt.scatter(y, y_pred, edgecolors=(0, 0, 0), s=200, color='darkmagenta')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Y5 Survival')
plt.show()

#%% Extracting feature importances
model.fit(X, y)
feature_importances = model.coef_

# Verify that the number of selected features matches the number of coefficients
if len(X.columns) != len(feature_importances):
    raise ValueError("The number of selected features does not match the number of coefficients")

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances with custom colors
plt.figure(figsize=(7, 8))
palette = sns.color_palette("RdBu", len(importance_df))  # Custom color palette
sns.barplot(x='Importance', y='Feature', data=importance_df, palette=palette)
plt.title('Feature Importances')
plt.show()

importance_df


#%% Plot dot plot for positive correlated variables
# Function to format y-tick labels to 2 decimal places
def format_fn(tick_val, tick_pos):
    return f'{tick_val:.1f}'

fig, axes = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

variables_positive = ['MetsAtDXCS_0', 'Race_Other', 'Grade_Unknown']

for i, var in enumerate(variables_positive):
    sns.regplot(x='Diag_Year', y=var, data=scaled_df, ax=axes[i], marker='x', scatter_kws={'s': 200, 'color': 'red'}, line_kws={"color": "red"}, label=var)
    sns.regplot(x='Diag_Year', y='Y5_survival', data=scaled_df, ax=axes[i], marker='o', scatter_kws={'s': 200, 'color': 'green'}, line_kws={"color": "green"}, label='Y5_survival')
    axes[i].set_ylabel(f'Mean {var}', fontsize=18)
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', labelsize=18)
    axes[i].tick_params(axis='y', labelsize=18)
    axes[i].yaxis.set_major_formatter(FuncFormatter(format_fn))
    axes[i].set_ylim(-2, 2)
    axes[i].grid(False)
    axes[i].legend(fontsize=18)

plt.tight_layout()
plt.show()

#%% Plot dot plot for negative correlated variables
fig, axes = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

variables_negative = ['HistologicBehavior_Other', 'HistologicBehavior_SRC_carcinoma', 'ExtensionCS_200']

for i, var in enumerate(variables_negative):
    sns.regplot(x='Diag_Year', y=var, data=scaled_df, ax=axes[i], marker='x', scatter_kws={'s': 200}, line_kws={"color": "blue"}, label=var)
    sns.regplot(x='Diag_Year', y='Y5_survival', data=scaled_df, ax=axes[i], marker='o', scatter_kws={'s': 200, 'color': 'green'}, line_kws={"color": "green"}, label='Y5_survival')
    axes[i].set_ylabel(f'Mean {var}', fontsize=18)
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', labelsize=18)
    axes[i].tick_params(axis='y', labelsize=18)
    axes[i].yaxis.set_major_formatter(FuncFormatter(format_fn))
    axes[i].set_ylim(-2, 2)
    axes[i].grid(False)
    axes[i].legend(fontsize=18)
    axes[i].legend(loc='lower center', fontsize=18)

plt.tight_layout()
plt.show()

#%% Determine correlations with Y5_survival (Pearson correlation coefficients)
variables_of_interest = ['MetsAtDXCS_0', 'Race_Other', 'Grade_Unknown','HistologicBehavior_Other', 'HistologicBehavior_SRC_carcinoma', 'ExtensionCS_200', 'Y5_survival']
correlation_matrix_linear = scaled_df[variables_of_interest].corr()
correlations_with_Y5_survival = correlation_matrix_linear['Y5_survival'].drop('Y5_survival')
print(correlations_with_Y5_survival)


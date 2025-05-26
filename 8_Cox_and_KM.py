# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 02:07:42 2024

@author: piman
"""

#%% Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import seaborn as sns
from scipy.stats import mannwhitneyu

#%% Set global font size
plt.rcParams.update({'font.size': 14})

#%% Load the CSV file
df = pd.read_csv('EC_df_Series.csv')


#%% Assuming all patients are dead, set the 'event' variable to 1
df['Event'] = 1

# Extract the survival time and event columns
survival_time = df['SurvivalMonths']
event = df['Event']

# Calculate the medians for Age and TumorSizeCS
age_median = df['Age'].median()
tumor_size_median = df['TumorSizeCS'].median()

# Create categorical variables based on the medians
df['Age_Cat'] = pd.cut(df['Age'], bins=[-float('inf'), age_median, float('inf')], labels=['Below_Median_Age', 'Above_Median_Age'])
df['TumorSizeCS_Cat'] = pd.cut(df['TumorSizeCS'], bins=[-float('inf'), tumor_size_median, float('inf')], labels=['Below_Median_TumorSize', 'Above_Median_TumorSize'])

#%% Cox Regression Analysis
# Prepare the data for Cox regression
cox_df = df[['SurvivalMonths', 'Event', 'SurgicalTreatment_No_surgery', 'SummaryStage_Dist+LN', 'TumorSizeCS_Cat', 'MetsAtDXCS_0', 'AJCC_M_M1b', 'Age_Cat']]

# Convert categorical variables to dummy variables
cox_df = pd.get_dummies(cox_df, columns=['SurgicalTreatment_No_surgery', 'SummaryStage_Dist+LN', 'TumorSizeCS_Cat', 'MetsAtDXCS_0', 'AJCC_M_M1b', 'Age_Cat'], drop_first=True)

# Fit the Cox proportional hazards model
cph = CoxPHFitter()
cph.fit(cox_df, duration_col='SurvivalMonths', event_col='Event')

# Print the summary
cph.print_summary()

plt.figure(figsize=(7, 5))

# Plot the coefficients
cph.plot()

plt.title('Cox Proportional Hazards Model Coefficients')
plt.show()

#%% Kaplan-Meier Plots in Subplots
kmf = KaplanMeierFitter()

variables = ['SurgicalTreatment_No_surgery', 'SummaryStage_Dist+LN', 'TumorSizeCS_group', 'MetsAtDXCS_0', 'AJCC_M_M1b', 'Age_group']

# Create groupings for continuous variables using quantiles
df['TumorSizeCS_group'] = pd.qcut(df['TumorSizeCS'], q=[0, 0.25, 0.75, 1], labels=['<= 25th percentile', '25th-75th percentile', '> 75th percentile'])
df['Age_group'] = pd.qcut(df['Age'], q=[0, 0.25, 0.75, 1], labels=['<= 25th percentile', '25th-75th percentile', '> 75th percentile'])

# Define colors for categorical variables
categorical_colors = {0: '#2980B9', 1: '#DC7633'}

# Define colors for continuous variables
continuous_colors = {'<= 25th percentile': '#2980B9', '25th-75th percentile': 'grey', '> 75th percentile': '#DC7633'}

num_vars = len(variables)
cols = 2  
rows = (num_vars + 1) // cols  

fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
axes = axes.flatten()

for i, var in enumerate(variables):
    ax = axes[i]
    for label in df[var].unique():
        mask = df[var] == label
        kmf.fit(survival_time[mask], event_observed=event[mask], label=f'{var}: {label}')
        if var in ['TumorSizeCS_group', 'Age_group']:
            color = continuous_colors[label]
        else:
            color = categorical_colors[int(label)]
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)
    
    # Calculate log-rank test p-value
    groups = df[var].unique()
    if len(groups) == 2:
        mask1 = df[var] == groups[0]
        mask2 = df[var] == groups[1]
        results = logrank_test(survival_time[mask1], survival_time[mask2], event_observed_A=event[mask1], event_observed_B=event[mask2])
        p_value = results.p_value
        ax.set_title(f'KM of {var}\n(log-rank p-value = {p_value:.3e})')
    else:
        results = multivariate_logrank_test(survival_time, df[var], event)
        p_value = results.p_value
        ax.set_title(f'KM of {var}\n(multivariate log-rank p-value = {p_value:.3e})')
    
    ax.set_xlabel('Survival Months')
    ax.set_ylabel('Survival Probability')
    ax.legend(title=var)
    ax.legend_.remove() # Hide legend
    ax.set_xlim(0, 60)
    ax.grid(False)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

#%% Box Plots with Statistical Difference

def plot_boxplots(df, variables, max_points=100, strip_alpha=0.7, strip_width=0.5):
    num_vars = len(variables)
    cols = 2  # Number of columns for subplots
    rows = (num_vars + 1) // cols  # Calculate number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
    axes = axes.flatten()

    for i, var in enumerate(variables):
        ax = axes[i]
        if var in ['SurgicalTreatment_No_surgery', 'SummaryStage_Dist+LN', 'MetsAtDXCS_0', 'AJCC_M_M1b']:
            colors = [categorical_colors[int(value)] for value in df[var].unique()]
            boxplot = sns.boxplot(x=var, y='SurvivalMonths', data=df, ax=ax, showfliers=False, palette=colors, width=0.4)
            df_sample = df.sample(min(len(df), max_points))
            sns.stripplot(x=var, y='SurvivalMonths', data=df_sample, ax=ax, palette=colors, alpha=strip_alpha, jitter=strip_width, size=8)
            grouped = df.groupby(var)['SurvivalMonths']
            groups = [group for name, group in grouped]
            if len(groups) == 2:
                stat, p_value = mannwhitneyu(groups[0], groups[1])
                ax.set_title(f'Box Plot of {var} vs Survival Months\n(p-value = {p_value:.3e})')
            else:
                ax.set_title(f'Box Plot of {var} vs Survival Months\n(More than 2 groups)')
        else:  # Continuous variables with quantile grouping
            if var in ['TumorSizeCS', 'Age']:
                group_var = f'{var}_group'
                grouped = df.groupby(group_var)['SurvivalMonths']

                # Perform Mann-Whitney U test between > 75th percentile and <= 25th percentile
                group1 = grouped.get_group('<= 25th percentile')
                group2 = grouped.get_group('> 75th percentile')
                stat, p_value = mannwhitneyu(group1, group2)

                boxplot = sns.boxplot(x=group_var, y='SurvivalMonths', data=df, ax=ax, showfliers=False, order=['<= 25th percentile', '25th-75th percentile', '> 75th percentile'], palette=[continuous_colors['<= 25th percentile'], continuous_colors['25th-75th percentile'], continuous_colors['> 75th percentile']], width=0.4)
                df_sample = df.sample(min(len(df), max_points))
                sns.stripplot(x=group_var, y='SurvivalMonths', data=df_sample, ax=ax, palette=[continuous_colors['<= 25th percentile'], continuous_colors['25th-75th percentile'], continuous_colors['> 75th percentile']], alpha=strip_alpha, jitter=strip_width, size=8, order=['<= 25th percentile', '25th-75th percentile', '> 75th percentile'])
                ax.set_title(f'Box Plot of {var} vs Survival Months\n(p-value = {p_value:.3e})')
            else:
                median_value = df[var].median()
                df[f'{var}_group'] = df[var].apply(lambda x: f'<= {median_value}' if x <= median_value else f'> {median_value}')
                grouped = df.groupby(f'{var}_group')['SurvivalMonths']

                # Perform Mann-Whitney U test
                group1 = grouped.get_group(f'<= {median_value}')
                group2 = grouped.get_group(f'> {median_value}')
                stat, p_value = mannwhitneyu(group1, group2)

                colors = ['blue', 'red']
                boxplot = sns.boxplot(x=f'{var}_group', y='SurvivalMonths', data=df, ax=ax, showfliers=False, palette=colors, width=0.4)
                df_sample = df.sample(min(len(df), max_points))
                sns.stripplot(x=f'{var}_group', y='SurvivalMonths', data=df_sample, ax=ax, palette=colors, alpha=strip_alpha, jitter=strip_width, size=8)
                ax.set_title(f'Box Plot of {var} vs Survival Months\n(p-value = {p_value:.3e})')

        ax.set_xlabel(var)
        ax.set_ylabel('Survival Months')
        ax.grid(False)
        ax.set_yscale('log')

        # Add median line
        medians = df.groupby(var)['SurvivalMonths'].median()
        for xtick, label in enumerate(ax.get_xticklabels()):
            if label.get_text() in medians.index:
                ax.text(xtick, medians.loc[label.get_text()], f'{medians.loc[label.get_text()]:.2f}', horizontalalignment='center', size='medium', color='black', weight='semibold')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# List of variables for which to create box plots
variables = ['SurgicalTreatment_No_surgery', 'SummaryStage_Dist+LN', 'TumorSizeCS', 'MetsAtDXCS_0', 'AJCC_M_M1b', 'Age']

# Plot box plots for each variable
plot_boxplots(df, variables, max_points=500, strip_alpha=0.3, strip_width=0.3)







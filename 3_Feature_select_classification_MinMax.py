# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 21:21:16 2024

@author: piman
"""

#%% Import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

#%% Import EC_df
EC_df = pd.read_csv('EC_df.csv')

#%% Drop the first column
EC_df = EC_df.drop(columns=['Unnamed: 0'])

#%% Set the "ID" column as the index
EC_df = EC_df.set_index('ID')

#%% Create Y5_survival column
EC_df['Y5_survival'] = EC_df['SurvivalMonths'].apply(lambda x: 1 if x >= 60 else 0)

#%% Drop Diag_Year and SurvivalMonths columns
EC_df = EC_df.drop(columns=['Diag_Year', 'SurvivalMonths'])

#%% Classify variables
continuous_vars = [
    'Age', 'TumorSizeCS', 'InSituMalignantTumorsTotal', 'BenignTumorsTotal'
]
nominal_vars = [
    'Race', 'Income', 'PrimarySite_code', 'Grade', 'SummaryStage', 
    'HistologicBehavior', 'AJCC_T', 'AJCC_N', 'AJCC_M', 'ExtensionCS', 'LymphNodesCS', 
    'MetsAtDXCS','SurgicalTreatment', 'SurgicalRadiationSequence'
]
binary_vars = [
    'Sex', 'Hispanic', 'MaritalStatus', 'SequenceNumber', 'TumorSizeCS_polyposis',
    'ScopeRegionalLNSurgery', 'Radiation', 'Chemotherapy'
]

#%% Min-max normalization of the continuous variables
scaler = MinMaxScaler()
EC_df[continuous_vars] = scaler.fit_transform(EC_df[continuous_vars])

#%% One-hot encoding the nominal variables using pd.get_dummies
EC_df = pd.get_dummies(EC_df, columns=nominal_vars, drop_first=False)

#%% Convert boolean values to integers
EC_df = EC_df.applymap(lambda x: 1 if x is True else (0 if x is False else x))

#%% Save EC_df as a CSV file
EC_df.to_csv('EC_df_unmodified.csv', index=True)

#%% Separate features and target variable
X = EC_df.drop(columns=['Y5_survival'])
y = EC_df['Y5_survival']

#%% Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#%% Handle class imbalance using SMOTE on the training set
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

#%% Train an XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_res, y_res)

#%% Make predictions and evaluate the model
y_pred = xgb_model.predict(X_test)

# Performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Matthews Correlation Coefficient (MCC): {mcc}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

#%% Feature importance
feature_importance = xgb_model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()

#%% Create EC_df_median and EC_df_zero DataFrames
median_importance = importance_df['Importance'].median()

# Select features with importance higher than the median
important_features_median = importance_df[importance_df['Importance'] > median_importance]['Feature']
EC_df_median = EC_df[important_features_median]
EC_df_median['Y5_survival'] = y  # Add the dependent variable

#%% Select features with importance greater than zero
important_features_zero = importance_df[importance_df['Importance'] > 0]['Feature']
EC_df_zero = EC_df[important_features_zero]
EC_df_zero['Y5_survival'] = y  # Add the dependent variable

#%% Create EC_df_25th
percentile_25th_importance = importance_df['Importance'].quantile(0.25)

# Select features with importance higher than the 25th percentile
important_features_25th = importance_df[importance_df['Importance'] > percentile_25th_importance]['Feature']
EC_df_25th = EC_df[important_features_25th]
EC_df_25th['Y5_survival'] = y  # Add the dependent variable

#%% Save EC_df_25th as a CSV file
EC_df_25th.to_csv('EC_df_25th.csv', index=True)

#%% Save EC_df_median as a CSV file
EC_df_median.to_csv('EC_df_median.csv', index=True)


#%% Save EC_df_zero as a CSV file
EC_df_zero.to_csv('EC_df_zero.csv', index=True)


#%% Function to train and evaluate XGBoost model
def evaluate_model(X, y):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle class imbalance using SMOTE on the training set
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    # Train an XGBoost model
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_res, y_res)
    
    # Make predictions and evaluate the model
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc}")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

#%% Evaluate the model on EC_df_median
print("Performance on EC_df_median:")
evaluate_model(EC_df_median.drop(columns=['Y5_survival']), EC_df_median['Y5_survival'])

#%% Evaluate the model on EC_df_zero
print("Performance on EC_df_zero:")
evaluate_model(EC_df_zero.drop(columns=['Y5_survival']), EC_df_zero['Y5_survival'])

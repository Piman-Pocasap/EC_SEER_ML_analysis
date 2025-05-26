# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:12:30 2024

@author: asus
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
Survival = EC_df['SurvivalMonths']
EC_df = EC_df.drop(columns=['SurvivalMonths'])

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

#%% Separate features and target variable
Year = EC_df['Diag_Year']
X = EC_df.drop(columns=['Diag_Year', 'Y5_survival'])
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

#%% Create EC_df_series
median_importance = importance_df['Importance'].median()

# Select features with importance higher than the median
important_features_median = importance_df[importance_df['Importance'] > median_importance]['Feature']
EC_df_series = EC_df[important_features_median]
EC_df_series['Y5_survival'] = y 
EC_df_series['Diag_Year'] = Year 
EC_df_series['SurvivalMonths'] = Survival

#%% Save EC_df_series as a CSV file
EC_df_series.to_csv('EC_df_series.csv', index=True)



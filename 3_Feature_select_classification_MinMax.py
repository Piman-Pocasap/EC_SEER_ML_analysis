# -*- coding: utf-8 -*-
#%%
"""
Revised: 27/03/2026

Main point: Move normalization and one hot encoder to after data splitting

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
    'MetsAtDXCS', 'SurgicalTreatment', 'SurgicalRadiationSequence'
]
binary_vars = [
    'Sex', 'Hispanic', 'MaritalStatus', 'SequenceNumber', 'TumorSizeCS_polyposis',
    'ScopeRegionalLNSurgery', 'Radiation', 'Chemotherapy'
]

#%% Separate features and target variable (before preprocessing to avoid leakage)
X_raw = EC_df.drop(columns=['Y5_survival'])
y = EC_df['Y5_survival']

#%% Split the dataset into training and testing sets (preprocessing happens AFTER split)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

#%% Fit preprocessing on the training set only, then transform train/test/full data
scaler = MinMaxScaler()
scaler.fit(X_train_raw[continuous_vars])

def _transform_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[continuous_vars] = scaler.transform(df[continuous_vars])
    df = pd.get_dummies(df, columns=nominal_vars, drop_first=False)
    df = df.applymap(lambda x: 1 if x is True else (0 if x is False else x))
    return df

X_train = _transform_features(X_train_raw)
X_test = _transform_features(X_test_raw)

# Align test columns to train columns (unknown categories in test -> all zeros)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

#%% Create a fully transformed dataset using training-fitted preprocessing (for feature selection exports)
X = _transform_features(X_raw).reindex(columns=X_train.columns, fill_value=0)
EC_df_processed = X.join(y)

#%% Save processed EC_df as a CSV file
EC_df_processed.to_csv('EC_df_unmodified.csv', index=True)

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
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#%% Feature Importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()

#%% Save importance_df as a CSV file
importance_df.to_csv('feature_importance.csv', index=False)

#%% Create datasets based on importance thresholds
median_importance = importance_df['Importance'].median()
percentile_25_importance = importance_df['Importance'].quantile(0.25)

# Create datasets
median_dataset = X[importance_df[importance_df['Importance'] > median_importance]['Feature']]
median_dataset['Y5_survival'] = y
median_dataset.to_csv('EC_df_median.csv', index=True)

greater_than_zero_dataset = X[importance_df[importance_df['Importance'] > 0]['Feature']]
greater_than_zero_dataset['Y5_survival'] = y

percentile_25_dataset = X[importance_df[importance_df['Importance'] > percentile_25_importance]['Feature']]
percentile_25_dataset['Y5_survival'] = y

#%% Model evaluation function
def evaluate_model(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_res, y_res)

    y_pred = xgb_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"--- {dataset_name} ---")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.show()

#%% Evaluate models on median and >0 importance datasets
evaluate_model(median_dataset.drop(columns=['Y5_survival']), median_dataset['Y5_survival'], "Median Importance Dataset")
evaluate_model(greater_than_zero_dataset.drop(columns=['Y5_survival']), greater_than_zero_dataset['Y5_survival'], ">0 Importance Dataset")

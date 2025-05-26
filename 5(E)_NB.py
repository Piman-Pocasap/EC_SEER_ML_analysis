# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:13:29 2024

@author: asus
"""

#%% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             precision_recall_curve, f1_score, recall_score,
                             precision_score, matthews_corrcoef, roc_auc_score, accuracy_score)
from imblearn.over_sampling import SMOTE

#%% Load the data from EC_df_median.csv
EC_df_median = pd.read_csv('EC_df_median.csv', index_col='ID')

#%% Separate features and target variable
X = EC_df_median.drop(columns=['Y5_survival'])
y = EC_df_median['Y5_survival']

#%% Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=56, stratify=y)

#%% Handle class imbalance using SMOTE on the training set
smote = SMOTE(random_state=56)
X_res, y_res = smote.fit_resample(X_train, y_train)

#%% Initialize Naive Bayes classifier with default parameters
nb = GaussianNB()

#%% Perform 5-fold cross-validation on the resampled training set
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=56)
cv_scores = cross_val_score(nb, X_res, y_res, cv=cv, scoring='accuracy')

#%% Print the cross-validation results
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean CV Accuracy: {np.mean(cv_scores)}')
print(f'Standard Deviation of CV Accuracy: {np.std(cv_scores)}')

#%% Train the final Naive Bayes model on the resampled training set
nb.fit(X_res, y_res)

#%% Evaluate the model on the training set
y_train_pred = nb.predict(X_res)

#%% Evaluate the model on the test set
y_test_pred = nb.predict(X_test)

#%% Evaluation Metrics for Test Set
print("Test Set Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Test Set Classification Report:\n", classification_report(y_test, y_test_pred))
print("Test Set ROC AUC Score:", roc_auc_score(y_test, y_test_pred))
print("Test Set Precision Score:", precision_score(y_test, y_test_pred))
print("Test Set Recall Score:", recall_score(y_test, y_test_pred))
print("Test Set F1 Score:", f1_score(y_test, y_test_pred))
print("Test Set Matthews Correlation Coefficient:", matthews_corrcoef(y_test, y_test_pred))

#%% Evaluation Metrics for Training Set
print("Training Set Confusion Matrix:\n", confusion_matrix(y_res, y_train_pred))
print("Training Set Classification Report:\n", classification_report(y_res, y_train_pred))
print("Training Set ROC AUC Score:", roc_auc_score(y_res, y_train_pred))
print("Training Set Precision Score:", precision_score(y_res, y_train_pred))
print("Training Set Recall Score:", recall_score(y_res, y_train_pred))
print("Training Set F1 Score:", f1_score(y_res, y_train_pred))
print("Training Set Matthews Correlation Coefficient:", matthews_corrcoef(y_res, y_train_pred))

#%% Plot Confusion Matrix for Test Set
cm_test = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Test Set')
plt.show()

#%% Plot Confusion Matrix for Training Set
cm_train = confusion_matrix(y_res, y_train_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Training Set')
plt.show()

#%% Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, nb.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#%% Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, nb.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(recall, precision, lw=2, color='b', label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower left")
plt.show()

#%% Feature importance using means and variances
means = nb.theta_
variances = nb.var_

# Calculate the importance as the inverse of variance (1/variance)
importance = 1 / variances

# Plot feature importance
plt.figure(figsize=(15, 10))
plt.barh(X.columns, importance[0], align="center")
plt.xlabel('Importance')
plt.title('Feature Importance Based on Naive Bayes')
plt.show()

# Generate table for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance[0]
}).sort_values(by='Importance', ascending=False)
print(feature_importance_df)

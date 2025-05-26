# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:12:21 2024

@author: piman
"""

#%% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             precision_recall_curve, f1_score, recall_score,
                             precision_score, matthews_corrcoef, roc_auc_score, accuracy_score)
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

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

#%% Initialize Random Forest classifier with regularization parameters
rf = RandomForestClassifier(
    random_state=56,
    n_estimators=100,           # Number of trees
    max_depth=10,               # Limit the depth of the trees
    min_samples_split=10,       # Minimum number of samples required to split an internal node
    min_samples_leaf=5          # Minimum number of samples required to be at a leaf node
)

#%% Perform 5-fold cross-validation on the resampled training set
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=56)
cv_scores = cross_val_score(rf, X_res, y_res, cv=cv, scoring='accuracy')

#%% Print the cross-validation results
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean CV Accuracy: {np.mean(cv_scores)}')
print(f'Standard Deviation of CV Accuracy: {np.std(cv_scores)}')

#%% Train the final model on the resampled training set
rf.fit(X_res, y_res)

#%% Evaluate the model on the training set
y_train_pred = rf.predict(X_res)

#%% Evaluate the model on the test set
y_test_pred = rf.predict(X_test)

#%% Evaluation Metrics for Test Set
print("Test Set Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Test Set Classification Report:\n", classification_report(y_test, y_test_pred))
print("Test Set ROC AUC Score:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))
print("Test Set Precision Score:", precision_score(y_test, y_test_pred))
print("Test Set Recall Score:", recall_score(y_test, y_test_pred))
print("Test Set F1 Score:", f1_score(y_test, y_test_pred))
print("Test Set Matthews Correlation Coefficient:", matthews_corrcoef(y_test, y_test_pred))

#%% Evaluation Metrics for Training Set
print("Training Set Confusion Matrix:\n", confusion_matrix(y_res, y_train_pred))
print("Training Set Classification Report:\n", classification_report(y_res, y_train_pred))
print("Training Set ROC AUC Score:", roc_auc_score(y_res, rf.predict_proba(X_res)[:, 1]))
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
fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
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
precision, recall, _ = precision_recall_curve(y_test, rf.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(recall, precision, lw=2, color='b', label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower left")
plt.show()


#%% Plot Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

# Calculate the median of the feature importances
median_importance = np.median(importances)

# Filter the features and importances to include only those greater than the median
filtered_indices = indices[importances[indices] > median_importance]
filtered_importances = importances[filtered_indices]
filtered_feature_names = feature_names[filtered_indices]

# Define a function to map importance values to colors
def importance_to_color(importance):
    if importance > 0.05:
        return 'teal'
    else:
        return 'darkgrey'

# Apply the function to get colors
colors = [importance_to_color(importance) for importance in filtered_importances]

plt.figure(figsize=(10, 10))
plt.title("Feature Importances")
bars = plt.barh(range(len(filtered_importances)), filtered_importances, color=colors, align="center")
plt.yticks(range(len(filtered_importances)), filtered_feature_names, rotation=0)  # Rotate y-ticks to horizontal
plt.ylim([-1, len(filtered_importances)])

# Invert y-axis to have the highest values at the top
plt.gca().invert_yaxis()

# Add a vertical line at the importance threshold of 0.05
threshold = 0.05
plt.axvline(x=threshold, color='darkred', linestyle='--')

plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

#%% Generate table for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': filtered_feature_names,
    'Importance': filtered_importances
})
print(feature_importance_df)

#%% Perform PCA
pca = PCA(n_components=3)  # set PC == 3
X_test_pca = pca.fit_transform(X_test)

#%% Create a DataFrame with PCA results and Y5_survival labels
pca_df = pd.DataFrame(data=X_test_pca, columns=['PC1', 'PC2', 'PC3'])
pca_df['Y5_survival'] = y_test.values

#%% Plot the PCA results with custom colors for Y5_survival
fig, axes = plt.subplots(1, 3, figsize=(30, 7))
sns.scatterplot(x='PC1', y='PC2', hue='Y5_survival', data=pca_df, palette={1: 'salmon', 0: 'teal'}, s=100, alpha=0.7, ax=axes[0])
axes[0].set_title('PCA of Test Set (PC1 vs PC2)')
axes[0].set_xlabel('Principal Component 1')
axes[0].set_ylabel('Principal Component 2')
axes[0].legend(title='Y5_survival', loc='best')

sns.scatterplot(x='PC1', y='PC3', hue='Y5_survival', data=pca_df, palette={1: 'salmon', 0: 'teal'}, s=100, alpha=0.7, ax=axes[1])
axes[1].set_title('PCA of Test Set (PC1 vs PC3)')
axes[1].set_xlabel('Principal Component 1')
axes[1].set_ylabel('Principal Component 3')
axes[1].legend(title='Y5_survival', loc='best')

sns.scatterplot(x='PC2', y='PC3', hue='Y5_survival', data=pca_df, palette={1: 'salmon', 0: 'teal'}, s=100, alpha=0.7, ax=axes[2])
axes[2].set_title('PCA of Test Set (PC2 vs PC3)')
axes[2].set_xlabel('Principal Component 2')
axes[2].set_ylabel('Principal Component 3')
axes[2].legend(title='Y5_survival', loc='best')

#%% Get the model's predictions on the test set
y_test_pred_prob = rf.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_pred_prob > 0.5).astype(int)

#%% Determine if predictions are correct
pca_df['Prediction'] = y_test_pred
pca_df['Correct'] = np.where(pca_df['Y5_survival'] == pca_df['Prediction'], 'Correct', 'Incorrect')

#%% Map colors
color_map = {('Correct', 0): 'blue', ('Correct', 1): 'red', 'Incorrect': 'grey'}
pca_df['Color'] = pca_df.apply(lambda row: color_map[(row['Correct'], row['Y5_survival'])] if row['Correct'] == 'Correct' else color_map['Incorrect'], axis=1)

#%% Plot the PCA results with custom colors for correct/incorrect predictions
fig, axes = plt.subplots(1, 3, figsize=(30, 7))
sns.scatterplot(x='PC1', y='PC2', hue='Color', data=pca_df, palette=['teal', 'salmon', 'dimgrey'], s=100, alpha=0.7, ax=axes[0])
axes[0].set_title('PCA of Test Set with Correct/Incorrect Predictions (PC1 vs PC2)')
axes[0].set_xlabel('Principal Component 1')
axes[0].set_ylabel('Principal Component 2')

sns.scatterplot(x='PC1', y='PC3', hue='Color', data=pca_df, palette=['teal', 'salmon', 'dimgrey'], s=100, alpha=0.7, ax=axes[1])
axes[1].set_title('PCA of Test Set with Correct/Incorrect Predictions (PC1 vs PC3)')
axes[1].set_xlabel('Principal Component 1')
axes[1].set_ylabel('Principal Component 3')

sns.scatterplot(x='PC2', y='PC3', hue='Color', data=pca_df, palette=['teal', 'salmon', 'dimgrey'], s=100, alpha=0.7, ax=axes[2])
axes[2].set_title('PCA of Test Set with Correct/Incorrect Predictions (PC2 vs PC3)')
axes[2].set_xlabel('Principal Component 2')
axes[2].set_ylabel('Principal Component 3')

#%% PCA plot (PC1 vs PC2)
# Set default font size for all elements
plt.rcParams.update({'font.size': 14})

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# First set of plots
sns.scatterplot(x='PC1', y='PC2', hue='Y5_survival', data=pca_df, palette={1: 'teal', 0: 'salmon'}, s=100, alpha=0.5, ax=axes[0])
axes[0].set_title('Actual', fontsize=18)
axes[0].set_xlabel('PC 1', fontsize=16)
axes[0].set_ylabel('PC 2', fontsize=16)
axes[0].legend([],[], frameon=False)

# Second set of plots
sns.scatterplot(x='PC1', y='PC2', hue='Color', data=pca_df, palette=['salmon', 'teal', 'black'], s=100, alpha=0.5, ax=axes[1])
axes[1].set_title('Predict', fontsize=18)
axes[1].set_xlabel('PC 1', fontsize=16)
axes[1].set_ylabel('PC 2', fontsize=16)

plt.tight_layout()
plt.show()
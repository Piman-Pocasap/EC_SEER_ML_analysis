# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:21:06 2024

@author: asus
"""

#%% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             precision_recall_curve, f1_score, recall_score,
                             precision_score, matthews_corrcoef, roc_auc_score, accuracy_score)
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
from sklearn.naive_bayes import GaussianNB
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

#%% Define a function to print evaluation metrics
def print_evaluation_metrics(model, X_test, y_test, y_test_pred, y_test_pred_prob=None):
    accuracy = accuracy_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_pred_prob) if y_test_pred_prob is not None else np.nan
    f1 = f1_score(y_test, y_test_pred)
    metrics = {
        'Accuracy': accuracy,
        'ROC AUC': roc_auc,
        'F1 Score': f1,
        'Sum': accuracy + roc_auc + f1
    }
    return metrics

#%% Define a function to plot ROC-AUC curves
def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(12, 10))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_test_pred_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_test_pred_prob = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_test_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC-AUC Curves', fontsize=16)
    plt.legend(loc='lower right', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

#%% Random Forest Classifier
rf = RandomForestClassifier(
    random_state=56,
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=56)
cv_scores = cross_val_score(rf, X_res, y_res, cv=cv, scoring='accuracy')
print(f'Random Forest Cross-Validation Scores: {cv_scores}')
print(f'Random Forest Mean CV Accuracy: {np.mean(cv_scores)}')
print(f'Random Forest Standard Deviation of CV Accuracy: {np.std(cv_scores)}')
rf.fit(X_res, y_res)
y_test_pred = rf.predict(X_test)
y_test_pred_prob = rf.predict_proba(X_test)[:, 1]
print("Random Forest Evaluation Metrics:")
rf_metrics = print_evaluation_metrics(rf, X_test, y_test, y_test_pred, y_test_pred_prob)

#%% Artificial Neural Network (ANN)
def build_ann_model(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

keras_clf = KerasClassifier(model=build_ann_model, input_dim=X_res.shape[1], epochs=100, batch_size=16, 
                            validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], verbose=0)
cv_scores = []
for train_idx, val_idx in cv.split(X_res, y_res):
    X_train_cv, X_val_cv = X_res.iloc[train_idx], X_res.iloc[val_idx]
    y_train_cv, y_val_cv = y_res.iloc[train_idx], y_res.iloc[val_idx]
    keras_clf.fit(X_train_cv, y_train_cv)
    scores = keras_clf.score(X_val_cv, y_val_cv)
    cv_scores.append(scores)
print(f'ANN Cross-Validation Scores: {cv_scores}')
print(f'ANN Mean CV Accuracy: {np.mean(cv_scores)}')
print(f'ANN Standard Deviation of CV Accuracy: {np.std(cv_scores)}')
keras_clf.fit(X_res, y_res)
y_test_pred_prob = keras_clf.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_pred_prob > 0.5).astype(int)
print("ANN Evaluation Metrics:")
ann_metrics = print_evaluation_metrics(keras_clf, X_test, y_test, y_test_pred, y_test_pred_prob)

#%% K-Nearest Neighbors Classifier
knn = KNeighborsClassifier()
cv_scores = cross_val_score(knn, X_res, y_res, cv=cv, scoring='accuracy')
print(f'KNN Cross-Validation Scores: {cv_scores}')
print(f'KNN Mean CV Accuracy: {np.mean(cv_scores)}')
print(f'KNN Standard Deviation of CV Accuracy: {np.std(cv_scores)}')
knn.fit(X_res, y_res)
y_test_pred = knn.predict(X_test)
y_test_pred_prob = knn.predict_proba(X_test)[:, 1]
print("KNN Evaluation Metrics:")
knn_metrics = print_evaluation_metrics(knn, X_test, y_test, y_test_pred, y_test_pred_prob)

#%% AdaBoost Classifier
base_estimator = DecisionTreeClassifier(max_depth=10)
ada_model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=100,
    learning_rate=0.1,
    random_state=56
)
cv_scores = cross_val_score(ada_model, X_res, y_res, cv=cv, scoring='accuracy')
print(f'AdaBoost Cross-Validation Scores: {cv_scores}')
print(f'AdaBoost Mean CV Accuracy: {np.mean(cv_scores)}')
print(f'AdaBoost Standard Deviation of CV Accuracy: {np.std(cv_scores)}')
ada_model.fit(X_res, y_res)
y_test_pred = ada_model.predict(X_test)
y_test_pred_prob = ada_model.predict_proba(X_test)[:, 1]
print("AdaBoost Evaluation Metrics:")
ada_metrics = print_evaluation_metrics(ada_model, X_test, y_test, y_test_pred, y_test_pred_prob)

#%% Naive Bayes Classifier
nb = GaussianNB()
cv_scores = cross_val_score(nb, X_res, y_res, cv=cv, scoring='accuracy')
print(f'Naive Bayes Cross-Validation Scores: {cv_scores}')
print(f'Naive Bayes Mean CV Accuracy: {np.mean(cv_scores)}')
print(f'Naive Bayes Standard Deviation of CV Accuracy: {np.std(cv_scores)}')
nb.fit(X_res, y_res)
y_test_pred = nb.predict(X_test)
y_test_pred_prob = nb.predict_proba(X_test)[:, 1]
print("Naive Bayes Evaluation Metrics:")
nb_metrics = print_evaluation_metrics(nb, X_test, y_test, y_test_pred, y_test_pred_prob)

#%% Plot ROC-AUC curves for all models
models = {
    'RF': rf,
    'ANN': keras_clf,
    'KNN': knn,
    'AdaBoost': ada_model,
    'Naive Bayes': nb
}
plot_roc_curves(models, X_test, y_test)

#%% Construct a table with performance metrics
metrics_df = pd.DataFrame({
    'Algorithm': ['Random Forest', 'ANN', 'KNN', 'AdaBoost', 'Naive Bayes'],
    'Accuracy': [rf_metrics['Accuracy'], ann_metrics['Accuracy'], knn_metrics['Accuracy'], ada_metrics['Accuracy'], nb_metrics['Accuracy']],
    'ROC AUC': [rf_metrics['ROC AUC'], ann_metrics['ROC AUC'], knn_metrics['ROC AUC'], ada_metrics['ROC AUC'], nb_metrics['ROC AUC']],
    'F1 Score': [rf_metrics['F1 Score'], ann_metrics['F1 Score'], knn_metrics['F1 Score'], ada_metrics['F1 Score'], nb_metrics['F1 Score']],
    'Sum': [rf_metrics['Sum'], ann_metrics['Sum'], knn_metrics['Sum'], ada_metrics['Sum'], nb_metrics['Sum']]
})

print(metrics_df)


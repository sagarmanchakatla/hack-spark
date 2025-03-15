import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('sleep.csv')

# Define features and target
X = df.drop(columns=['SLEEP_APNEA'])
y = df['SLEEP_APNEA']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Save model and scaler
joblib.dump(model, 'sleep_apnea_model.pkl')
joblib.dump(scaler, 'scaler.pkl')



# import joblib
# from flask import Flask, request, jsonify
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# from imblearn.over_sampling import SMOTE
# import pandas as pd

# # Load dataset
# df = pd.read_csv('sleep_apnea_dataset.csv')

# # Feature Engineering: Add BMI
# df['BMI'] = df['WEIGHT'] / ((df['HEIGHT'] / 100) ** 2)

# # Define features and target
# X = df.drop(columns=['SLEEP_APNEA'])
# y = df['SLEEP_APNEA']

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Handle class imbalance using SMOTE
# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)

# # Train Random Forest model with hyperparameter tuning
# model = RandomForestClassifier(random_state=42)

# # Define hyperparameters for Grid Search
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # Perform Grid Search
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
# grid_search.fit(X_train, y_train)

# # Best model
# best_model = grid_search.best_estimator_

# # Evaluate model using cross-validation
# cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
# print(f'Cross-Validation ROC-AUC Scores: {cv_scores}')
# print(f'Mean ROC-AUC: {np.mean(cv_scores):.2f}')

# # Evaluate on test set
# y_pred = best_model.predict(X_test)
# y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# # Calculate metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, y_pred_proba)
# conf_matrix = confusion_matrix(y_test, y_pred)

# print(f'Test Accuracy: {accuracy:.2f}')
# print(f'Precision: {precision:.2f}')
# print(f'Recall: {recall:.2f}')
# print(f'F1-Score: {f1:.2f}')
# print(f'ROC-AUC: {roc_auc:.2f}')
# print('Confusion Matrix:')
# print(conf_matrix)

# # Save model and scaler
# joblib.dump(best_model, 'sleep_apnea_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
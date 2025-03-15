import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('sleep_apnea_dataset.csv')

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
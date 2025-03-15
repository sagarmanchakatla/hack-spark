import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

# Load and clean the dataset
df = pd.read_csv('framingham.csv')
df.dropna(inplace=True)
df.to_csv('framingham.csv', index=False)

data = np.genfromtxt('framingham.csv', delimiter=',', skip_header=1)
X, y = data[:, 0:15], data[:, 15]

# Plotting function
def plot_data(X, y):
    pos = y == 1
    neg = y == 0
    plt.figure(figsize=(10, 6))
    plt.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10, label='CHD Risk')
    plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1, label='No CHD Risk')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('CHD Risk Classification')
    plt.legend()
    plt.grid(True)
    plt.show()

# Sigmoid function
def sigmoid(z):
    z = np.array(z)
    return 1 / (1 + np.exp(-z))

# Setup the data matrix
m, n = X.shape
X = np.concatenate([np.ones((m, 1)), X], axis=1)

# Initialize fitting parameters
theta = np.zeros(n + 1)

# Cost function and gradient
def cost_function(theta, X, y, lambda_):
    m = y.size
    h = sigmoid(X @ theta)
    J = (-1 / m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h)) + (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
    grad = (1 / m) * (X.T @ (h - y))
    grad[1:] += (lambda_ / m) * theta[1:]
    return J, grad

# Optimize the cost function
lambda_ = 1
result = optimize.minimize(fun=cost_function, x0=theta, args=(X, y, lambda_), method='TNC', jac=True)
optimal_theta = result.x

print('Optimized theta:', optimal_theta)

# Prediction function
def predict(theta, patient_features):
    # Add the intercept term (1) to the patient features
    X = np.concatenate(([1], patient_features))
    z = np.dot(theta, X)
    probability = sigmoid(z)
    return probability, probability >= 0.5

# Function to interpret the importance of features
def interpret_features(theta, feature_names):
    # Create a dataframe to store the feature names and their coefficients
    feature_importance = pd.DataFrame({
        'Feature': ['Intercept'] + feature_names,
        'Coefficient': theta,
        'Absolute Value': np.abs(theta)
    })
    # Sort by absolute value to see most influential features
    feature_importance = feature_importance.sort_values('Absolute Value', ascending=False)
    return feature_importance

# Feature names for interpretation
feature_names = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 
                 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 
                 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

# Print feature importance
importance = interpret_features(optimal_theta, feature_names)
print("\nFeature Importance:")
print(importance)

# Make predictions for the two sample patients
patient1 = np.array([1, 39, 4.0, 0, 0.0, 0.0, 0, 0, 0, 195.0, 106.0, 70.0, 26.97, 80.0, 77.0])
patient2 = np.array([0, 46, 2.0, 1, 10.0, 0.0, 0, 0, 0, 250.0, 121.0, 81.0, 28.73, 95.0, 76.0])

# Get predictions
prob1, pred1 = predict(optimal_theta, patient1)
prob2, pred2 = predict(optimal_theta, patient2)

print("\nPatient 1 (39-year-old male):")
print(f"Probability of CHD in next 10 years: {prob1:.2%}")
print(f"Prediction: {'CHD Risk' if pred1 else 'No CHD Risk'}")

print("\nPatient 2 (46-year-old female):")
print(f"Probability of CHD in next 10 years: {prob2:.2%}")
print(f"Prediction: {'CHD Risk' if pred2 else 'No CHD Risk'}")

# Visualize the feature importance
plt.figure(figsize=(12, 8))
plt.barh(importance['Feature'], importance['Coefficient'])
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Importance for CHD Risk Prediction')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.show()
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle

# Load the dataset
data = pd.read_csv("diabetes_prediction_dataset.csv")
data.columns = data.columns.str.strip().str.replace(" ", "_")

# Apply one-hot encoding to categorical variables
data_encoded = pd.get_dummies(data, drop_first=True)

# Separate features and target variable
X = data_encoded.drop("diabetes", axis=1)
y = data_encoded["diabetes"]

print(y.value_counts())

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Save the imputer for later use
pickle.dump(imputer, open("imputer.pkl", "wb"))
# Save the column names for later use
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

# Calculate class weights based on imbalance ratio
class_weights = {0: len(y) / (2 * (len(y) - sum(y))), 1: len(y) / (2 * sum(y))}

# Apply SMOTE
smote = SMOTE(sampling_strategy='minority')
X_smote, y_smote = smote.fit_resample(X_imputed, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(class_weight=class_weights, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
print("Score: ", model.score(X_test, y_test))
test_predictions = model.predict(X_test)
report = classification_report(y_test, test_predictions)
print("Classification Report:")
print(report)

# Save the trained model
joblib.dump(model, "trained_model.joblib")

print("Model training completed and saved successfully.")
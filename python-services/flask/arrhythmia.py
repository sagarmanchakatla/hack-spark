from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load('../heart_arrhythmia/arrhythmia_prediction_model.pkl')

# Define the prediction function
def predict_arrhythmia_risk(heart_rate, body_fat, basal_energy, total_calories, weight, steps):
    data = [[heart_rate, body_fat, basal_energy, total_calories, weight, steps]]
    df_input = pd.DataFrame(data, columns=['HEART_RATE', 'BODY_FAT_PERCENTAGE', 'BASAL_ENERGY_BURNED',
                                           'TOTAL_CALORIES_BURNED', 'WEIGHT', 'STEPS'])

    # Predict probability
    risk_prob = model.predict_proba(df_input)[0][1]

    # Return probability and risk category
    risk_category = "High" if risk_prob > 0.7 else "Medium" if risk_prob > 0.3 else "Low"

    return risk_prob, risk_category

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json(force=True)

    # Extract the features
    heart_rate = data['heart_rate']
    body_fat = data['body_fat']
    basal_energy = data['basal_energy']
    total_calories = data['total_calories']
    weight = data['weight']
    steps = data['steps']

    # Make the prediction
    risk_prob, risk_category = predict_arrhythmia_risk(heart_rate, body_fat, basal_energy, total_calories, weight, steps)

    # Return the result as JSON
    result = {
        'risk_probability': risk_prob,
        'risk_category': risk_category
    }
    return jsonify(result)

# Run the application
if __name__ == '__main__':
    app.run(debug=True)

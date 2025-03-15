from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
heart_arrhythmia_model = joblib.load('../heart_arrhythmia/arrhythmia_prediction_model.pkl')

sleep_apnea_model = joblib.load('../sleep_apnea/sleep_apnea_model.pkl')
sleep_apnea_scaler = joblib.load('../sleep_apnea/scaler.pkl')


def predict_sleep_apnea(heart_rate, body_fat, basal_energy, weight, height, sleep_asleep):
    user_data = np.array([[heart_rate, body_fat, basal_energy, weight, height, sleep_asleep]])
    user_data = sleep_apnea_scaler.transform(user_data)
    prediction = sleep_apnea_model.predict(user_data)[0]
    return 'Sleep Apnea Detected' if prediction == 1 else 'No Sleep Apnea'


# Define the prediction function
def predict_arrhythmia_risk(heart_rate, body_fat, basal_energy, total_calories, weight, steps):
    data = [[heart_rate, body_fat, basal_energy, total_calories, weight, steps]]
    df_input = pd.DataFrame(data, columns=['HEART_RATE', 'BODY_FAT_PERCENTAGE', 'BASAL_ENERGY_BURNED',
                                           'TOTAL_CALORIES_BURNED', 'WEIGHT', 'STEPS'])

    # Predict probability
    risk_prob = heart_arrhythmia_model.predict_proba(df_input)[0][1]

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
    height = data['height']
    sleep_asleep = data['sleep_asleep']

    # Make the prediction
    risk_prob, risk_category = predict_arrhythmia_risk(heart_rate, body_fat, basal_energy, total_calories, weight, steps)
    
    sleep_apnea_result = predict_sleep_apnea(heart_rate, body_fat, basal_energy, weight, height, sleep_asleep)

    # Return the result as JSON
    result = {
        'risk_probability': risk_prob,
        'risk_category': risk_category,
        'sleep_apnea_result': sleep_apnea_result
    }
    return jsonify(result)

# Run the application
if __name__ == '__main__':
    app.run(debug=True)

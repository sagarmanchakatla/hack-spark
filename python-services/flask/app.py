from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import pickle

# Initialize the Flask application
app = Flask(__name__)

# Load the trained models
heart_arrhythmia_model = joblib.load('../heart_arrhythmia/arrhythmia_prediction_model.pkl')
sleep_apnea_model = joblib.load('../sleep_apnea/sleep_apnea_model.pkl')
sleep_apnea_scaler = joblib.load('../sleep_apnea/scaler.pkl')
diabetes_model = joblib.load("../diabeticstrained_model.joblib")
diabetes_imputer = pickle.load(open("../diabetics/imputer.pkl", "rb"))
diabetes_columns = pickle.load(open("../diabetics/columns.pkl", "rb"))

# Sleep Apnea prediction function
def predict_sleep_apnea(heart_rate, body_fat, basal_energy, weight, height, sleep_asleep):
    user_data = np.array([[heart_rate, body_fat, basal_energy, weight, height, sleep_asleep]])
    user_data = sleep_apnea_scaler.transform(user_data)
    prediction = sleep_apnea_model.predict(user_data)[0]
    return 'Sleep Apnea Detected' if prediction == 1 else 'No Sleep Apnea'

# Heart Arrhythmia prediction function
def predict_arrhythmia_risk(heart_rate, body_fat, basal_energy, total_calories, weight, steps):
    data = [[heart_rate, body_fat, basal_energy, total_calories, weight, steps]]
    df_input = pd.DataFrame(data, columns=['HEART_RATE', 'BODY_FAT_PERCENTAGE', 'BASAL_ENERGY_BURNED',
                                           'TOTAL_CALORIES_BURNED', 'WEIGHT', 'STEPS'])

    # Predict probability
    risk_prob = heart_arrhythmia_model.predict_proba(df_input)[0][1]

    # Return probability and risk category
    risk_category = "High" if risk_prob > 0.7 else "Medium" if risk_prob > 0.3 else "Low"

    return risk_prob, risk_category

# Diabetes prediction function
def preprocess_diabetes_data(input_data):
    # Convert 'Yes' and 'No' to 1 and 0
    input_data['hypertension'] = input_data['hypertension'].apply(lambda x: 1 if x == 'Yes' else 0)
    input_data['heart_disease'] = input_data['heart_disease'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    input_data_encoded = pd.get_dummies(input_data, drop_first=True)
    input_data_encoded = input_data_encoded.reindex(columns=diabetes_columns, fill_value=0)
    input_data_imputed = diabetes_imputer.transform(input_data_encoded)
    
    return input_data_imputed

# Heart arrhythmia prediction route
@app.route('/heart', methods=['POST'])
def predict_heart():
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
        'risk_probability': float(risk_prob),
        'risk_category': risk_category
    }
    return jsonify(result)

# Sleep apnea prediction route
@app.route('/sleep', methods=['POST'])
def predict_sleep():
    data = request.get_json(force=True)

    # Extract the features
    heart_rate = data['heart_rate']
    body_fat = data['body_fat']
    basal_energy = data['basal_energy']
    weight = data['weight']
    height = data['height']
    sleep_asleep = data['sleep_asleep']
    
    sleep_apnea_result = predict_sleep_apnea(heart_rate, body_fat, basal_energy, weight, height, sleep_asleep)
    
    result = {
        'sleep_apnea_result': sleep_apnea_result
    }
    
    return jsonify(result)

# Diabetes prediction route
@app.route('/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Preprocess input data
        input_data = pd.DataFrame([data], columns=diabetes_columns)
        input_data_imputed = preprocess_diabetes_data(input_data)

        # Make predictions
        prediction = diabetes_model.predict(input_data_imputed)
        probability = diabetes_model.predict_proba(input_data_imputed)[0][1]  # Probability of being diabetic
        
        # Prepare response
        result = {
            'prediction': 'diabetic' if prediction[0] == 1 else 'not_diabetic',
            'probability': float(probability),
            'message': 'You are possibly diabetic' if prediction[0] == 1 else 'You are not diabetic'
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
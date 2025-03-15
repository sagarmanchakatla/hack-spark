from flask import Flask, jsonify, request
import pandas as pd
import joblib
import pickle

app = Flask(__name__)

# Load the trained model and preprocessing components
model = joblib.load("trained_model.joblib")
imputer = pickle.load(open("imputer.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Preprocess input data
def preprocess_input_data(input_data):
    # Convert 'Yes' and 'No' to 1 and 0
    input_data['hypertension'] = input_data['hypertension'].apply(lambda x: 1 if x == 'Yes' else 0)
    input_data['heart_disease'] = input_data['heart_disease'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    input_data_encoded = pd.get_dummies(input_data, drop_first=True)
    input_data_encoded = input_data_encoded.reindex(columns=columns, fill_value=0)
    input_data_imputed = imputer.transform(input_data_encoded)
    
    return input_data_imputed

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Preprocess input data
        input_data = pd.DataFrame([data], columns=columns)
        input_data_imputed = preprocess_input_data(input_data)

        # Make predictions
        prediction = model.predict(input_data_imputed)
        probability = model.predict_proba(input_data_imputed)[0][1]  # Probability of being diabetic
        
        # Prepare response
        result = {
            'prediction': 'diabetic' if prediction[0] == 1 else 'not_diabetic',
            'probability': float(probability),
            'message': 'You are possibly diabetic' if prediction[0] == 1 else 'You are not diabetic'
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
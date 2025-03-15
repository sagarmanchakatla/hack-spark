# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd
# import numpy as np
# import pickle

# # Initialize the Flask application
# app = Flask(__name__)

# # Load the trained models
# heart_arrhythmia_model = joblib.load('../heart_arrhythmia/arrhythmia_prediction_model.pkl')
# sleep_apnea_model = joblib.load('../sleep_apnea/sleep_apnea_model.pkl')
# sleep_apnea_scaler = joblib.load('../sleep_apnea/scaler.pkl')
# diabetes_model = joblib.load("../diabeticstrained_model.joblib")
# diabetes_imputer = pickle.load(open("../diabetics/imputer.pkl", "rb"))
# diabetes_columns = pickle.load(open("../diabetics/columns.pkl", "rb"))

# # Sleep Apnea prediction function
# def predict_sleep_apnea(heart_rate, body_fat, basal_energy, weight, height, sleep_asleep):
#     user_data = np.array([[heart_rate, body_fat, basal_energy, weight, height, sleep_asleep]])
#     user_data = sleep_apnea_scaler.transform(user_data)
#     prediction = sleep_apnea_model.predict(user_data)[0]
#     return 'Sleep Apnea Detected' if prediction == 1 else 'No Sleep Apnea'

# # Heart Arrhythmia prediction function
# def predict_arrhythmia_risk(heart_rate, body_fat, basal_energy, total_calories, weight, steps):
#     data = [[heart_rate, body_fat, basal_energy, total_calories, weight, steps]]
#     df_input = pd.DataFrame(data, columns=['HEART_RATE', 'BODY_FAT_PERCENTAGE', 'BASAL_ENERGY_BURNED',
#                                            'TOTAL_CALORIES_BURNED', 'WEIGHT', 'STEPS'])

#     # Predict probability
#     risk_prob = heart_arrhythmia_model.predict_proba(df_input)[0][1]

#     # Return probability and risk category
#     risk_category = "High" if risk_prob > 0.7 else "Medium" if risk_prob > 0.3 else "Low"

#     return risk_prob, risk_category

# # Diabetes prediction function
# def preprocess_diabetes_data(input_data):
#     # Convert 'Yes' and 'No' to 1 and 0
#     input_data['hypertension'] = input_data['hypertension'].apply(lambda x: 1 if x == 'Yes' else 0)
#     input_data['heart_disease'] = input_data['heart_disease'].apply(lambda x: 1 if x == 'Yes' else 0)
    
#     input_data_encoded = pd.get_dummies(input_data, drop_first=True)
#     input_data_encoded = input_data_encoded.reindex(columns=diabetes_columns, fill_value=0)
#     input_data_imputed = diabetes_imputer.transform(input_data_encoded)
    
#     return input_data_imputed

# # Heart arrhythmia prediction route
# @app.route('/heart', methods=['POST'])
# def predict_heart():
#     # Get the input data from the request
#     data = request.get_json(force=True)

#     # Extract the features
#     heart_rate = data['heart_rate']
#     body_fat = data['body_fat']
#     basal_energy = data['basal_energy']
#     total_calories = data['total_calories']
#     weight = data['weight']
#     steps = data['steps']

#     # Make the prediction
#     risk_prob, risk_category = predict_arrhythmia_risk(heart_rate, body_fat, basal_energy, total_calories, weight, steps)
    
#     # Return the result as JSON
#     result = {
#         'risk_probability': float(risk_prob),
#         'risk_category': risk_category
#     }
#     return jsonify(result)

# # Sleep apnea prediction route
# @app.route('/sleep', methods=['POST'])
# def predict_sleep():
#     data = request.get_json(force=True)

#     # Extract the features
#     heart_rate = data['heart_rate']
#     body_fat = data['body_fat']
#     basal_energy = data['basal_energy']
#     weight = data['weight']
#     height = data['height']
#     sleep_asleep = data['sleep_asleep']
    
#     sleep_apnea_result = predict_sleep_apnea(heart_rate, body_fat, basal_energy, weight, height, sleep_asleep)
    
#     result = {
#         'sleep_apnea_result': sleep_apnea_result
#     }
    
#     return jsonify(result)

# # Diabetes prediction route
# @app.route('/diabetes', methods=['POST'])
# def predict_diabetes():
#     try:
#         # Get JSON data from request
#         data = request.get_json(force=True)
        
#         if not data:
#             return jsonify({'error': 'No data provided'}), 400
        
#         # Preprocess input data
#         input_data = pd.DataFrame([data], columns=diabetes_columns)
#         input_data_imputed = preprocess_diabetes_data(input_data)

#         # Make predictions
#         prediction = diabetes_model.predict(input_data_imputed)
#         probability = diabetes_model.predict_proba(input_data_imputed)[0][1]  # Probability of being diabetic
        
#         # Prepare response
#         result = {
#             'prediction': 'diabetic' if prediction[0] == 1 else 'not_diabetic',
#             'probability': float(probability),
#             'message': 'You are possibly diabetic' if prediction[0] == 1 else 'You are not diabetic'
#         }
        
#         return jsonify(result)

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # Run the application
# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import pickle
import os
import requests

# Load environment variables
GROQ_API_KEY = "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"

# Initialize the Flask application
app = Flask(__name__)

# Load the trained models
heart_arrhythmia_model = joblib.load('../heart_arrhythmia/arrhythmia_prediction_model.pkl')
sleep_apnea_model = joblib.load('../sleep_apnea/sleep_apnea_model.pkl')
sleep_apnea_scaler = joblib.load('../sleep_apnea/scaler.pkl')
diabetes_model = joblib.load("../diabetics/trained_model.joblib")
diabetes_imputer = pickle.load(open("../diabetics/imputer.pkl", "rb"))
diabetes_columns = pickle.load(open("../diabetics/columns.pkl", "rb"))

# Function to get recommendations from Groq API using Mistral LLM
def get_recommendations(condition, user_data, prediction_result):
    """
    Get personalized recommendations using Groq API with Mistral LLM
    
    Args:
        condition (str): The health condition (heart_arrhythmia, sleep_apnea, diabetes)
        user_data (dict): User's health data
        prediction_result (dict): Prediction results from the model
    
    Returns:
        dict: Recommendations from the LLM
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Create a prompt based on the condition and user data
    if condition == "heart_arrhythmia":
        prompt = f"""
        As a healthcare advisor, provide personalized recommendations for someone with the following heart health data:
        - Heart Rate: {user_data['heart_rate']} bpm
        - Body Fat Percentage: {user_data['body_fat']}%
        - Basal Energy Burned: {user_data['basal_energy']} calories
        - Total Calories Burned: {user_data['total_calories']} calories
        - Weight: {user_data['weight']} kg
        - Steps per day: {user_data['steps']}
        
        Their heart arrhythmia risk assessment is: {prediction_result['risk_category']} (probability: {prediction_result['risk_probability']:.2f})
        
        Provide 5 specific, actionable recommendations to improve their heart health. Include:
        1. Lifestyle changes
        2. Exercise suggestions
        3. Dietary advice
        4. Monitoring recommendations
        5. When to seek medical attention
        
        Format the response as JSON with recommendations and explanation fields.
        """
    
    elif condition == "sleep_apnea":
        prompt = f"""
        As a healthcare advisor, provide personalized recommendations for someone with the following health data:
        - Heart Rate: {user_data['heart_rate']} bpm
        - Body Fat Percentage: {user_data['body_fat']}%
        - Basal Energy Burned: {user_data['basal_energy']} calories
        - Weight: {user_data['weight']} kg
        - Height: {user_data['height']} cm
        - Sleep Duration: {user_data['sleep_asleep']} hours
        
        Sleep apnea assessment: {prediction_result['sleep_apnea_result']}
        
        Provide 5 specific, actionable recommendations to improve their sleep quality and reduce sleep apnea risk. Include:
        1. Sleep position and environment changes
        2. Lifestyle modifications
        3. Weight management strategies if applicable
        4. Exercise recommendations
        5. When to consult a sleep specialist
        
        Format the response as JSON with recommendations and explanation fields.
        """
    
    elif condition == "diabetes":
        # Extract key diabetes risk factors from user data
        age = user_data.get('age', 'N/A')
        bmi = user_data.get('bmi', 'N/A')
        glucose = user_data.get('glucose', 'N/A')
        hypertension = user_data.get('hypertension', 'N/A')
        
        prompt = f"""
        As a healthcare advisor, provide personalized recommendations for someone with the following health data:
        - Age: {age}
        - BMI: {bmi}
        - Glucose Level: {glucose}
        - Hypertension: {hypertension}
        
        Diabetes assessment: {prediction_result['message']} (probability: {prediction_result['probability']:.2f})
        
        Provide 5 specific, actionable recommendations to manage their diabetes risk. Include:
        1. Dietary changes
        2. Exercise recommendations
        3. Monitoring suggestions
        4. Lifestyle modifications
        5. When to consult with healthcare providers
        
        Format the response as JSON with recommendations and explanation fields.
        """
    
    # Make request to Groq API
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json={
                "model": "mixtral-8x7b-32768",
                "messages": [
                    {"role": "system", "content": "You are a healthcare advisor providing evidence-based recommendations. Respond in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1024
            }
        )
        
        if response.status_code == 200:
            # Extract the recommendations from the response
            content = response.json()["choices"][0]["message"]["content"]
            
            # Handle potential non-JSON responses by wrapping them
            try:
                recommendations = eval(content)
            except:
                recommendations = {
                    "recommendations": [content],
                    "explanation": "Generated recommendations based on your health data."
                }
            
            return recommendations
        else:
            return {
                "recommendations": ["Unable to generate personalized recommendations at this time."],
                "explanation": "Please consult with a healthcare professional for personalized advice."
            }
    
    except Exception as e:
        print(f"Error calling Groq API: {str(e)}")
        return {
            "recommendations": ["Unable to generate personalized recommendations at this time."],
            "explanation": "Please consult with a healthcare professional for personalized advice."
        }

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


def calculate_stress_level(heart_rate, sleep_duration, steps, self_reported_stress=5):
    # Normalize each factor to a scale of 0-1
    heart_rate_score = heart_rate / 84
    sleep_score = (8 - sleep_duration) / 8
    steps_score = (10000 - steps) / 10000
    stress_score = self_reported_stress / 10

    # Calculate overall stress level
    stress_level = heart_rate_score + sleep_score + steps_score + stress_score
    return stress_level

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
    
    # Prepare the prediction result
    prediction_result = {
        'risk_probability': float(risk_prob),
        'risk_category': risk_category
    }
    
    # Get personalized recommendations
    recommendations = get_recommendations(
        condition="heart_arrhythmia", 
        user_data=data, 
        prediction_result=prediction_result
    )
    
    # Return the result as JSON with recommendations
    result = {
        **prediction_result,
        'recommendations': recommendations
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
    
    # Prepare the prediction result
    prediction_result = {
        'sleep_apnea_result': sleep_apnea_result
    }
    
    # Get personalized recommendations
    recommendations = get_recommendations(
        condition="sleep_apnea", 
        user_data=data, 
        prediction_result=prediction_result
    )
    
    # Return the result as JSON with recommendations
    result = {
        **prediction_result,
        'recommendations': recommendations
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
        input_data = pd.DataFrame([data])
        input_data_imputed = preprocess_diabetes_data(input_data)

        # Make predictions
        prediction = diabetes_model.predict(input_data_imputed)
        probability = diabetes_model.predict_proba(input_data_imputed)[0][1]  # Probability of being diabetic
        
        # Prepare prediction result
        prediction_result = {
            'prediction': 'diabetic' if prediction[0] == 1 else 'not_diabetic',
            'probability': float(probability),
            'message': 'You are possibly diabetic' if prediction[0] == 1 else 'You are not diabetic'
        }
        
        # Get personalized recommendations
        recommendations = get_recommendations(
            condition="diabetes", 
            user_data=data, 
            prediction_result=prediction_result
        )
        
        # Return the result as JSON with recommendations
        result = {
            **prediction_result,
            'recommendations': recommendations
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500



def get_stress_recommendations(stress_level):
    recommendations = []

    if stress_level > 2.5:
        recommendations.append("High stress detected. Consider taking a break and practicing relaxation techniques.")
    elif stress_level > 1.5:
        recommendations.append("Moderate stress detected. Ensure you are getting enough rest and engaging in stress-reducing activities.")
    else:
        recommendations.append("Low stress detected. Keep up the good habits!")

    recommendations.append("Maintain a balanced diet and stay hydrated.")
    recommendations.append("Engage in regular physical activity to reduce stress levels.")
    recommendations.append("Practice mindfulness and meditation to improve mental well-being.")
    recommendations.append("Consider consulting a healthcare professional if stress levels remain high.")

    return recommendations

# Get recommendations based on the calculated stress level
# recommendations = get_stress_recommendations(stress_level)
# for rec in recommendations:
#     print(f"- {rec}")


@app.route('/stress', methods=['POST'])
def assess_stress():
    data = request.get_json(force=True)

    # Extract the features
    heart_rate = data['heart_rate']
    sleep_duration = data['sleep_asleep']
    steps = data['steps']
    self_reported_stress = data.get('self_reported_stress', 5)  # Default to 5 if not provided

    # Calculate stress level
    stress_level = calculate_stress_level(heart_rate, sleep_duration, steps, self_reported_stress)
    print(stress_level)
    # Get personalized recommendations
    recommendations = get_stress_recommendations(stress_level)

    # Return the result as JSON with recommendations
    result = {
        'stress_level': stress_level,
        'recommendations': recommendations
    }

    return jsonify(result)



# Run the application
if __name__ == '__main__':
    app.run(debug=True)
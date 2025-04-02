from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok  # Required for Google Colab
import tensorflow as tf
import numpy as np
import joblib
import json
import os

app = Flask(__name__)
run_with_ngrok(app)  # Enable public URL for Flask in Colab

# Configuration
DEBUG = True

try:
    # Load model and preprocessing
    model = tf.keras.models.load_model('trained_model.keras')
    scaler = joblib.load('prostate_scaler.pkl')
    print("✅ Model and scaler loaded successfully")
    print(f"Model output shape: {model.output_shape}")  # Debug model output
except Exception as e:
    print(f"❌ Error loading model/scaler: {str(e)}")
    raise

try:
    with open('selected_features.json', 'r') as f:
        selected_features = json.load(f)
    print("✅ Features loaded:", selected_features)
except Exception as e:
    print(f"❌ Error loading features: {str(e)}")
    selected_features = []

# Constants
TREATMENT_OPTIONS = [
    "Active Surveillance", "Radiation", "Immunotherapy",
    "Chemotherapy", "Surgery", "Hormone Therapy"
]

MAPPINGS = {
    'Alcohol_Consumption': {'Low': 0, 'Moderate': 1, 'High': 2},
    'Cancer_Stage': {'Localized': 0, 'Metastatic': 1, 'Advanced': 2},
    'YesNo': {'No': 0, 'Yes': 1}
}

DISCLAIMER = """<div class="disclaimer">
<strong>Important:</strong> This tool provides preliminary recommendations only. 
Always consult with a qualified healthcare professional for medical decisions. 
Model accuracy: 87% (validation set).</div>"""

def determine_treatment(features, recommendation_prob):
    """Fallback treatment determination logic"""
    age = features['Age']
    cancer_stage = features['Cancer_Stage']
    
    if recommendation_prob < 0.3:
        return 0  # Active Surveillance
    elif age > 75:
        return 5 if cancer_stage == 2 else 1  # Hormone Therapy for advanced, else Radiation
    elif cancer_stage == 2:  # Advanced
        return 3  # Chemotherapy
    else:
        return 4 if age < 65 else 1  # Surgery for younger patients, else Radiation

@app.route('/')
def home():
    return render_template('index.html',
                         features=selected_features,
                         disclaimer=DISCLAIMER,
                         alcohol_options=MAPPINGS['Alcohol_Consumption'].keys(),
                         stage_options=MAPPINGS['Cancer_Stage'].keys())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("\n📋 Received form data:", request.form)
        
        # Process and validate inputs
        features = {
            'PSA_Level': float(request.form['PSA_Level']),
            'Prostate_Volume': float(request.form['Prostate_Volume']),
            'BMI': float(request.form['BMI']),
            'Age': float(request.form['Age']),
            'Screening_Age': float(request.form['Screening_Age']),
            'Alcohol_Consumption': MAPPINGS['Alcohol_Consumption'][request.form['Alcohol_Consumption']],
            'Cancer_Stage': MAPPINGS['Cancer_Stage'][request.form['Cancer_Stage']],
            'Erectile_Dysfunction': MAPPINGS['YesNo'][request.form['Erectile_Dysfunction']],
            'Follow_Up_Required': MAPPINGS['YesNo'][request.form['Follow_Up_Required']],
            'Exercise_Regularly': MAPPINGS['YesNo'][request.form['Exercise_Regularly']]
        }
        
        # Prepare data for model
        processed_data = [features[field] for field in selected_features]
        scaled_data = scaler.transform(np.array(processed_data).reshape(1, -1))
        
        # Get prediction
        prediction = model.predict(scaled_data)
        print("🤖 Raw model output:", prediction)
        
        # Handle model output
        if prediction.shape[1] == 1:  # Single output model
            recommendation_prob = float(prediction[0][0])
            treatment_idx = determine_treatment(features, recommendation_prob)
            print("ℹ️ Using fallback treatment determination")
        else:  # Dual output model
            recommendation_prob = float(prediction[0][0])
            treatment_idx = int(prediction[0][1])
        
        # Prepare results
        recommendation = "Recommended" if recommendation_prob > 0.5 else "Not Recommended"
        confidence = min(max(
            recommendation_prob if recommendation == "Recommended" else 1 - recommendation_prob, 
            0.0
        ), 1.0)
        treatment_type = TREATMENT_OPTIONS[treatment_idx % len(TREATMENT_OPTIONS)]
        
        # Generate next steps
        next_steps = {
            "Active Surveillance": ["Regular monitoring", "Annual checkups"],
            "Radiation": ["Consult radiation oncologist"],
            "Surgery": ["Pre-surgical evaluation"],
            # ... add others
        }.get(treatment_type, ["Consult specialist"])
        
        return render_template('result.html',
                            recommendation=recommendation,
                            confidence=f"{confidence*100:.1f}%",
                            treatment_type=treatment_type,
                            next_steps=next_steps,
                            disclaimer=DISCLAIMER)

    except ValueError as e:
        error_msg = f"Invalid input: {str(e)}"
        print(f"❌ {error_msg}")
        return render_template('error.html', error_message=error_msg), 400
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        print(f"❌ {error_msg}")
        return render_template('error.html', error_message=error_msg), 500

if __name__ == '__main__':
    app.run()

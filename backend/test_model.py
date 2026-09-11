"""Test the current model with realistic clinical scenarios to diagnose prediction issues."""
import joblib
import pandas as pd
import numpy as np

MODEL_DIR = 'models'

model = joblib.load(f'{MODEL_DIR}/best_model.pkl')
scaler = joblib.load(f'{MODEL_DIR}/scaler.pkl')
label_encoders = joblib.load(f'{MODEL_DIR}/label_encoders.pkl')
metadata = joblib.load(f'{MODEL_DIR}/metadata.pkl')

FEATURE_ORDER = metadata['feature_names']
print(f"Model: {metadata['model_name']}")
print(f"Features: {FEATURE_ORDER}")
print(f"Accuracy: {metadata['accuracy']:.4f}")
print(f"ROC AUC: {metadata['roc_auc']:.4f}")

# Show label encoder mappings
print("\n=== Label Encoder Mappings ===")
for name, le in label_encoders.items():
    print(f"{name}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Show scaler stats
print("\n=== Scaler Means ===")
for f, m, s in zip(FEATURE_ORDER, scaler.mean_, scaler.scale_):
    print(f"  {f}: mean={m:.4f}, scale={s:.4f}")

def predict_case(case_name, raw_data):
    """Predict a test case and show probability."""
    features = {}
    for feat in FEATURE_ORDER:
        val = raw_data.get(feat)
        if feat in label_encoders:
            try:
                val = label_encoders[feat].transform([val])[0]
            except (ValueError, KeyError):
                val = 0
        features[feat] = val
    
    df = pd.DataFrame([[features[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
    scaled = scaler.transform(df)
    prob = model.predict_proba(scaled)[0][1]
    pred = model.predict(scaled)[0]
    
    print(f"\n--- {case_name} ---")
    print(f"  Probability of diabetes: {prob*100:.1f}%")
    print(f"  Prediction: {'DIABETIC' if pred == 1 else 'NON-DIABETIC'}")
    return prob

# Test Case 1: Clearly healthy person
predict_case("HEALTHY 30-year-old", {
    'gender': 'Male', 'age': 30, 'pregnancies': 0,
    'hypertension': 0, 'heart_disease': 0, 'smoking_history': 'never',
    'bmi': 22.0, 'HbA1c_level': 5.0, 'blood_glucose_level': 85,
    'family_history': 'No', 'physical_activity': 'Active', 'diet_quality': 'Excellent',
    'stress_level': 3, 'sleep_hours': 7.5, 'alcohol_consumption': 'None',
    'waist_circumference': 70, 'cholesterol_total': 170, 'bp_systolic': 110, 'bp_diastolic': 70
})

# Test Case 2: Very high risk person (should be ~90%+ risk)
predict_case("HIGH-RISK 60-year-old", {
    'gender': 'Male', 'age': 60, 'pregnancies': 0,
    'hypertension': 1, 'heart_disease': 1, 'smoking_history': 'current',
    'bmi': 38.0, 'HbA1c_level': 8.5, 'blood_glucose_level': 250,
    'family_history': 'Yes', 'physical_activity': 'Sedentary', 'diet_quality': 'Poor',
    'stress_level': 9, 'sleep_hours': 4.0, 'alcohol_consumption': 'Heavy',
    'waist_circumference': 110, 'cholesterol_total': 280, 'bp_systolic': 160, 'bp_diastolic': 100
})

# Test Case 3: Moderate risk - pre-diabetic range values
predict_case("MODERATE-RISK 50-year-old", {
    'gender': 'Female', 'age': 50, 'pregnancies': 3,
    'hypertension': 0, 'heart_disease': 0, 'smoking_history': 'former',
    'bmi': 29.0, 'HbA1c_level': 6.0, 'blood_glucose_level': 115,
    'family_history': 'Yes', 'physical_activity': 'Light', 'diet_quality': 'Average',
    'stress_level': 6, 'sleep_hours': 6.0, 'alcohol_consumption': 'Moderate',
    'waist_circumference': 92, 'cholesterol_total': 220, 'bp_systolic': 135, 'bp_diastolic': 85
})

# Test Case 4: Young healthy female
predict_case("HEALTHY 25-year-old female", {
    'gender': 'Female', 'age': 25, 'pregnancies': 0,
    'hypertension': 0, 'heart_disease': 0, 'smoking_history': 'never',
    'bmi': 21.0, 'HbA1c_level': 4.8, 'blood_glucose_level': 80,
    'family_history': 'No', 'physical_activity': 'Active', 'diet_quality': 'Good',
    'stress_level': 2, 'sleep_hours': 8.0, 'alcohol_consumption': 'Occasional',
    'waist_circumference': 65, 'cholesterol_total': 160, 'bp_systolic': 105, 'bp_diastolic': 65
})

# Test Case 5: Borderline person - just slightly elevated
predict_case("BORDERLINE 45-year-old", {
    'gender': 'Male', 'age': 45, 'pregnancies': 0,
    'hypertension': 1, 'heart_disease': 0, 'smoking_history': 'former',
    'bmi': 28.0, 'HbA1c_level': 5.8, 'blood_glucose_level': 108,
    'family_history': 'Yes', 'physical_activity': 'Moderate', 'diet_quality': 'Average',
    'stress_level': 5, 'sleep_hours': 6.5, 'alcohol_consumption': 'Moderate',
    'waist_circumference': 95, 'cholesterol_total': 210, 'bp_systolic': 130, 'bp_diastolic': 82
})

# Test Case 6: Diabetic-range values but young
predict_case("YOUNG with DIABETIC-RANGE values", {
    'gender': 'Male', 'age': 35, 'pregnancies': 0,
    'hypertension': 0, 'heart_disease': 0, 'smoking_history': 'never',
    'bmi': 24.0, 'HbA1c_level': 7.0, 'blood_glucose_level': 180,
    'family_history': 'No', 'physical_activity': 'Moderate', 'diet_quality': 'Good',
    'stress_level': 4, 'sleep_hours': 7.0, 'alcohol_consumption': 'None',
    'waist_circumference': 80, 'cholesterol_total': 190, 'bp_systolic': 118, 'bp_diastolic': 76
})

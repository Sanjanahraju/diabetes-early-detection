"""
Automated Verification Script for Diabetes Prediction Pipeline
================================================================
Validates that the production pipeline and API prediction functions:
1. Produce continuous, non-zero, clinically meaningful probabilities.
2. Maintain 0 <= probability <= 1.0 and 0 <= risk_percentage <= 100.0.
3. Test diverse patient profiles (younger, older, male, female, low-risk,
   intermediate pre-diabetic, and high-risk).
4. Test actual ground-truth rows from diabetes_enhanced.csv (without target).
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'diabetes_enhanced.csv')

def run_tests():
    pipeline_path = os.path.join(MODEL_DIR, 'production_pipeline.pkl')
    print("=" * 80)
    print(f"LOADING PIPELINE: {pipeline_path}")
    pipeline = joblib.load(pipeline_path)
    metadata = joblib.load(os.path.join(MODEL_DIR, 'metadata.pkl'))
    print(f"Model Name: {metadata['model_name']}")
    print(f"Accuracy:   {metadata['accuracy']*100:.2f}%")
    print(f"ROC-AUC:    {metadata['roc_auc']:.4f}")
    print(f"F1 Score:   {metadata['f1_score']:.4f}")
    print(f"Classes:    {pipeline.classes_}")
    print("=" * 80)

    classes = pipeline.classes_
    pos_idx = list(classes).index(1)

    # ---------------------------------------------------------
    # PART 1: Diverse Clinical Patient Profiles
    # ---------------------------------------------------------
    profiles = [
        ("Young Healthy Athlete (22y F, Normal HbA1c 4.6)", {
            'gender': 'Female', 'age': 22, 'pregnancies': 0, 'hypertension': 0, 'heart_disease': 0,
            'smoking_history': 'never', 'bmi': 20.0, 'HbA1c_level': 4.6, 'blood_glucose_level': 78,
            'family_history': 'No', 'physical_activity': 'Active', 'diet_quality': 'Excellent',
            'stress_level': 2, 'sleep_hours': 8.0, 'alcohol_consumption': 'None',
            'waist_circumference': 68.0, 'cholesterol_total': 155, 'bp_systolic': 108, 'bp_diastolic': 70
        }),
        ("Average Young Male (32y M, Normal HbA1c 5.1)", {
            'gender': 'Male', 'age': 32, 'pregnancies': 0, 'hypertension': 0, 'heart_disease': 0,
            'smoking_history': 'never', 'bmi': 23.5, 'HbA1c_level': 5.1, 'blood_glucose_level': 88,
            'family_history': 'No', 'physical_activity': 'Moderate', 'diet_quality': 'Good',
            'stress_level': 3, 'sleep_hours': 7.5, 'alcohol_consumption': 'None',
            'waist_circumference': 80.0, 'cholesterol_total': 175, 'bp_systolic': 118, 'bp_diastolic': 76
        }),
        ("Middle-Aged Mild Risk (42y M, HbA1c 5.4, Family Hist Yes)", {
            'gender': 'Male', 'age': 42, 'pregnancies': 0, 'hypertension': 0, 'heart_disease': 0,
            'smoking_history': 'former', 'bmi': 25.5, 'HbA1c_level': 5.4, 'blood_glucose_level': 96,
            'family_history': 'Yes', 'physical_activity': 'Moderate', 'diet_quality': 'Good',
            'stress_level': 4, 'sleep_hours': 7.0, 'alcohol_consumption': 'Occasional',
            'waist_circumference': 86.0, 'cholesterol_total': 190, 'bp_systolic': 122, 'bp_diastolic': 78
        }),
        ("Early Pre-Diabetic (47y F, HbA1c 5.7, Gluc 104, Pregnancies 2)", {
            'gender': 'Female', 'age': 47, 'pregnancies': 2, 'hypertension': 0, 'heart_disease': 0,
            'smoking_history': 'never', 'bmi': 27.2, 'HbA1c_level': 5.7, 'blood_glucose_level': 104,
            'family_history': 'Yes', 'physical_activity': 'Light', 'diet_quality': 'Average',
            'stress_level': 5, 'sleep_hours': 6.5, 'alcohol_consumption': 'None',
            'waist_circumference': 89.0, 'cholesterol_total': 205, 'bp_systolic': 128, 'bp_diastolic': 82
        }),
        ("Confirmed Pre-Diabetic (52y M, HbA1c 6.0, Gluc 114, Hyp 1)", {
            'gender': 'Male', 'age': 52, 'pregnancies': 0, 'hypertension': 1, 'heart_disease': 0,
            'smoking_history': 'former', 'bmi': 29.0, 'HbA1c_level': 6.0, 'blood_glucose_level': 114,
            'family_history': 'Yes', 'physical_activity': 'Light', 'diet_quality': 'Average',
            'stress_level': 6, 'sleep_hours': 6.0, 'alcohol_consumption': 'Moderate',
            'waist_circumference': 94.0, 'cholesterol_total': 220, 'bp_systolic': 134, 'bp_diastolic': 84
        }),
        ("High-Risk Borderline (56y F, HbA1c 6.3, Gluc 124, Hyp 1)", {
            'gender': 'Female', 'age': 56, 'pregnancies': 3, 'hypertension': 1, 'heart_disease': 0,
            'smoking_history': 'former', 'bmi': 31.5, 'HbA1c_level': 6.3, 'blood_glucose_level': 124,
            'family_history': 'Yes', 'physical_activity': 'Sedentary', 'diet_quality': 'Average',
            'stress_level': 7, 'sleep_hours': 5.5, 'alcohol_consumption': 'None',
            'waist_circumference': 98.0, 'cholesterol_total': 235, 'bp_systolic': 140, 'bp_diastolic': 88
        }),
        ("Early Diabetic (55y M, HbA1c 6.6, Gluc 132, Current Smoker)", {
            'gender': 'Male', 'age': 55, 'pregnancies': 0, 'hypertension': 1, 'heart_disease': 0,
            'smoking_history': 'current', 'bmi': 32.5, 'HbA1c_level': 6.6, 'blood_glucose_level': 132,
            'family_history': 'Yes', 'physical_activity': 'Sedentary', 'diet_quality': 'Poor',
            'stress_level': 7, 'sleep_hours': 5.5, 'alcohol_consumption': 'Moderate',
            'waist_circumference': 102.0, 'cholesterol_total': 245, 'bp_systolic': 144, 'bp_diastolic': 90
        }),
        ("Confirmed Diabetic (60y M, HbA1c 7.5, Gluc 160, HD 1)", {
            'gender': 'Male', 'age': 60, 'pregnancies': 0, 'hypertension': 1, 'heart_disease': 1,
            'smoking_history': 'current', 'bmi': 34.5, 'HbA1c_level': 7.5, 'blood_glucose_level': 160,
            'family_history': 'Yes', 'physical_activity': 'Sedentary', 'diet_quality': 'Poor',
            'stress_level': 8, 'sleep_hours': 5.0, 'alcohol_consumption': 'Heavy',
            'waist_circumference': 108.0, 'cholesterol_total': 260, 'bp_systolic': 152, 'bp_diastolic': 95
        }),
    ]

    print("\n" + "-" * 80)
    print("TEST 1: CLINICAL PROFILE PREDICTIONS")
    print("-" * 80)

    all_valid = True
    non_zero_count = 0

    for title, patient_data in profiles:
        input_df = pd.DataFrame([patient_data])
        probs = pipeline.predict_proba(input_df)[0]
        prob = float(probs[pos_idx])
        risk_pct = round(prob * 100, 2)
        pred_class = int(pipeline.predict(input_df)[0])

        assert 0.0 <= prob <= 1.0, f"Probability out of bounds: {prob}"
        assert 0.0 <= risk_pct <= 100.0, f"Percentage out of bounds: {risk_pct}"

        if risk_pct > 0.0:
            non_zero_count += 1

        category = "Low Risk" if risk_pct < 20 else ("Moderate Risk" if risk_pct < 50 else "High Risk")
        print(f"Profile: {title}")
        print(f"  Probability: {prob:.4f} | Risk Score: {risk_pct:5.1f}% | Category: {category:13s} | Class: {pred_class}")

    print(f"\nNon-zero probability rate across profiles: {non_zero_count}/{len(profiles)} ({non_zero_count/len(profiles)*100:.0f}%)")
    assert non_zero_count > 0, "ERROR: All predictions returned exactly 0%!"

    # ---------------------------------------------------------
    # PART 2: Actual Dataset Rows (diabetes = 0 and diabetes = 1)
    # ---------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 2: ACTUAL DATASET GROUND TRUTH SAMPLES")
    print("-" * 80)

    df = pd.read_csv(DATA_PATH)
    df['alcohol_consumption'] = df['alcohol_consumption'].fillna('None')

    pos_samples = df[df['diabetes'] == 1].head(5)
    neg_samples = df[df['diabetes'] == 0].head(5)

    print("\n--- Actual Diabetic Records (Target = 1) ---")
    pos_probs = []
    for idx, row in pos_samples.iterrows():
        feat_dict = {col: row[col] for col in metadata['feature_names']}
        input_df = pd.DataFrame([feat_dict])
        prob = float(pipeline.predict_proba(input_df)[0][pos_idx])
        pos_probs.append(prob)
        print(f"Row {idx:5d}: HbA1c={row['HbA1c_level']:.1f}%, Glucose={row['blood_glucose_level']} mg/dL, Age={row['age']}y -> Model Risk: {prob*100:5.1f}% (Pred: {int(pipeline.predict(input_df)[0])})")

    print("\n--- Actual Non-Diabetic Records (Target = 0) ---")
    neg_probs = []
    for idx, row in neg_samples.iterrows():
        feat_dict = {col: row[col] for col in metadata['feature_names']}
        input_df = pd.DataFrame([feat_dict])
        prob = float(pipeline.predict_proba(input_df)[0][pos_idx])
        neg_probs.append(prob)
        print(f"Row {idx:5d}: HbA1c={row['HbA1c_level']:.1f}%, Glucose={row['blood_glucose_level']} mg/dL, Age={row['age']}y -> Model Risk: {prob*100:5.1f}% (Pred: {int(pipeline.predict(input_df)[0])})")

    mean_pos_prob = np.mean(pos_probs)
    mean_neg_prob = np.mean(neg_probs)
    print(f"\nMean Risk for Diabetic Samples:     {mean_pos_prob*100:.2f}%")
    print(f"Mean Risk for Non-Diabetic Samples: {mean_neg_prob*100:.2f}%")
    assert mean_pos_prob > mean_neg_prob, "Diabetic samples should have substantially higher risk than non-diabetic samples"

    print("\n" + "=" * 80)
    print("ALL AUTOMATED VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    run_tests()

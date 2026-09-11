"""
Dataset Generator for Early Detection of Type 2 Diabetes Mellitus
=================================================================
Generates an enhanced dataset of 100,000 patient records with 18 features
using clinically-validated statistical distributions and correlations.

Features:
- 8 base clinical features (mirroring Kaggle diabetes prediction dataset)
- 10 enhanced lifestyle and medical features
- Realistic correlations between features and diabetes outcome
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

NUM_SAMPLES = 100000
DIABETES_RATIO = 0.085  # ~8.5% prevalence (realistic global estimate)


def generate_dataset():
    """Generate a comprehensive diabetes dataset with clinically-correlated features."""
    
    n = NUM_SAMPLES
    n_diabetic = int(n * DIABETES_RATIO)
    n_healthy = n - n_diabetic
    
    # --- Generate diabetes labels first ---
    diabetes = np.array([0] * n_healthy + [1] * n_diabetic)
    np.random.shuffle(diabetes)
    
    # --- Helper: generate feature with different distributions for diabetic/healthy ---
    def correlated_feature(healthy_params, diabetic_params, dist='normal', clip=None):
        values = np.zeros(n)
        h_mask = diabetes == 0
        d_mask = diabetes == 1
        
        if dist == 'normal':
            values[h_mask] = np.random.normal(healthy_params[0], healthy_params[1], h_mask.sum())
            values[d_mask] = np.random.normal(diabetic_params[0], diabetic_params[1], d_mask.sum())
        elif dist == 'uniform':
            values[h_mask] = np.random.uniform(healthy_params[0], healthy_params[1], h_mask.sum())
            values[d_mask] = np.random.uniform(diabetic_params[0], diabetic_params[1], d_mask.sum())
        
        if clip:
            values = np.clip(values, clip[0], clip[1])
        return values
    
    def correlated_binary(healthy_prob, diabetic_prob):
        values = np.zeros(n, dtype=int)
        h_mask = diabetes == 0
        d_mask = diabetes == 1
        values[h_mask] = np.random.binomial(1, healthy_prob, h_mask.sum())
        values[d_mask] = np.random.binomial(1, diabetic_prob, d_mask.sum())
        return values
    
    def correlated_categorical(categories, healthy_probs, diabetic_probs):
        values = np.empty(n, dtype=object)
        h_mask = diabetes == 0
        d_mask = diabetes == 1
        values[h_mask] = np.random.choice(categories, h_mask.sum(), p=healthy_probs)
        values[d_mask] = np.random.choice(categories, d_mask.sum(), p=diabetic_probs)
        return values
    
    # ========== BASE FEATURES ==========
    
    # 1. Gender
    gender = correlated_categorical(
        ['Male', 'Female'],
        [0.48, 0.52],   # healthy: slightly more female
        [0.55, 0.45]    # diabetic: slightly more male (literature-supported)
    )
    
    # 2. Age (older → higher risk)
    age = correlated_feature(
        (38, 14),   # healthy: mean 38, std 14
        (55, 12),   # diabetic: mean 55, std 12
        clip=(18, 90)
    ).astype(int)
    
    # 3. Hypertension (strong correlation with diabetes)
    hypertension = correlated_binary(0.08, 0.45)
    
    # 4. Heart Disease
    heart_disease = correlated_binary(0.03, 0.20)
    
    # 5. Smoking History
    smoking_history = correlated_categorical(
        ['never', 'former', 'current', 'not current', 'ever'],
        [0.45, 0.20, 0.15, 0.10, 0.10],   # healthy
        [0.25, 0.30, 0.20, 0.15, 0.10]    # diabetic: more former/current smokers
    )
    
    # 6. BMI (strong predictor)
    bmi = correlated_feature(
        (25.5, 5.5),   # healthy: mean 25.5
        (33.0, 6.0),   # diabetic: mean 33 (obese range)
        clip=(12, 55)
    )
    bmi = np.round(bmi, 1)
    
    # 7. HbA1c Level (PRIMARY diagnostic marker)
    hba1c = correlated_feature(
        (5.0, 0.5),    # healthy: mean 5.0% (normal <5.7)
        (7.2, 1.2),    # diabetic: mean 7.2% (diagnostic ≥6.5)
        clip=(3.5, 14.0)
    )
    hba1c = np.round(hba1c, 1)
    
    # 8. Blood Glucose Level (fasting, mg/dL)
    blood_glucose = correlated_feature(
        (95, 15),      # healthy: mean 95 mg/dL (normal <100)
        (175, 45),     # diabetic: mean 175 mg/dL (diagnostic ≥126)
        clip=(60, 350)
    )
    blood_glucose = np.round(blood_glucose, 0).astype(int)
    
    # ========== ENHANCED FEATURES ==========
    
    # 9. Family History (Yes/No — strongest non-modifiable risk factor)
    family_history = correlated_categorical(
        ['Yes', 'No'],
        [0.15, 0.85],   # healthy: 15% have family history
        [0.55, 0.45]    # diabetic: 55% have family history (2-6x risk)
    )
    
    # 10. Physical Activity
    physical_activity = correlated_categorical(
        ['Sedentary', 'Light', 'Moderate', 'Active'],
        [0.15, 0.25, 0.35, 0.25],   # healthy
        [0.40, 0.30, 0.20, 0.10]    # diabetic: more sedentary
    )
    
    # 11. Diet Quality
    diet_quality = correlated_categorical(
        ['Poor', 'Average', 'Good', 'Excellent'],
        [0.10, 0.30, 0.40, 0.20],   # healthy
        [0.35, 0.35, 0.20, 0.10]    # diabetic: poorer diet
    )
    
    # 12. Stress Level (1-10)
    stress_level = correlated_feature(
        (4.5, 2.0),    # healthy: moderate stress
        (6.8, 1.8),    # diabetic: higher stress
        clip=(1, 10)
    ).astype(int)
    
    # 13. Sleep Hours
    sleep_hours = correlated_feature(
        (7.2, 1.0),    # healthy: ~7.2 hours
        (5.8, 1.5),    # diabetic: less sleep
        clip=(3, 12)
    )
    sleep_hours = np.round(sleep_hours, 1)
    
    # 14. Alcohol Consumption
    alcohol_consumption = correlated_categorical(
        ['None', 'Occasional', 'Moderate', 'Heavy'],
        [0.35, 0.35, 0.20, 0.10],   # healthy
        [0.20, 0.25, 0.30, 0.25]    # diabetic: more heavy drinkers
    )
    
    # 15. Waist Circumference (cm) — central obesity indicator
    # Generate correlated with gender and BMI
    waist = np.zeros(n)
    for i in range(n):
        base = 60 if gender[i] == 'Female' else 70
        waist[i] = base + (bmi[i] - 20) * 1.8 + np.random.normal(0, 5)
    waist = np.clip(waist, 55, 160)
    waist = np.round(waist, 1)
    
    # 16. Total Cholesterol (mg/dL)
    cholesterol = correlated_feature(
        (190, 30),     # healthy: mean 190
        (235, 40),     # diabetic: higher cholesterol
        clip=(100, 400)
    )
    cholesterol = np.round(cholesterol, 0).astype(int)
    
    # 17. Systolic Blood Pressure (mmHg)
    bp_systolic = correlated_feature(
        (118, 12),     # healthy
        (142, 18),     # diabetic: hypertensive range
        clip=(85, 200)
    )
    bp_systolic = np.round(bp_systolic, 0).astype(int)
    
    # 18. Diastolic Blood Pressure (mmHg)
    bp_diastolic = correlated_feature(
        (76, 8),       # healthy
        (90, 12),      # diabetic
        clip=(50, 130)
    )
    bp_diastolic = np.round(bp_diastolic, 0).astype(int)

    # 19. Pregnancies (0 for males; for females, multiparity correlates with higher T2DM and GDM risk)
    pregnancies = np.zeros(n, dtype=int)
    for i in range(n):
        if gender[i] == 'Female':
            if diabetes[i] == 1:
                # Diabetic females: mean ~3.2 pregnancies
                p = int(np.clip(np.random.negative_binomial(3, 0.48), 0, 15))
            else:
                # Healthy females: mean ~1.3 pregnancies
                p = int(np.clip(np.random.negative_binomial(2, 0.62), 0, 10))
            pregnancies[i] = p
    
    # ========== BUILD DATAFRAME ==========
    
    df = pd.DataFrame({
        'gender': gender,
        'age': age,
        'pregnancies': pregnancies,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': hba1c,
        'blood_glucose_level': blood_glucose,
        'family_history': family_history,
        'physical_activity': physical_activity,
        'diet_quality': diet_quality,
        'stress_level': stress_level,
        'sleep_hours': sleep_hours,
        'alcohol_consumption': alcohol_consumption,
        'waist_circumference': waist,
        'cholesterol_total': cholesterol,
        'bp_systolic': bp_systolic,
        'bp_diastolic': bp_diastolic,
        'diabetes': diabetes
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, 'diabetes_enhanced.csv')
    df.to_csv(out_path, index=False)
    
    print("=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total records: {len(df):,}")
    print(f"Features: {len(df.columns) - 1}")
    print(f"Diabetic cases: {df['diabetes'].sum():,} ({df['diabetes'].mean()*100:.1f}%)")
    print(f"Non-diabetic cases: {(df['diabetes'] == 0).sum():,} ({(1-df['diabetes'].mean())*100:.1f}%)")
    print(f"\nSaved to: {out_path}")
    print(f"\nFeature columns:")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype} | unique={df[col].nunique()}")
    
    return df


if __name__ == '__main__':
    df = generate_dataset()
    print(f"\nFirst 5 rows:")
    print(df.head())

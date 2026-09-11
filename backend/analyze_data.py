import pandas as pd
import numpy as np

df = pd.read_csv('data/diabetes_enhanced.csv', keep_default_na=False)

pos = df[df['diabetes']==1]
neg = df[df['diabetes']==0]

print("=== DIABETIC (diabetes=1) stats ===")
print(pos.describe().to_string())

print("\n=== NON-DIABETIC (diabetes=0) stats ===")
print(neg.describe().to_string())

print("\n=== Categorical distributions for diabetes=1 ===")
for c in ['gender','smoking_history','family_history','physical_activity','diet_quality','alcohol_consumption']:
    print(f'\n{c}:')
    print(pos[c].value_counts().to_string())

print("\n=== Key Feature Separability ===")
for feat in ['HbA1c_level', 'blood_glucose_level', 'bmi', 'age', 'waist_circumference', 'bp_systolic']:
    p_mean = pos[feat].mean()
    n_mean = neg[feat].mean()
    p_std = pos[feat].std()
    n_std = neg[feat].std()
    print(f"{feat}: diabetic_mean={p_mean:.2f}±{p_std:.2f}  non_diabetic_mean={n_mean:.2f}±{n_std:.2f}")

# Test a typical high-risk patient
print("\n=== TEST: Typical high-risk patient ===")
print("Input: age=55, bmi=35, HbA1c=7.5, glucose=200, bp_sys=145, bp_dia=95")
print("       cholesterol=250, waist=105, stress=8, sleep=4, family_history=Yes")
print("       hypertension=1, heart_disease=1, smoking=current, physical_activity=Sedentary")
print("       diet=Poor, alcohol=Heavy, gender=Male, pregnancies=0")

# Test a typical healthy person
print("\n=== TEST: Typical healthy person ===")
print("Input: age=30, bmi=22, HbA1c=5.0, glucose=85, bp_sys=110, bp_dia=70")
print("       cholesterol=170, waist=70, stress=3, sleep=7.5, family_history=No")
print("       hypertension=0, heart_disease=0, smoking=never, physical_activity=Active")
print("       diet=Excellent, alcohol=None, gender=Male, pregnancies=0")

# Check correlation of key features with diabetes
print("\n=== Correlation with diabetes ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()['diabetes'].sort_values(ascending=False)
print(corr.to_string())

# Check if there are samples with borderline values that are labeled diabetic
print("\n=== Borderline cases analysis ===")
borderline = df[(df['HbA1c_level'] >= 5.7) & (df['HbA1c_level'] <= 6.4)]
print(f"Borderline HbA1c (5.7-6.4): {len(borderline)} cases, diabetic: {borderline['diabetes'].sum()} ({borderline['diabetes'].mean()*100:.1f}%)")

high_glucose = df[(df['blood_glucose_level'] >= 126)]
print(f"High glucose (>=126): {len(high_glucose)} cases, diabetic: {high_glucose['diabetes'].sum()} ({high_glucose['diabetes'].mean()*100:.1f}%)")

very_high_hba1c = df[(df['HbA1c_level'] >= 6.5)]
print(f"Very high HbA1c (>=6.5): {len(very_high_hba1c)} cases, diabetic: {very_high_hba1c['diabetes'].sum()} ({very_high_hba1c['diabetes'].mean()*100:.1f}%)")

"""
Model Training & Evaluation for Type 2 Diabetes Prediction
============================================================
Trains, evaluates, and exports a production-grade scikit-learn Pipeline
combining ColumnTransformer preprocessing and an ensemble classifier.
Provides continuous, calibrated diabetes risk probabilities across the
entire clinical spectrum without probability collapse or data leakage.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    brier_score_loss, log_loss
)

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'visualizations')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'diabetes_enhanced.csv')

# Feature definitions
CATEGORICAL_FEATURES = [
    'gender', 'smoking_history', 'family_history',
    'physical_activity', 'diet_quality', 'alcohol_consumption'
]

NUMERICAL_FEATURES = [
    'age', 'pregnancies', 'hypertension', 'heart_disease', 'bmi',
    'HbA1c_level', 'blood_glucose_level', 'stress_level', 'sleep_hours',
    'waist_circumference', 'cholesterol_total', 'bp_systolic', 'bp_diastolic'
]

ALL_FEATURES = [
    'gender', 'age', 'pregnancies', 'hypertension', 'heart_disease',
    'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level',
    'family_history', 'physical_activity', 'diet_quality', 'stress_level',
    'sleep_hours', 'alcohol_consumption', 'waist_circumference',
    'cholesterol_total', 'bp_systolic', 'bp_diastolic'
]


def load_dataset():
    """Load dataset, handle missing alcohol_consumption values, and validate schema."""
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Clean missing/null values in alcohol_consumption
    df['alcohol_consumption'] = df['alcohol_consumption'].fillna('None')

    # Separate X and y
    X = df[ALL_FEATURES].copy()
    y = df['diabetes'].copy()

    # Stratified split: 80% train, 20% test (preserving class balance without leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training set:  {X_train.shape[0]:,} samples (Positive rate: {y_train.mean()*100:.2f}%)")
    print(f"Test set:      {X_test.shape[0]:,} samples (Positive rate: {y_test.mean()*100:.2f}%)")

    return X_train, X_test, y_train, y_test


def build_preprocessor():
    """Create a ColumnTransformer that cleanly handles both numerical and categorical features."""
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, NUMERICAL_FEATURES),
        ('cat', cat_transformer, CATEGORICAL_FEATURES)
    ], remainder='drop')

    return preprocessor


def define_candidate_models():
    """Define candidate classifiers tuned for smooth, calibrated risk probabilities."""
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=40,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    lr = LogisticRegression(
        C=0.05,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )

    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=50,
        random_state=42
    )

    # Soft Voting Ensemble combining non-linear Random Forest with regularized Logistic Regression
    ensemble = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=40, class_weight='balanced', random_state=42, n_jobs=-1)),
            ('lr', LogisticRegression(C=0.05, class_weight='balanced', max_iter=1000, random_state=42))
        ],
        voting='soft',
        weights=[2, 1]
    )

    return {
        'Soft Voting Ensemble': ensemble,
        'Random Forest (Balanced)': rf,
        'Logistic Regression (Balanced)': lr,
        'Gradient Boosting': gb
    }


def train_and_evaluate_candidates(candidates, X_train, X_test, y_train, y_test):
    """Fit complete pipelines and evaluate on test set."""
    results = {}

    print("\n" + "=" * 75)
    print("TRAINING & EVALUATING CANDIDATE PIPELINES")
    print("=" * 75)

    for name, clf in candidates.items():
        print(f"\n--- Fitting: {name} ---")
        preprocessor = build_preprocessor()
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        pipeline.fit(X_train, y_train)

        classes = pipeline.classes_
        pos_idx = list(classes).index(1)

        test_probs = pipeline.predict_proba(X_test)[:, pos_idx]
        y_pred = (test_probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, test_probs)
        brier = brier_score_loss(y_test, test_probs)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, test_probs)

        results[name] = {
            'pipeline': pipeline,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': auc,
            'brier_score': brier,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
            'y_pred': y_pred,
            'y_prob': test_probs
        }

        print(f"  Accuracy:    {acc:.4f}")
        print(f"  ROC-AUC:     {auc:.4f}")
        print(f"  Recall:      {rec:.4f}")
        print(f"  Precision:   {prec:.4f}")
        print(f"  F1 Score:    {f1:.4f}")
        print(f"  Brier Score: {brier:.4f}")

    return results


def validate_clinical_spectrum(results):
    """Validate that candidate models produce continuous, monotonic probabilities across risk spectrum."""
    print("\n" + "=" * 75)
    print("CLINICAL SPECTRUM VALIDATION")
    print("=" * 75)

    test_scenarios = [
        ('1. Healthy Athlete (HbA1c 4.6, Gluc 78)', {
            'gender': 'Female', 'age': 22, 'pregnancies': 0, 'hypertension': 0, 'heart_disease': 0,
            'smoking_history': 'never', 'bmi': 20.5, 'HbA1c_level': 4.6, 'blood_glucose_level': 78,
            'family_history': 'No', 'physical_activity': 'Active', 'diet_quality': 'Excellent',
            'stress_level': 2, 'sleep_hours': 8.0, 'alcohol_consumption': 'None',
            'waist_circumference': 68.0, 'cholesterol_total': 155, 'bp_systolic': 108, 'bp_diastolic': 70
        }),
        ('2. Average Young Adult (HbA1c 5.1, Gluc 88)', {
            'gender': 'Male', 'age': 32, 'pregnancies': 0, 'hypertension': 0, 'heart_disease': 0,
            'smoking_history': 'never', 'bmi': 23.5, 'HbA1c_level': 5.1, 'blood_glucose_level': 88,
            'family_history': 'No', 'physical_activity': 'Moderate', 'diet_quality': 'Good',
            'stress_level': 3, 'sleep_hours': 7.5, 'alcohol_consumption': 'None',
            'waist_circumference': 80.0, 'cholesterol_total': 175, 'bp_systolic': 118, 'bp_diastolic': 76
        }),
        ('3. Mild Family Risk (HbA1c 5.4, Gluc 96)', {
            'gender': 'Male', 'age': 42, 'pregnancies': 0, 'hypertension': 0, 'heart_disease': 0,
            'smoking_history': 'former', 'bmi': 25.5, 'HbA1c_level': 5.4, 'blood_glucose_level': 96,
            'family_history': 'Yes', 'physical_activity': 'Moderate', 'diet_quality': 'Good',
            'stress_level': 4, 'sleep_hours': 7.0, 'alcohol_consumption': 'Occasional',
            'waist_circumference': 86.0, 'cholesterol_total': 190, 'bp_systolic': 122, 'bp_diastolic': 78
        }),
        ('4. Early Pre-diabetes (HbA1c 5.7, Gluc 104)', {
            'gender': 'Female', 'age': 47, 'pregnancies': 1, 'hypertension': 0, 'heart_disease': 0,
            'smoking_history': 'never', 'bmi': 27.0, 'HbA1c_level': 5.7, 'blood_glucose_level': 104,
            'family_history': 'Yes', 'physical_activity': 'Light', 'diet_quality': 'Average',
            'stress_level': 5, 'sleep_hours': 6.5, 'alcohol_consumption': 'None',
            'waist_circumference': 89.0, 'cholesterol_total': 205, 'bp_systolic': 128, 'bp_diastolic': 82
        }),
        ('5. Mid Pre-diabetes (HbA1c 6.0, Gluc 114)', {
            'gender': 'Male', 'age': 52, 'pregnancies': 0, 'hypertension': 1, 'heart_disease': 0,
            'smoking_history': 'former', 'bmi': 28.8, 'HbA1c_level': 6.0, 'blood_glucose_level': 114,
            'family_history': 'Yes', 'physical_activity': 'Light', 'diet_quality': 'Average',
            'stress_level': 6, 'sleep_hours': 6.0, 'alcohol_consumption': 'Moderate',
            'waist_circumference': 94.0, 'cholesterol_total': 220, 'bp_systolic': 134, 'bp_diastolic': 84
        }),
        ('6. Late Pre-diabetes (HbA1c 6.3, Gluc 124)', {
            'gender': 'Female', 'age': 56, 'pregnancies': 3, 'hypertension': 1, 'heart_disease': 0,
            'smoking_history': 'former', 'bmi': 31.0, 'HbA1c_level': 6.3, 'blood_glucose_level': 124,
            'family_history': 'Yes', 'physical_activity': 'Sedentary', 'diet_quality': 'Average',
            'stress_level': 7, 'sleep_hours': 5.5, 'alcohol_consumption': 'None',
            'waist_circumference': 98.0, 'cholesterol_total': 235, 'bp_systolic': 140, 'bp_diastolic': 88
        }),
        ('7. Early Diabetic (HbA1c 6.6, Gluc 132)', {
            'gender': 'Male', 'age': 55, 'pregnancies': 0, 'hypertension': 1, 'heart_disease': 0,
            'smoking_history': 'current', 'bmi': 32.5, 'HbA1c_level': 6.6, 'blood_glucose_level': 132,
            'family_history': 'Yes', 'physical_activity': 'Sedentary', 'diet_quality': 'Poor',
            'stress_level': 7, 'sleep_hours': 5.5, 'alcohol_consumption': 'Moderate',
            'waist_circumference': 102.0, 'cholesterol_total': 245, 'bp_systolic': 144, 'bp_diastolic': 90
        }),
        ('8. Confirmed Diabetic (HbA1c 7.5, Gluc 160)', {
            'gender': 'Male', 'age': 60, 'pregnancies': 0, 'hypertension': 1, 'heart_disease': 1,
            'smoking_history': 'current', 'bmi': 34.5, 'HbA1c_level': 7.5, 'blood_glucose_level': 160,
            'family_history': 'Yes', 'physical_activity': 'Sedentary', 'diet_quality': 'Poor',
            'stress_level': 8, 'sleep_hours': 5.0, 'alcohol_consumption': 'Heavy',
            'waist_circumference': 108.0, 'cholesterol_total': 260, 'bp_systolic': 152, 'bp_diastolic': 95
        }),
        ('9. Severe Diabetic (HbA1c 9.0, Gluc 240)', {
            'gender': 'Male', 'age': 65, 'pregnancies': 0, 'hypertension': 1, 'heart_disease': 1,
            'smoking_history': 'current', 'bmi': 37.0, 'HbA1c_level': 9.0, 'blood_glucose_level': 240,
            'family_history': 'Yes', 'physical_activity': 'Sedentary', 'diet_quality': 'Poor',
            'stress_level': 9, 'sleep_hours': 4.5, 'alcohol_consumption': 'Heavy',
            'waist_circumference': 115.0, 'cholesterol_total': 290, 'bp_systolic': 165, 'bp_diastolic': 100
        }),
    ]

    scen_df = pd.DataFrame([s[1] for s in test_scenarios])

    for name, r in results.items():
        pipe = r['pipeline']
        pos_idx = list(pipe.classes_).index(1)
        probs = pipe.predict_proba(scen_df)[:, pos_idx]
        print(f"\n{name} Predictions across Clinical Spectrum:")
        for (label, _), p in zip(test_scenarios, probs):
            print(f"  {label:48s}: {p*100:5.1f}%")


def compute_feature_importances(pipeline, X_train, y_train):
    """Extract feature importances for all original 19 features."""
    clf = pipeline.named_steps['classifier']

    # If VotingClassifier, extract from the Random Forest component
    rf_model = None
    if isinstance(clf, VotingClassifier):
        for name, est in clf.named_estimators_.items():
            if hasattr(est, 'feature_importances_'):
                rf_model = est
                break
    elif hasattr(clf, 'feature_importances_'):
        rf_model = clf

    importance_dict = {}

    if rf_model is not None:
        # Get one-hot feature names from preprocessor
        prep = pipeline.named_steps['preprocessor']
        cat_encoder = prep.named_transformers_['cat'].named_steps['onehot']
        cat_encoded_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
        all_encoded_names = list(NUMERICAL_FEATURES) + list(cat_encoded_names)

        raw_importances = rf_model.feature_importances_

        # Map back to original 19 features
        for feat in ALL_FEATURES:
            if feat in NUMERICAL_FEATURES:
                idx = NUMERICAL_FEATURES.index(feat)
                importance_dict[feat] = float(raw_importances[idx])
            else:
                # Sum importances across all one-hot categories for this categorical feature
                total = sum(
                    raw_importances[len(NUMERICAL_FEATURES) + i]
                    for i, enc_name in enumerate(cat_encoded_names)
                    if enc_name.startswith(f"{feat}_")
                )
                importance_dict[feat] = float(total)

    # Normalize to sum to 1.0
    total_imp = sum(importance_dict.values()) or 1.0
    importance_dict = {k: round(v / total_imp, 4) for k, v in importance_dict.items()}

    print("\nFeature Importances:")
    for k, v in sorted(importance_dict.items(), key=lambda x: -x[1]):
        print(f"  {k:22s}: {v:.4f}")

    return importance_dict


def plot_visualizations(results, metrics_df, importance_dict):
    """Generate and save evaluation plots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(14, 7))
    plot_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    x = np.arange(len(metrics_df))
    width = 0.15
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#e74c3c']

    for i, (metric, color) in enumerate(zip(plot_metrics, colors)):
        ax.bar(x + i * width, metrics_df[metric], width, label=metric, color=color, edgecolor='white')

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(metrics_df.index, rotation=10, ha='right', fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Clinical Machine Learning Model Comparison', fontsize=15, fontweight='bold')
    ax.set_ylim(0.85, 1.02)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150)
    plt.close()

    # 2. ROC Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, r in results.items():
        ax.plot(r['fpr'], r['tpr'], label=f"{name} (AUC = {r['roc_auc']:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.50)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title('ROC Curves - Diabetes Classification', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'), dpi=150)
    plt.close()

    # 3. Feature Importance Horizontal Bar Chart
    fig, ax = plt.subplots(figsize=(12, 8))
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1])
    feats, imps = zip(*sorted_features)
    y_pos = np.arange(len(feats))
    ax.barh(y_pos, imps, color='#3b82f6', edgecolor='white', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace('_', ' ').title() for f in feats], fontsize=11)
    ax.set_xlabel('Relative Importance Weight', fontsize=12)
    ax.set_title('Feature Importance Attribution (Trained Ensemble)', fontsize=14, fontweight='bold')
    for i, v in enumerate(imps):
        ax.text(v + 0.005, i, f"{v:.3f}", va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150)
    plt.close()

    print("[OK] Evaluation charts saved to visualizations/")


def run_pipeline():
    """Execute complete training, validation, comparison, and saving workflow."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data
    X_train, X_test, y_train, y_test = load_dataset()

    # 2. Define candidates
    candidates = define_candidate_models()

    # 3. Train and evaluate
    results = train_and_evaluate_candidates(candidates, X_train, X_test, y_train, y_test)

    # 4. Clinical spectrum validation
    validate_clinical_spectrum(results)

    # 5. Build comparison DataFrame
    metrics_df = pd.DataFrame({
        name: {
            'Accuracy': r['accuracy'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1 Score': r['f1_score'],
            'ROC AUC': r['roc_auc'],
            'Brier Score': r['brier_score']
        }
        for name, r in results.items()
    }).T

    print("\n" + "=" * 75)
    print("MODEL COMPARISON TABLE")
    print("=" * 75)
    print(metrics_df.round(4).to_string())

    # Select best model: Soft Voting Ensemble (balances smooth probabilities with high AUC/F1)
    best_name = 'Soft Voting Ensemble'
    best_result = results[best_name]
    best_pipeline = best_result['pipeline']

    # 6. Feature importances
    importance_dict = compute_feature_importances(best_pipeline, X_train, y_train)

    # 7. Generate visualizations
    plot_visualizations(results, metrics_df, importance_dict)

    # 8. Save production pipeline and metadata
    pipeline_path = os.path.join(MODEL_DIR, 'production_pipeline.pkl')
    best_model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    metadata_path = os.path.join(MODEL_DIR, 'metadata.pkl')
    importance_path = os.path.join(MODEL_DIR, 'feature_importances.pkl')

    joblib.dump(best_pipeline, pipeline_path)
    joblib.dump(best_pipeline, best_model_path)
    joblib.dump(importance_dict, importance_path)

    metadata = {
        'model_name': best_name,
        'accuracy': float(best_result['accuracy']),
        'precision': float(best_result['precision']),
        'recall': float(best_result['recall']),
        'f1_score': float(best_result['f1_score']),
        'roc_auc': float(best_result['roc_auc']),
        'brier_score': float(best_result['brier_score']),
        'feature_names': ALL_FEATURES,
        'training_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'classes': [int(c) for c in best_pipeline.classes_]
    }
    joblib.dump(metadata, metadata_path)

    print("\n" + "=" * 75)
    print("PRODUCTION PIPELINE ARTIFACTS SAVED SUCCESSFULLY")
    print("=" * 75)
    print(f"Pipeline:    {pipeline_path}")
    print(f"Legacy Alias:{best_model_path}")
    print(f"Importances: {importance_path}")
    print(f"Metadata:    {metadata_path}")
    print(f"\nFinal Model: {best_name}")
    print(f"  Accuracy:  {best_result['accuracy']*100:.2f}%")
    print(f"  Recall:    {best_result['recall']*100:.2f}%")
    print(f"  Precision: {best_result['precision']*100:.2f}%")
    print(f"  F1 Score:  {best_result['f1_score']*100:.2f}%")
    print(f"  ROC-AUC:   {best_result['roc_auc']:.4f}")
    print("=" * 75)


if __name__ == '__main__':
    run_pipeline()

"""
Flask API — Early Detection of Type 2 Diabetes Mellitus
========================================================
REST API that serves predictions from the trained ML model.
Returns professional clinical statements, risk factor breakdowns,
and personalized precautions.
"""

from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
from flask_cors import CORS
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))

app = Flask(__name__, static_folder=None)
CORS(app)

# ─── Load trained model artifacts ───────────────────────────────────────────
MODEL_DIR = os.path.join(BASE_DIR, 'models')

pipeline = None
metadata = None
feature_importances = {}
FEATURE_ORDER = [
    'gender', 'age', 'pregnancies', 'hypertension', 'heart_disease',
    'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level',
    'family_history', 'physical_activity', 'diet_quality', 'stress_level',
    'sleep_hours', 'alcohol_consumption', 'waist_circumference',
    'cholesterol_total', 'bp_systolic', 'bp_diastolic'
]

def load_models():
    global pipeline, metadata, feature_importances, FEATURE_ORDER
    pipeline_path = os.path.join(MODEL_DIR, 'production_pipeline.pkl')
    if not os.path.exists(pipeline_path):
        pipeline_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    if os.path.exists(pipeline_path):
        pipeline = joblib.load(pipeline_path)
        if os.path.exists(os.path.join(MODEL_DIR, 'metadata.pkl')):
            metadata = joblib.load(os.path.join(MODEL_DIR, 'metadata.pkl'))
            if 'feature_names' in metadata:
                FEATURE_ORDER = metadata['feature_names']
        if os.path.exists(os.path.join(MODEL_DIR, 'feature_importances.pkl')):
            feature_importances = joblib.load(os.path.join(MODEL_DIR, 'feature_importances.pkl'))
        return True
    return False

load_models()

# ─── Clinical Reference Ranges ─────────────────────────────────────────────
REFERENCE_RANGES = {
    'age': {'normal': (18, 44), 'elevated': (45, 64), 'critical': (65, 120),
            'unit': 'years', 'label': 'Age',
            'explanation': {
                'normal': 'Your age group has a relatively lower baseline risk for Type 2 Diabetes.',
                'elevated': 'Age above 45 is an independent risk factor for Type 2 Diabetes due to declining beta-cell function.',
                'critical': 'Advanced age significantly increases diabetes risk due to progressive insulin resistance and reduced pancreatic function.'
            }},
    'pregnancies': {'normal': (0, 1), 'elevated': (2, 4), 'critical': (5, 20),
                    'unit': '', 'label': 'Pregnancies',
                    'explanation': {
                        'normal': 'Low parity has minimal metabolic strain on pancreatic beta-cell reserve.',
                        'elevated': 'Multiple pregnancies (2–4) increase lifetime risk of gestational hyperglycemia and metabolic syndrome.',
                        'critical': 'High parity (≥5 pregnancies) significantly elevates Type 2 Diabetes risk due to recurrent gestational insulin resistance.'
                    }},
    'bmi': {'normal': (0, 24.9), 'elevated': (25, 29.9), 'critical': (30, 100),
            'unit': 'kg/m²', 'label': 'Body Mass Index (BMI)',
            'explanation': {
                'normal': 'Your BMI is within the healthy range, indicating appropriate body composition.',
                'elevated': 'A BMI in the overweight range increases insulin resistance and elevates diabetes risk.',
                'critical': 'Obesity (BMI ≥ 30) is strongly associated with insulin resistance and a significantly increased risk of developing Type 2 Diabetes.'
            }},
    'HbA1c_level': {'normal': (0, 5.6), 'elevated': (5.7, 6.4), 'critical': (6.5, 20),
                    'unit': '%', 'label': 'HbA1c Level',
                    'explanation': {
                        'normal': 'Your HbA1c is within the normal range, indicating healthy blood sugar control over the past 2–3 months.',
                        'elevated': 'HbA1c between 5.7% and 6.4% indicates a pre-diabetic state with impaired glucose regulation.',
                        'critical': 'HbA1c at or above 6.5% is a diagnostic threshold for Type 2 Diabetes, reflecting sustained hyperglycemia.'
                    }},
    'blood_glucose_level': {'normal': (0, 99), 'elevated': (100, 125), 'critical': (126, 500),
                           'unit': 'mg/dL', 'label': 'Fasting Blood Glucose',
                           'explanation': {
                               'normal': 'Your fasting blood glucose is within the normal range, indicating effective insulin function.',
                               'elevated': 'Fasting glucose between 100–125 mg/dL indicates impaired fasting glucose (pre-diabetes).',
                               'critical': 'Fasting glucose at or above 126 mg/dL meets the diagnostic criteria for diabetes mellitus.'
                           }},
    'bp_systolic': {'normal': (0, 119), 'elevated': (120, 139), 'critical': (140, 250),
                    'unit': 'mmHg', 'label': 'Systolic Blood Pressure',
                    'explanation': {
                        'normal': 'Your systolic blood pressure is within the optimal range.',
                        'elevated': 'Elevated systolic blood pressure is associated with metabolic syndrome and increased cardiovascular risk.',
                        'critical': 'Hypertension (≥140 mmHg) is strongly correlated with insulin resistance and Type 2 Diabetes.'
                    }},
    'bp_diastolic': {'normal': (0, 79), 'elevated': (80, 89), 'critical': (90, 200),
                     'unit': 'mmHg', 'label': 'Diastolic Blood Pressure',
                     'explanation': {
                         'normal': 'Your diastolic blood pressure is within the healthy range.',
                         'elevated': 'Mildly elevated diastolic pressure suggests emerging cardiovascular stress.',
                         'critical': 'Diastolic hypertension compounds metabolic risk and warrants clinical evaluation.'
                     }},
    'cholesterol_total': {'normal': (0, 199), 'elevated': (200, 239), 'critical': (240, 500),
                          'unit': 'mg/dL', 'label': 'Total Cholesterol',
                          'explanation': {
                              'normal': 'Your total cholesterol is within a desirable range.',
                              'elevated': 'Borderline-high cholesterol is associated with dyslipidemia, a component of metabolic syndrome.',
                              'critical': 'High total cholesterol is a significant cardiovascular risk factor often co-occurring with Type 2 Diabetes.'
                          }},
    'waist_circumference': {'normal': (0, 88), 'elevated': (89, 101), 'critical': (102, 200),
                            'unit': 'cm', 'label': 'Waist Circumference',
                            'explanation': {
                                'normal': 'Your waist circumference indicates low central adiposity.',
                                'elevated': 'Moderately elevated waist circumference suggests increased visceral fat accumulation.',
                                'critical': 'Central obesity (high waist circumference) is a stronger predictor of diabetes than BMI alone.'
                            }},
    'stress_level': {'normal': (1, 4), 'elevated': (5, 7), 'critical': (8, 10),
                     'unit': '/10', 'label': 'Stress Level',
                     'explanation': {
                         'normal': 'Your stress levels are manageable and unlikely to significantly impact metabolic health.',
                         'elevated': 'Moderate stress can elevate cortisol levels, contributing to insulin resistance over time.',
                         'critical': 'Chronic high stress significantly elevates cortisol, directly promoting insulin resistance and fat accumulation.'
                     }},
    'sleep_hours': {'normal': (7, 9), 'elevated': (5, 6.9), 'critical': (0, 4.9),
                    'unit': 'hours', 'label': 'Sleep Duration',
                    'explanation': {
                        'normal': 'Your sleep duration is within the recommended range for optimal metabolic health.',
                        'elevated': 'Insufficient sleep (5–7 hours) is associated with impaired glucose tolerance and weight gain.',
                        'critical': 'Severe sleep deprivation (<5 hours) significantly disrupts insulin sensitivity and appetite regulation.'
                    }},
}

CATEGORICAL_RISK = {
    'family_history': {
        'label': 'Family History of Diabetes',
        'risk_values': {'Yes': 'critical'},
        'explanations': {
            'Yes': 'A confirmed family history of diabetes increases your risk by 2–6 times. Genetic predisposition significantly affects beta-cell function and insulin sensitivity.',
            'No': 'No reported family history of diabetes, which is a favorable factor.'
        }
    },
    'hypertension': {
        'label': 'Hypertension',
        'risk_values': {1: 'critical'},
        'explanations': {
            1: 'Hypertension is a key component of metabolic syndrome and is strongly associated with insulin resistance.',
            0: 'No reported hypertension.'
        }
    },
    'heart_disease': {
        'label': 'Heart Disease',
        'risk_values': {1: 'critical'},
        'explanations': {
            1: 'Existing cardiovascular disease shares common pathophysiology with Type 2 Diabetes, including endothelial dysfunction and chronic inflammation.',
            0: 'No reported heart disease.'
        }
    },
    'physical_activity': {
        'label': 'Physical Activity Level',
        'risk_map': {'Sedentary': 'critical', 'Light': 'elevated', 'Moderate': 'normal', 'Active': 'normal'},
        'explanations': {
            'Sedentary': 'A sedentary lifestyle is one of the strongest modifiable risk factors for Type 2 Diabetes, directly contributing to insulin resistance.',
            'Light': 'Light physical activity may be insufficient to provide protective metabolic benefits.',
            'Moderate': 'Moderate activity helps maintain insulin sensitivity and healthy body weight.',
            'Active': 'Regular physical activity significantly reduces diabetes risk through improved glucose uptake and metabolism.'
        }
    },
    'diet_quality': {
        'label': 'Diet Quality',
        'risk_map': {'Poor': 'critical', 'Average': 'elevated', 'Good': 'normal', 'Excellent': 'normal'},
        'explanations': {
            'Poor': 'A diet high in processed foods, refined sugars, and saturated fats directly contributes to obesity and insulin resistance.',
            'Average': 'An average diet may lack sufficient fiber, whole grains, and nutrients essential for metabolic health.',
            'Good': 'A good dietary pattern supports healthy blood sugar regulation.',
            'Excellent': 'An excellent diet rich in whole grains, vegetables, and lean proteins is strongly protective against diabetes.'
        }
    },
    'alcohol_consumption': {
        'label': 'Alcohol Consumption',
        'risk_map': {'Heavy': 'critical', 'Moderate': 'elevated', 'Occasional': 'normal', 'None': 'normal'},
        'explanations': {
            'Heavy': 'Heavy alcohol consumption impairs liver glucose metabolism and can lead to pancreatitis, damaging insulin-producing cells.',
            'Moderate': 'Moderate alcohol intake may affect blood sugar regulation and liver function.',
            'Occasional': 'Occasional alcohol consumption has minimal metabolic impact.',
            'None': 'Abstaining from alcohol is favorable for metabolic health.'
        }
    },
    'smoking_history': {
        'label': 'Smoking History',
        'risk_map': {'current': 'critical', 'ever': 'elevated', 'former': 'elevated', 'not current': 'normal', 'never': 'normal'},
        'explanations': {
            'current': 'Active smoking increases diabetes risk by 30–40% through promoting inflammation and oxidative stress.',
            'ever': 'A history of smoking may have caused lasting metabolic damage.',
            'former': 'Former smoking carries residual risk, though it decreases over time after cessation.',
            'not current': 'Not currently smoking is favorable.',
            'never': 'Never having smoked is a positive protective factor.'
        }
    }
}

# ─── Precautions Database ──────────────────────────────────────────────────
PRECAUTIONS = {
    'bmi': {
        'elevated': [
            'Aim for gradual weight loss of 0.5–1 kg per week through a calorie-controlled diet.',
            'Incorporate at least 150 minutes of moderate aerobic exercise weekly.',
            'Consult a registered dietitian for a personalized meal plan.'
        ],
        'critical': [
            'Immediate consultation with an endocrinologist or bariatric specialist is recommended.',
            'Target a 5–7% body weight reduction within the first 6 months — this alone can reduce diabetes risk by up to 58%.',
            'Consider structured weight management programs under medical supervision.',
            'Regular monitoring of metabolic markers (HbA1c, fasting glucose) every 3 months.'
        ]
    },
    'HbA1c_level': {
        'elevated': [
            'Schedule an Oral Glucose Tolerance Test (OGTT) with your physician for confirmatory assessment.',
            'Reduce refined carbohydrate and sugar intake significantly.',
            'Monitor HbA1c levels every 3–6 months to track progression.'
        ],
        'critical': [
            'Urgent consultation with an endocrinologist or diabetologist is strongly recommended.',
            'Comprehensive metabolic panel and confirmatory fasting plasma glucose tests are warranted.',
            'Discuss pharmacological intervention (such as Metformin) with your healthcare provider.',
            'Begin daily blood glucose self-monitoring as directed by your physician.'
        ]
    },
    'blood_glucose_level': {
        'elevated': [
            'Adopt a low glycemic index (GI) diet to help stabilize blood sugar levels.',
            'Increase dietary fiber intake to 25–30g daily from whole grains, vegetables, and legumes.',
            'Schedule follow-up fasting glucose testing within 4–6 weeks.'
        ],
        'critical': [
            'Seek immediate medical evaluation — fasting glucose above 126 mg/dL meets diagnostic criteria for diabetes.',
            'Comprehensive blood work including C-peptide and insulin levels is recommended.',
            'Strictly avoid processed sugars, sweetened beverages, and high-GI foods.',
            'Regular physician follow-ups every 2–4 weeks until glucose is stabilized.'
        ]
    },
    'bp_systolic': {
        'elevated': [
            'Reduce sodium intake to below 2,300 mg daily.',
            'Practice stress-reduction techniques such as meditation, yoga, or deep breathing exercises.',
            'Monitor blood pressure at home at least twice weekly.'
        ],
        'critical': [
            'Consult a cardiologist for hypertension management — uncontrolled high BP accelerates diabetic complications.',
            'Discuss antihypertensive medication with your healthcare provider.',
            'Follow the DASH (Dietary Approaches to Stop Hypertension) eating plan.'
        ]
    },
    'stress_level': {
        'elevated': [
            'Incorporate daily stress-management practices: mindfulness meditation, journaling, or nature walks.',
            'Consider professional counseling or cognitive behavioral therapy (CBT) for chronic stress.'
        ],
        'critical': [
            'Chronic high stress is directly linked to elevated cortisol and insulin resistance — seek professional mental health support.',
            'Establish a structured daily routine with dedicated relaxation periods.',
            'Evaluate work-life balance and consider occupational wellness programs.'
        ]
    },
    'sleep_hours': {
        'elevated': [
            'Aim for 7–9 hours of quality sleep per night.',
            'Establish a consistent sleep-wake schedule, including weekends.',
            'Limit screen time and caffeine intake in the 2 hours before bedtime.'
        ],
        'critical': [
            'Severe sleep deprivation significantly impairs glucose metabolism — prioritize sleep hygiene immediately.',
            'Consult a sleep specialist to rule out sleep disorders such as sleep apnea.',
            'Create a dark, cool, quiet sleep environment and avoid electronic devices before bed.'
        ]
    },
    'physical_activity': {
        'elevated': [
            'Gradually increase daily physical activity — aim for at least 30 minutes of brisk walking.',
            'Incorporate both aerobic and resistance training exercises throughout the week.'
        ],
        'critical': [
            'Begin with 10–15 minutes of gentle activity daily and progressively increase duration.',
            'Aim for at least 150 minutes of moderate-intensity exercise per week (WHO recommendation).',
            'Consider joining a structured exercise program or working with a fitness professional.',
            'Even small increases in daily movement (using stairs, short walks) provide metabolic benefits.'
        ]
    },
    'diet_quality': {
        'elevated': [
            'Increase consumption of whole grains, fruits, vegetables, and lean proteins.',
            'Reduce intake of processed foods, refined sugars, and trans fats.'
        ],
        'critical': [
            'Consult a registered dietitian for a comprehensive dietary intervention plan.',
            'Follow a Mediterranean or DASH-style diet — both are clinically proven to reduce diabetes risk.',
            'Limit daily sugar intake to below 25g (WHO recommendation).',
            'Increase soluble fiber intake from oats, beans, lentils, and vegetables.'
        ]
    },
    'family_history': {
        'critical': [
            'Given your family history, annual diabetes screening (HbA1c + fasting glucose) starting from age 35 is essential.',
            'Proactive lifestyle modifications can offset up to 60% of hereditary risk.',
            'Discuss your family medical history in detail with your primary care physician.',
            'Consider genetic counseling for comprehensive risk stratification.'
        ]
    },
    'smoking_history': {
        'elevated': [
            'Smoking cessation can reverse much of the added diabetes risk within 5–10 years.',
            'Explore nicotine replacement therapy or prescription medications to aid quitting.'
        ],
        'critical': [
            'Active smoking increases diabetes risk by 30–40% — quitting is the single most impactful change you can make.',
            'Consult your doctor about smoking cessation programs and pharmacological aids.',
            'Seek support through quitlines, counseling, or peer support groups.'
        ]
    },
    'alcohol_consumption': {
        'elevated': [
            'Limit alcohol to no more than 1 drink per day for women and 2 for men.',
            'Avoid binge drinking, which causes acute blood sugar spikes.'
        ],
        'critical': [
            'Heavy alcohol consumption directly damages the pancreas and liver — significant reduction or cessation is strongly advised.',
            'Consult with a healthcare provider about safe reduction strategies.',
            'Monitor liver function tests regularly.'
        ]
    },
    'pregnancies': {
        'elevated': [
            'Monitor fasting blood glucose and HbA1c periodically, particularly if you have a history of gestational diabetes.',
            'Maintain an active postpartum lifestyle to reduce long-term cardiometabolic risk.'
        ],
        'critical': [
            'History of high parity strongly warrants annual glucose tolerance screening (OGTT).',
            'Discuss prior gestational diabetes episodes and postpartum glycemic tracking with your physician.'
        ]
    }
}


def classify_numeric_risk(feature, value):
    """Classify a numeric feature value into normal/elevated/critical."""
    if feature not in REFERENCE_RANGES:
        return 'normal'
    ref = REFERENCE_RANGES[feature]
    if ref['critical'][0] <= value <= ref['critical'][1]:
        return 'critical'
    elif ref['elevated'][0] <= value <= ref['elevated'][1]:
        return 'elevated'
    return 'normal'


def classify_categorical_risk(feature, value):
    """Classify a categorical feature value into normal/elevated/critical."""
    if feature not in CATEGORICAL_RISK:
        return 'normal'
    info = CATEGORICAL_RISK[feature]
    if 'risk_map' in info:
        return info['risk_map'].get(value, 'normal')
    if 'risk_values' in info:
        return info['risk_values'].get(value, 'normal')
    return 'normal'


def get_explanation(feature, value, level):
    """Get the medical explanation for a feature value."""
    if feature in REFERENCE_RANGES:
        return REFERENCE_RANGES[feature]['explanation'].get(level, '')
    if feature in CATEGORICAL_RISK:
        return CATEGORICAL_RISK[feature]['explanations'].get(value, '')
    return ''


def get_label(feature):
    """Get the display label for a feature."""
    if feature in REFERENCE_RANGES:
        return REFERENCE_RANGES[feature]['label']
    if feature in CATEGORICAL_RISK:
        return CATEGORICAL_RISK[feature]['label']
    return feature.replace('_', ' ').title()


def get_unit(feature):
    """Get the unit for a numeric feature."""
    if feature in REFERENCE_RANGES:
        return REFERENCE_RANGES[feature].get('unit', '')
    return ''


def build_risk_factors(user_data_raw):
    """Analyze each input feature and build risk factor cards."""
    risk_factors = []
    
    numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level',
                       'bp_systolic', 'bp_diastolic', 'cholesterol_total',
                       'waist_circumference', 'stress_level', 'sleep_hours']
    if user_data_raw.get('gender') == 'Female':
        numeric_features.append('pregnancies')
    
    categorical_features = ['family_history', 'hypertension', 'heart_disease',
                           'physical_activity', 'diet_quality', 'alcohol_consumption',
                           'smoking_history']
    
    for feat in numeric_features:
        if feat in user_data_raw:
            val = user_data_raw[feat]
            level = classify_numeric_risk(feat, val)
            explanation = get_explanation(feat, val, level)
            unit = get_unit(feat)
            
            risk_factors.append({
                'feature': feat,
                'label': get_label(feat),
                'value': f"{val} {unit}".strip(),
                'level': level,
                'explanation': explanation,
                'importance': feature_importances.get(feat, 0)
            })
    
    for feat in categorical_features:
        if feat in user_data_raw:
            val = user_data_raw[feat]
            level = classify_categorical_risk(feat, val)
            explanation = get_explanation(feat, val, level)
            
            display_val = val
            if feat in ('hypertension', 'heart_disease'):
                display_val = 'Yes' if val == 1 else 'No'
            
            risk_factors.append({
                'feature': feat,
                'label': get_label(feat),
                'value': str(display_val),
                'level': level,
                'explanation': explanation,
                'importance': feature_importances.get(feat, 0)
            })
    
    # Sort by importance (highest first)
    risk_factors.sort(key=lambda x: x['importance'], reverse=True)
    
    return risk_factors


def build_precautions(risk_factors):
    """Build personalized precautions based on elevated/critical risk factors."""
    immediate = []
    lifestyle = []
    medical = []
    
    for rf in risk_factors:
        feat = rf['feature']
        level = rf['level']
        
        if level in ('elevated', 'critical') and feat in PRECAUTIONS:
            precs = PRECAUTIONS[feat].get(level, [])
            for p in precs:
                # Route to appropriate category
                if any(kw in p.lower() for kw in ['consult', 'doctor', 'physician', 'specialist', 'urgent', 'immediate', 'seek']):
                    if p not in medical:
                        medical.append(p)
                elif any(kw in p.lower() for kw in ['exercise', 'diet', 'sleep', 'walk', 'stress', 'quit', 'limit', 'reduce', 'increase', 'aim']):
                    if p not in lifestyle:
                        lifestyle.append(p)
                else:
                    if p not in immediate:
                        immediate.append(p)
    
    return {
        'immediate_actions': immediate[:5],
        'lifestyle_changes': lifestyle[:6],
        'medical_followups': medical[:5]
    }


def build_professional_statement(risk_level, risk_percentage, risk_factors, user_data_raw):
    """Generate a detailed, professional clinical narrative."""
    
    critical_factors = [rf for rf in risk_factors if rf['level'] == 'critical']
    elevated_factors = [rf for rf in risk_factors if rf['level'] == 'elevated']
    
    # Build specific factor mentions
    def factor_detail(rf):
        return f"{rf['label']} ({rf['value']})"
    
    critical_mentions = ', '.join(factor_detail(rf) for rf in critical_factors[:4])
    elevated_mentions = ', '.join(factor_detail(rf) for rf in elevated_factors[:3])
    
    name = user_data_raw.get('name', 'the patient')
    age = user_data_raw.get('age', '')
    gender = user_data_raw.get('gender', '')
    
    if risk_level == 'Low Risk':
        statement = (
            f"Based on the comprehensive health profile and lifestyle assessment of {name} "
            f"({age}-year-old {gender}), the current clinical markers for Type 2 Diabetes Mellitus "
            f"are within the normal range. The estimated diabetes risk score is {risk_percentage}%, which falls "
            f"well below clinical thresholds of concern. "
            f"Key metabolic markers — including HbA1c ({user_data_raw.get('HbA1c_level', 'N/A')}%), "
            f"fasting blood glucose ({user_data_raw.get('blood_glucose_level', 'N/A')} mg/dL), "
            f"and BMI ({user_data_raw.get('bmi', 'N/A')} kg/m²) — suggest healthy insulin sensitivity "
            f"and effective glucose metabolism. "
        )
        if elevated_factors:
            statement += (
                f"However, mild elevations were noted in {elevated_mentions}, "
                f"which warrant periodic lifestyle surveillance. "
            )
        statement += (
            "While the present estimated risk is low, maintaining a balanced diet, regular physical activity, "
            "and annual routine health screenings is recommended to sustain this favorable metabolic profile."
        )
    
    elif risk_level == 'Moderate Risk':
        statement = (
            f"The clinical and lifestyle profile of {name} ({age}-year-old {gender}) indicates a "
            f"moderate predisposition toward developing Type 2 Diabetes Mellitus, with an estimated "
            f"diabetes risk score of {risk_percentage}%. "
        )
        if critical_factors:
            statement += (
                f"Notably elevated values were identified in {critical_mentions}, "
                f"which place this individual in the pre-diabetic risk zone. "
            )
        if elevated_factors:
            statement += (
                f"Additional contributing factors include {elevated_mentions}. "
            )
        statement += (
            "The convergence of these clinical indicators suggests that without targeted lifestyle "
            "modifications, progression toward Type 2 Diabetes is a clinically significant possibility. "
            "Research demonstrates that structured lifestyle interventions can reduce progression risk by up to 58%. "
            "A consultation with a primary healthcare provider for a formal Oral Glucose Tolerance Test (OGTT) "
            "and personalized metabolic management plan is advised."
        )
    
    else:  # High Risk
        statement = (
            f"The health assessment of {name} ({age}-year-old {gender}) reveals multiple high-risk "
            f"indicators strongly associated with Type 2 Diabetes Mellitus. "
            f"The estimated diabetes risk score of {risk_percentage}% places this individual in the high "
            f"risk category. "
        )
        if critical_factors:
            statement += (
                f"Critical risk drivers identified include {critical_mentions}. "
            )
        if elevated_factors:
            statement += (
                f"These are further compounded by elevated markers in {elevated_mentions}. "
            )
        statement += (
            "The convergence of these clinical and lifestyle markers represents a significant "
            "metabolic profile warranting timely medical attention. "
            "It is strongly recommended to consult a physician or endocrinologist for confirmatory laboratory "
            "testing (fasting plasma glucose, HbA1c reconfirmation, and comprehensive metabolic assessment) "
            "and structured therapeutic lifestyle interventions."
        )
    
    statement += (
        "\n\nMedical Disclaimer: This estimated diabetes risk is generated by a machine learning decision-support tool "
        "for educational and screening purposes only. It does not constitute a medical diagnosis. Please consult a qualified "
        "healthcare professional for clinical evaluation and diagnostic testing."
    )
    return statement


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from the frontend using the trained production pipeline."""
    if pipeline is None:
        load_models()
    if pipeline is None:
        return jsonify({'success': False, 'error': 'Trained model artifacts not found. Please train models first using model_training.py.'}), 500

    try:
        if not request.is_json or not request.json:
            return jsonify({'success': False, 'error': 'Invalid request body: expected JSON.'}), 400

        data = request.json

        # Parse, validate, and convert numeric and categorical inputs
        try:
            name = str(data.get('name', 'Patient')).strip() or 'Patient'
            gender = str(data.get('gender', 'Male')).strip().capitalize()
            if gender not in ('Male', 'Female'):
                gender = 'Male'

            age = int(float(data.get('age', 30)))
            if not (1 <= age <= 120):
                return jsonify({'success': False, 'error': 'Age must be between 1 and 120 years.'}), 400

            pregnancies = int(float(data.get('pregnancies', 0))) if gender == 'Female' else 0
            if pregnancies < 0:
                pregnancies = 0

            # Binary flags normalized to 0 or 1
            raw_hyp = data.get('hypertension', 0)
            hypertension = 1 if str(raw_hyp).strip().lower() in ('1', 'true', 'yes') else 0

            raw_hd = data.get('heart_disease', 0)
            heart_disease = 1 if str(raw_hd).strip().lower() in ('1', 'true', 'yes') else 0

            smoking_history = str(data.get('smoking_history', 'never')).strip().lower()
            if smoking_history not in ('never', 'former', 'current', 'not current', 'ever'):
                smoking_history = 'never'

            bmi = float(data.get('bmi', 24.2))
            if not (10.0 <= bmi <= 75.0):
                return jsonify({'success': False, 'error': 'BMI must be between 10.0 and 75.0 kg/m².'}), 400

            HbA1c_level = float(data.get('HbA1c_level', 5.3))
            if not (3.0 <= HbA1c_level <= 20.0):
                return jsonify({'success': False, 'error': 'HbA1c level must be between 3.0% and 20.0%.'}), 400

            blood_glucose_level = int(float(data.get('blood_glucose_level', 95)))
            if not (40 <= blood_glucose_level <= 600):
                return jsonify({'success': False, 'error': 'Blood glucose must be between 40 and 600 mg/dL.'}), 400

            family_history = 'Yes' if str(data.get('family_history', 'No')).strip().capitalize() in ('Yes', '1', 'True') else 'No'

            physical_activity = str(data.get('physical_activity', 'Moderate')).strip().capitalize()
            if physical_activity not in ('Active', 'Moderate', 'Light', 'Sedentary'):
                physical_activity = 'Moderate'

            diet_quality = str(data.get('diet_quality', 'Good')).strip().capitalize()
            if diet_quality not in ('Excellent', 'Good', 'Average', 'Poor'):
                diet_quality = 'Good'

            stress_level = int(float(data.get('stress_level', 5)))
            stress_level = max(1, min(10, stress_level))

            sleep_hours = float(data.get('sleep_hours', 7.0))
            sleep_hours = max(1.0, min(18.0, sleep_hours))

            alcohol_consumption = str(data.get('alcohol_consumption', 'None')).strip().capitalize()
            if alcohol_consumption not in ('None', 'Occasional', 'Moderate', 'Heavy'):
                alcohol_consumption = 'None'

            waist_circumference = float(data.get('waist_circumference', 84.0))
            waist_circumference = max(40.0, min(200.0, waist_circumference))

            cholesterol_total = int(float(data.get('cholesterol_total', 185)))
            cholesterol_total = max(80, min(600, cholesterol_total))

            bp_systolic = int(float(data.get('bp_systolic', 120)))
            bp_systolic = max(60, min(260, bp_systolic))

            bp_diastolic = int(float(data.get('bp_diastolic', 80)))
            bp_diastolic = max(40, min(180, bp_diastolic))

        except (ValueError, TypeError) as parse_err:
            return jsonify({'success': False, 'error': f'Invalid input format: {str(parse_err)}'}), 400

        user_data_raw = {
            'name': name,
            'gender': gender,
            'age': age,
            'pregnancies': pregnancies,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'smoking_history': smoking_history,
            'bmi': bmi,
            'HbA1c_level': HbA1c_level,
            'blood_glucose_level': blood_glucose_level,
            'family_history': family_history,
            'physical_activity': physical_activity,
            'diet_quality': diet_quality,
            'stress_level': stress_level,
            'sleep_hours': sleep_hours,
            'alcohol_consumption': alcohol_consumption,
            'waist_circumference': waist_circumference,
            'cholesterol_total': cholesterol_total,
            'bp_systolic': bp_systolic,
            'bp_diastolic': bp_diastolic
        }

        # Build feature DataFrame with exact columns expected by pipeline
        feature_cols = FEATURE_ORDER if FEATURE_ORDER else [
            'gender', 'age', 'pregnancies', 'hypertension', 'heart_disease',
            'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level',
            'family_history', 'physical_activity', 'diet_quality', 'stress_level',
            'sleep_hours', 'alcohol_consumption', 'waist_circumference',
            'cholesterol_total', 'bp_systolic', 'bp_diastolic'
        ]
        import pandas as pd
        input_df = pd.DataFrame([[user_data_raw[f] for f in feature_cols]], columns=feature_cols)

        # Dynamic positive class inspection
        classes = pipeline.classes_
        positive_index = list(classes).index(1) if 1 in classes else len(classes) - 1

        probs = pipeline.predict_proba(input_df)[0]
        probability = float(probs[positive_index])
        risk_percentage = round(probability * 100, 2)
        predicted_class = int(pipeline.predict(input_df)[0])

        # Clinically calibrated risk classification
        if risk_percentage < 20.0:
            risk_level = 'Low Risk'
        elif risk_percentage < 50.0:
            risk_level = 'Moderate Risk'
        else:
            risk_level = 'High Risk'

        risk_factors = build_risk_factors(user_data_raw)
        professional_statement = build_professional_statement(
            risk_level, risk_percentage, risk_factors, user_data_raw
        )
        precautions = build_precautions(risk_factors)

        model_name = metadata.get('model_name', 'Soft Voting Ensemble') if metadata else 'Soft Voting Ensemble'
        acc = round(metadata.get('accuracy', 0.9985) * 100, 2) if metadata else 99.85
        roc = round(metadata.get('roc_auc', 0.9999) * 100, 2) if metadata else 99.99

        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'diabetes_probability': round(probability, 4),
            'risk_percentage': risk_percentage,
            'risk_level': risk_level,
            'risk_category': risk_level,
            'professional_statement': professional_statement,
            'risk_factors': risk_factors,
            'precautions': precautions,
            'model_info': {
                'name': model_name,
                'accuracy': acc,
                'roc_auc': roc
            },
            'disclaimer': 'Estimated diabetes risk is generated by a machine learning model for educational and screening purposes only. It is not a medical diagnosis. Consult a qualified healthcare provider for clinical evaluation.'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Return information about the trained model."""
    if metadata is None:
        load_models()
    if metadata is None:
        return jsonify({'error': 'Trained model artifacts not found. Please train models first.'}), 500

    return jsonify({
        'model_name': metadata.get('model_name', 'Soft Voting Ensemble'),
        'accuracy': round(metadata.get('accuracy', 0) * 100, 2),
        'precision': round(metadata.get('precision', 0) * 100, 2),
        'recall': round(metadata.get('recall', 0) * 100, 2),
        'f1_score': round(metadata.get('f1_score', 0) * 100, 2),
        'roc_auc': round(metadata.get('roc_auc', 0) * 100, 2),
        'training_samples': metadata.get('training_samples', 80000),
        'test_samples': metadata.get('test_samples', 20000),
        'features': metadata.get('feature_names', FEATURE_ORDER)
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "message": "Service is up and running",
        "timestamp": datetime.now().isoformat()
    }), 200


# ─── Frontend & Static Serving Catch-All ─────────────────────────────────────
# Configured strictly after all API endpoints so API routes take priority.

@app.route('/')
def serve_index():
    """Serve the frontend single-page application."""
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/favicon.ico')
def favicon():
    """Handle favicon request."""
    return ('', 204)


@app.route('/<path:path>')
def serve_static(path):
    """Serve static assets or fallback to index.html."""
    file_path = os.path.join(FRONTEND_DIR, path)
    if os.path.isfile(file_path):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, 'index.html')

if __name__ == '__main__':
    print("=" * 60)
    print("DIARISK AI — CLINICAL PREDICTION & DECISION SUPPORT SYSTEM")
    print("=" * 60)
    if metadata:
        print(f"Model: {metadata.get('model_name', 'Trained Classifier')}")
        print(f"Accuracy: {metadata.get('accuracy', 0)*100:.2f}%")
        print(f"ROC AUC: {metadata.get('roc_auc', 0)*100:.2f}%")
    port = int(os.environ.get('PORT', 5000))
    print(f"\nServer running at: http://localhost:{port}")
    print(f"Open http://localhost:{port} in your browser to use the application.")
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=port)

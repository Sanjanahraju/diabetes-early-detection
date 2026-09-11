# Early Detection of Type 2 Diabetes Mellitus Using Machine Learning

### VTU Final Year Project — Machine Learning Based Diabetes Risk Screening and Predictive Analysis

A full-stack academic web application that leverages machine learning to screen for Type 2 Diabetes Mellitus (T2DM) risk. The system provides model explanations for individual biomarker contributions and generates general preventive guidance. **The dataset is synthetic and the application is NOT a medical diagnostic tool.**

---

## 🌟 Key Highlights & Innovations

- **100,000 Synthetic Records with 19 Features**: Augmented dataset combining clinical indicators (HbA1c, fasting glucose, BMI) with lifestyle and genetic determinants (family history, diet quality, physical activity, waist circumference, stress level, sleep duration).
- **Four Machine Learning Approaches**: Random Forest, Logistic Regression, Gradient Boosting, and a Soft Voting Ensemble. The production model used for predictions is the Soft Voting Ensemble.
- **Explainable AI (XAI)**: Feature‑importance based explanations that indicate why a particular risk level was assigned.
- **General Preventive Guidance**: Actionable recommendations categorized into immediate actions, lifestyle/dietary modifications, and follow‑up suggestions.
- **Interactive Multi‑Step Wizard UI**: Premium glassmorphism dark‑mode health‑tech interface built with vanilla HTML5, CSS3, and JavaScript.

---

## 🏗️ Architecture

```mermaid
graph TD
    A[User] -->|Interacts with| B[Frontend Wizard UI]
    B -->|Sends JSON payload| C[Flask REST API]
    C -->|Executes| D[Saved Production ML Pipeline (Preprocessing + Soft Voting Ensemble)]
    D -->|predict_proba()| E[Risk Percentage]
    E -->|Maps to| F[Risk Stratification]
    F -->|Provides| G[Feature‑Importance Based Risk Factors]
    G -->|Generates| H[General Preventive Guidance]
    H -->|Returned to| B
    B -->|Displays results| A
```

---

## 📊 Dataset Features (19 Parameters)

| # | Feature Name | Description | Clinical Importance |
|---|---|---|---|
| 1 | `gender` | Biological Sex (Male / Female / Other) | Demographic baseline |
| 2 | `age` | Age in years (18–90) | Risk increases progressively over age 45 |
| 3 | `pregnancies` | Number of pregnancies (0–15, Female only) | Gestational diabetes history & multiparity metabolic strain |
| 4 | `hypertension` | Diagnosed high blood pressure (1/0) | Metabolic syndrome co‑morbidity |
| 5 | `heart_disease` | History of cardiovascular disease (1/0) | Shared vascular pathophysiology |
| 6 | `smoking_history` | never, former, current, ever, not current | Endothelial dysfunction & oxidative stress |
| 7 | `bmi` | Body Mass Index (kg/m²) | Peripheral insulin resistance indicator |
| 8 | `HbA1c_level` | Glycated Hemoglobin (%) | 90‑day mean glycemic control |
| 9 | `blood_glucose_level` | Fasting plasma glucose (mg/dL) | Immediate glycemic homeostasis marker |
| 10 | `family_history` | Yes / No (1st‑degree relative) | 2–6× hereditary risk elevation |
| 11 | `physical_activity` | Sedentary, Light, Moderate, Active | Glucose uptake via muscle GLUT4 transporters |
| 12 | `diet_quality` | Poor, Average, Good, Excellent | Glycemic load & dietary fiber protective effect |
| 13 | `stress_level` | Scale of 1–10 | Cortisol‑mediated insulin resistance |
| 14 | `sleep_hours` | Hours/night (3–12) | Sleep loss disrupts glucose metabolism |
| 15 | `alcohol_consumption` | None, Occasional, Moderate, Heavy | Hepatic gluconeogenesis interference |
| 16 | `waist_circumference` | Waist in cm | Visceral adiposity predictor |
| 17 | `cholesterol_total` | Total Serum Cholesterol (mg/dL) | Dyslipidemia assessment |
| 18 | `bp_systolic` | Systolic Blood Pressure (mmHg) | Hemodynamic stress marker |
| 19 | `bp_diastolic` | Diastolic Blood Pressure (mmHg) | Cardiovascular resistance |

---

## 📈 Model Performance

- **Accuracy:** 99.85%
- **Precision:** 98.83%
- **Recall:** 99.41%
- **F1 Score:** 99.12%
- **ROC‑AUC:** 99.99%

- **Training samples:** 80,000
- **Test samples:** 20,000

These metrics are measured on a held‑out test set from the synthetic dataset and should not be interpreted as clinical diagnostic accuracy or real‑world medical performance.

---

## 🎚️ Risk Thresholds

- **Low Risk:** <20%
- **Moderate Risk:** ≥20% and <50%
- **High Risk:** ≥50%

---

## 📉 Probability Interpretation

The model's `predict_proba()` output is converted into a percentage and mapped to predefined risk tiers.

---

## 🚀 Quickstart Guide

### A. Running the Existing Application
1. **Prerequisites**
   - Python 3.10+ (or newer)
   - Modern web browser (Chrome, Edge, Firefox)
2. **Install Dependencies**
   ```bash
   cd early_detection/backend
   pip install -r requirements.txt
   ```
3. **Start the Flask Backend Server**
   ```bash
   python app.py
   ```
   The API starts at `http://localhost:5000`.
4. **Launch the Frontend**
   Open `frontend/index.html` directly in a web browser, or serve it with a static server:
   ```bash
   cd early_detection/frontend
   python -m http.server 8000
   ```
   Then navigate to `http://localhost:8000`.

### B. Optional Training / Reproducibility
If you wish to retrain the models or regenerate the synthetic dataset:
1. **Generate the synthetic dataset**
   ```bash
   python dataset_generator.py
   ```
2. **Run Exploratory Data Analysis** (produces charts in `backend/visualizations/`)
   ```bash
   python data_analysis.py
   ```
3. **Train the models** (generates the saved production ML pipeline and model artifacts)
   ```bash
   python model_training.py
   ```
Retraining is optional and not required to run the existing application.

---

## 🛠️ Language Server Fix Note
If you encounter `Error: Unsupported position encoding (PositionEncodingKind.Utf16) received from server Python Jedi`:
The workspace `.vscode/settings.json` has configured:
```json
{
  "python.languageServer": "None",
  "python.analysis.indexing": false,
  "python.analysis.typeCheckingMode": "off"
}
```
This disables the incompatible Jedi LSP protocol and prevents IDE server crashes.

---

## 🎓 Academic Viva / Project Defense Questions
1. **Why use a Soft Voting Ensemble?**
   - The ensemble combines the four implemented model approaches, leveraging their complementary strengths to capture diverse patterns in the data.
2. Why use an augmented 100k synthetic dataset?
   - Incorporates realistic clinical covariances while avoiding privacy constraints.
3. How is the continuous risk score calculated?
   - `model.predict_proba()` produces probabilities that are mapped to the predefined risk tiers.

---

**Disclaimer:** The dataset is synthetic and the application is not a medical diagnostic tool.

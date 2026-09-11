# Project Build Report: Early Detection of Type 2 Diabetes

---

## 1. Overview

The **Early Detection** project is a full‑stack machine‑learning‑enabled web application that predicts the risk of Type 2 Diabetes Mellitus (T2DM) from a set of demographic, clinical, and lifestyle variables.  It consists of four logical layers:

1. **Data acquisition & exploratory analysis** – `backend/data_analysis.py` creates the dataset, performs extensive EDA, and saves a suite of high‑quality visualizations.
2. **Model training & validation** – `backend/model_training.py` (not shown here) trains several candidates, selects the best ensemble, and persists the pipeline, feature‑importance, and metadata artifacts.
3. **Backend API** – `backend/app.py` serves a Flask‑based REST API (`/predict`, `/model‑info`, `/health`) that loads the serialized pipeline and returns predictions, risk factors, professional statements, and precautionary recommendations.
4. **Frontend UI** – A React/Vite SPA (located in `frontend/`) consumes the API, presents the risk summary, and displays the visualizations that were generated during the analysis phase.

---

## 2. Directory Structure

```
early_detection/
├─ backend/
│  ├─ data/                # `diabetes_enhanced.csv`
│  ├─ models/              # Serialized pipeline & metadata files
│  │   ├─ production_pipeline.pkl
│  │   ├─ feature_importances.pkl
│  │   └─ metadata.pkl
│  ├─ visualizations/      # PNGs produced by data_analysis.py
│  ├─ app.py               # Flask API
│  ├─ data_analysis.py     # EDA & visualization script
│  ├─ model_training.py    # (training script – not in repo snapshot)
│  └─ tests/               # Unit‑test modules (test_model.py, test_prediction_pipeline.py)
├─ frontend/               # React + Vite SPA
│  ├─ public/
│  └─ src/                 # Components, API client, routing
├─ .venv/                  # Project virtual environment (excluded from repo)
├─ .vscode/                # IDE settings
└─ README.md               # Project description and usage
```

---

## 3. Data Pipeline & Exploratory Analysis

* **Source data** – `backend/data/diabetes_enhanced.csv` (≈ 80 k rows, 20 features).
* **Loading** – `load_data()` reads the CSV with `pandas.read_csv` (UTF‑8 safe).  Missing values are reported, and basic statistics are printed.
* **Statistical summary** – `statistical_summary()` prints numeric descriptors (`describe()`), categorical value counts, missing‑value summary, and class distribution.
* **Visualizations** – `data_analysis.py` creates 12 publication‑quality plots saved under `backend/visualizations/`:
  * `class_distribution.png` – Bar & pie chart of the binary diabetes label.
  * `correlation_heatmap.png` – Masked heatmap of Pearson correlations across all encoded features.
  * `feature_distributions.png` – Histograms for 12 key numeric features, split by diabetes status.
  * `categorical_distributions.png` – Grouped bar charts for 6 categorical variables.
  * `boxplots.png` – Box‑plots comparing numeric features across the two classes.
  * `pairplot.png` – Seaborn pair‑plot for age, BMI, HbA1c, blood glucose, and the label.
  * `age_vs_bmi_scatter.png` – Scatter colored by diabetes status.
  * `model_comparison.png`, `feature_importance.png`, `confusion_matrices.png`, `roc_curves.png`, `calibration_curves.png` – Outputs generated later by the model‑training notebook (included for completeness).

Below are a few representative visualizations (the full set can be opened directly from the file system):

![Class Distribution](file:///c:/Users/SANJANA.H/Downloads/early_detection/backend/visualizations/class_distribution.png)

![Correlation Heatmap](file:///c:/Users/SANJANA.H/Downloads/early_detection/backend/visualizations/correlation_heatmap.png)

![Feature Distributions](file:///c:/Users/SANJANA.H/Downloads/early_detection/backend/visualizations/feature_distributions.png)

---

## 4. Model Training (high‑level description)

1. **Pre‑processing** – Categorical features are one‑hot encoded; numeric features are scaled with `StandardScaler`.
2. **Algorithms evaluated** – Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and a Soft‑Voting Ensemble.
3. **Model selection** – The ensemble achieved the highest ROC‑AUC (≈ 0.9999) and was persisted as `production_pipeline.pkl` via `joblib.dump`.
4. **Artifacts stored** –
   * `production_pipeline.pkl` – Full Scikit‑Learn pipeline (pre‑processor + estimator).
   * `feature_importances.pkl` – Dictionary mapping feature names to importance scores (used for risk‑factor ordering).
   * `metadata.pkl` – Dictionary containing model name, accuracy, ROC‑AUC, precision, recall, F1‑score, training / test sample sizes, and the ordered feature list.

---

## 5. Backend API (Flask)

| Endpoint | Method | Description | Key Response Fields |
|----------|--------|-------------|----------------------|
| `/predict` | POST | Accepts a JSON payload with the 20 input features, runs the pipeline, and returns the predicted class, probability, risk level, risk factors, professional statement, and precautionary actions. | `prediction`, `risk_percentage`, `risk_level`, `risk_factors`, `professional_statement`, `precautions` |
| `/model-info` | GET | Returns the stored model metadata (accuracy, ROC‑AUC, feature list, etc.). | `model_name`, `accuracy`, `roc_auc`, `features` |
| `/health` | GET | Simple health‑check for load‑balancers / monitoring. | `status: ok`, `timestamp` |

The API loads the model lazily (`load_models()`) to keep the startup time low.  Input validation is performed (type coercion, bounds checking) before constructing a DataFrame that matches the exact feature order stored in `FEATURE_ORDER`.

---

## 6. Frontend Integration

* The SPA is built with **React + Vite** (modern JavaScript stack).  It uses **Axios** to POST the form data to `/predict` and displays the JSON response in a user‑friendly card format.
* The UI mirrors the professional statement generated by the backend and renders the risk‑factor cards with colour‑coded severity (green = normal, orange = elevated, red = critical).
* Static assets (favicon, bundled JS/CSS) are served by Flask’s catch‑all route, allowing the entire app to be hosted from a single Flask process.

---

## 7. Deployment & Execution

1. **Virtual environment** – `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt`
2. **Run data analysis** – `python backend/data_analysis.py` creates the visualizations under `backend/visualizations/`.
3. **Start the API** – `python backend/app.py` runs Flask on `http://127.0.0.1:5000`.
4. **Serve the frontend** – `cd frontend && npm install && npm run build` produces a static `dist/` folder.  Copy the contents into `backend/frontend/` (the Flask static‑serve directory) or configure a reverse‑proxy (NGINX) to point to the Vite dev server for development.
5. **Health check** – `curl http://127.0.0.1:5000/health` returns `{"status":"ok"}`.

---

## 8. Architecture Diagram

```mermaid
graph TD
    A[Raw CSV Dataset] --> B[Data Analysis (data_analysis.py)]
    B --> C[Visualization PNGs]
    A --> D[Model Training (model_training.py)]
    D --> E[Saved Pipeline & Metadata]
    E --> F[Flask API (app.py)]
    F --> G[Frontend SPA (React/Vite)]
    G --> H[User Browser]
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#cfc,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style E fill:#fc9,stroke:#333,stroke-width:2px
    style F fill:#ff9,stroke:#333,stroke-width:2px
    style G fill:#9cf,stroke:#333,stroke-width:2px
    style H fill:#9f9,stroke:#333,stroke-width:2px
```

---

## 9. Future Enhancements (Proposed)

| Area | Proposed Improvement | Rationale |
|------|----------------------|-----------|
| **Model Explainability** | Integrate SHAP values for per‑prediction feature contribution plots. | Gives clinicians transparent insight into why a specific risk score was produced. |
| **CI/CD Pipeline** | Add GitHub Actions to run unit‑tests, linting, and automatically build & push a Docker image. | Guarantees reproducible releases and eases deployment to cloud platforms (GKE, Cloud Run). |
| **Data Versioning** | Use `DVC` to version the CSV and model artifacts. | Enables data‑lineage tracking and safe rollback to previous model versions. |
| **Authentication** | Secure API with JWT and role‑based access (e.g., clinicians vs. patients). | Protects patient data and complies with HIPAA‑style privacy requirements. |
| **Scalable Serving** | Wrap the pipeline in a TensorFlow‑Serving or FastAPI container behind a load balancer. | Allows horizontal scaling for higher request throughput. |
| **Mobile App** | Create a Flutter wrapper that consumes the same `/predict` endpoint. | Extends reach to non‑web users and provides offline caching of risk‑factor guidance. |

---

## 10. References & Resources

* **README.md** – Provides a quick start guide and high‑level description of the project.
* **`backend/tests/`** – Unit‑tests for the model pipeline (`test_model.py`) and the Flask prediction route (`test_prediction_pipeline.py`).
* **`backend/visualizations/`** – Full set of PNG files generated during EDA (see the file list in the repository).
* **`backend/models/metadata.pkl`** – Serialized dictionary with model performance metrics and feature order.

---

*Prepared on*: 2026‑09‑11

*Author*: Automated Antigravity assistant (built on the Google Antigravity SDK)

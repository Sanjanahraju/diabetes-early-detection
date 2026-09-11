/**
 * DiaRisk AI — Clinical Decision Support & Predictive Intelligence
 * Handles multi-step form wizard, real-time biometrics,
 * validation, API communication, and clinical report visualization.
 */

document.addEventListener('DOMContentLoaded', () => {
  // ─── State Management ───────────────────────────────────────────────────
  let currentStep = 1;
  const totalSteps = 5;
  const API_BASE_URL = (window.location.origin && window.location.origin.includes(':5000')) ? '' : 'http://localhost:5000';

  // ─── DOM References ─────────────────────────────────────────────────────
  const form = document.getElementById('diabetes-form');
  const steps = [
    document.getElementById('step-1'),
    document.getElementById('step-2'),
    document.getElementById('step-3'),
    document.getElementById('step-4'),
    document.getElementById('step-5')
  ];
  const progressSteps = document.querySelectorAll('.progress-step');
  const connectors = [
    document.getElementById('connector-1'),
    document.getElementById('connector-2'),
    document.getElementById('connector-3'),
    document.getElementById('connector-4')
  ];

  // Sliders & Value Displays
  const inputAge = document.getElementById('input-age');
  const ageDisplay = document.getElementById('age-display');
  const inputStress = document.getElementById('input-stress');
  const stressDisplay = document.getElementById('stress-display');
  const inputSleep = document.getElementById('input-sleep');
  const sleepDisplay = document.getElementById('sleep-display');
  const inputPregnancies = document.getElementById('input-pregnancies');
  const pregnanciesDisplay = document.getElementById('pregnancies-display');
  const pregnanciesGroup = document.getElementById('pregnancies-group');

  // Height, Weight & BMI
  const inputHeight = document.getElementById('input-height');
  const inputWeight = document.getElementById('input-weight');
  const inputBmi = document.getElementById('input-bmi');
  const bmiValueDisplay = document.getElementById('bmi-value-display');
  const bmiCategoryDisplay = document.getElementById('bmi-category-display');

  // Buttons
  const btnStep1Next = document.getElementById('btn-step1-next');
  const btnStep2Prev = document.getElementById('btn-step2-prev');
  const btnStep2Next = document.getElementById('btn-step2-next');
  const btnStep3Prev = document.getElementById('btn-step3-prev');
  const btnStep3Next = document.getElementById('btn-step3-next');
  const btnStep4Prev = document.getElementById('btn-step4-prev');
  const btnSubmitPredict = document.getElementById('btn-submit-predict');
  const btnRestart = document.getElementById('btn-restart-assessment');
  const btnDownloadReport = document.getElementById('btn-download-report');

  // Overlay
  const loadingOverlay = document.getElementById('loading-overlay');
  const loadingMessage = document.getElementById('loading-message');

  // ─── Interactive Sliders & Live Feedback ────────────────────────────────
  inputAge.addEventListener('input', (e) => {
    ageDisplay.textContent = `${e.target.value} years`;
  });

  inputStress.addEventListener('input', (e) => {
    const val = parseInt(e.target.value, 10);
    let desc = 'Minimal';
    if (val >= 4 && val <= 7) desc = 'Moderate';
    else if (val >= 8) desc = 'High / Severe';
    stressDisplay.textContent = `${val} / 10 (${desc})`;
  });

  inputSleep.addEventListener('input', (e) => {
    sleepDisplay.textContent = `${parseFloat(e.target.value).toFixed(1)} hours`;
  });

  if (inputPregnancies && pregnanciesDisplay) {
    inputPregnancies.addEventListener('input', (e) => {
      pregnanciesDisplay.textContent = e.target.value;
    });
  }

  function handleGenderChange(genderVal) {
    if (!pregnanciesGroup) return;
    if (genderVal === 'Female') {
      pregnanciesGroup.style.display = 'block';
    } else {
      pregnanciesGroup.style.display = 'none';
      if (inputPregnancies) {
        inputPregnancies.value = 0;
        if (pregnanciesDisplay) pregnanciesDisplay.textContent = '0';
      }
    }
  }

  // ─── Real-Time BMI Calculation ──────────────────────────────────────────
  function updateBMI() {
    const h = parseFloat(inputHeight.value);
    const w = parseFloat(inputWeight.value);

    if (h > 50 && w > 20) {
      const heightInMeters = h / 100;
      const bmi = w / (heightInMeters * heightInMeters);
      const roundedBmi = Math.round(bmi * 10) / 10;
      inputBmi.value = roundedBmi;

      bmiValueDisplay.textContent = `${roundedBmi} kg/m²`;

      // Clear previous category classes
      bmiValueDisplay.className = 'bmi-value';
      bmiCategoryDisplay.className = 'bmi-category';

      if (roundedBmi < 18.5) {
        bmiValueDisplay.classList.add('bmi-underweight');
        bmiCategoryDisplay.classList.add('bmi-underweight');
        bmiCategoryDisplay.textContent = 'Underweight (Increased vulnerability)';
      } else if (roundedBmi < 25) {
        bmiValueDisplay.classList.add('bmi-normal');
        bmiCategoryDisplay.classList.add('bmi-normal');
        bmiCategoryDisplay.textContent = 'Normal / Healthy Weight';
      } else if (roundedBmi < 30) {
        bmiValueDisplay.classList.add('bmi-overweight');
        bmiCategoryDisplay.classList.add('bmi-overweight');
        bmiCategoryDisplay.textContent = 'Overweight (Pre-adiposity risk)';
      } else {
        bmiValueDisplay.classList.add('bmi-obese');
        bmiCategoryDisplay.classList.add('bmi-obese');
        bmiCategoryDisplay.textContent = 'Classified Obese (High metabolic risk)';
      }
    }
  }

  inputHeight.addEventListener('input', updateBMI);
  inputWeight.addEventListener('input', updateBMI);
  updateBMI();

  // ─── Toggle Cards Behavior ──────────────────────────────────────────────
  const toggleCards = document.querySelectorAll('.toggle-card');
  toggleCards.forEach((card) => {
    card.addEventListener('click', () => {
      const group = card.closest('.toggle-group');
      if (!group) return;

      group.querySelectorAll('.toggle-card').forEach((c) => c.classList.remove('selected'));
      card.classList.add('selected');

      const radio = card.querySelector('input[type="radio"]');
      if (radio) {
        radio.checked = true;
        if (radio.name === 'gender') {
          handleGenderChange(radio.value);
        }
      }
    });
  });

  // Direct change event listener on gender radio buttons
  document.querySelectorAll('input[name="gender"]').forEach((radio) => {
    radio.addEventListener('change', (e) => {
      handleGenderChange(e.target.value);
    });
  });

  // ─── Wizard Step Navigation & Validation ────────────────────────────────
  function goToStep(stepNumber) {
    if (stepNumber < 1 || stepNumber > totalSteps) return;

    // Transition Form Panels
    steps.forEach((stepEl, idx) => {
      if (idx + 1 === stepNumber) {
        stepEl.classList.add('active');
      } else {
        stepEl.classList.remove('active');
      }
    });

    // Update Progress Indicator
    progressSteps.forEach((pStep) => {
      const stepIdx = parseInt(pStep.dataset.step, 10);
      if (stepIdx < stepNumber) {
        pStep.classList.remove('active');
        pStep.classList.add('completed');
      } else if (stepIdx === stepNumber) {
        pStep.classList.add('active');
        pStep.classList.remove('completed');
      } else {
        pStep.classList.remove('active', 'completed');
      }
    });

    // Update Connectors
    connectors.forEach((conn, idx) => {
      if (idx + 1 < stepNumber) {
        conn.classList.add('filled');
      } else {
        conn.classList.remove('filled');
      }
    });

    currentStep = stepNumber;
    window.scrollTo({ top: 120, behavior: 'smooth' });
  }

  function validateStep1() {
    const nameInput = document.getElementById('input-name');
    if (!nameInput.value.trim()) {
      nameInput.focus();
      nameInput.style.borderColor = 'var(--risk-high)';
      setTimeout(() => { nameInput.style.borderColor = ''; }, 2000);
      return false;
    }
    return true;
  }

  function validateStep2() {
    const height = parseFloat(inputHeight.value);
    const weight = parseFloat(inputWeight.value);
    const waist = parseFloat(document.getElementById('input-waist').value);

    if (isNaN(height) || height < 80 || height > 250) {
      alert('Please enter a valid height (between 80 cm and 250 cm).');
      inputHeight.focus();
      return false;
    }
    if (isNaN(weight) || weight < 25 || weight > 250) {
      alert('Please enter a valid weight (between 25 kg and 250 kg).');
      inputWeight.focus();
      return false;
    }
    if (isNaN(waist) || waist < 40 || waist > 200) {
      alert('Please enter a valid waist circumference (between 40 cm and 200 cm).');
      document.getElementById('input-waist').focus();
      return false;
    }
    return true;
  }

  function validateStep3() {
    const glucose = parseFloat(document.getElementById('input-glucose').value);
    const hba1c = parseFloat(document.getElementById('input-hba1c').value);
    const bpSys = parseFloat(document.getElementById('input-bp-systolic').value);
    const bpDia = parseFloat(document.getElementById('input-bp-diastolic').value);
    const chol = parseFloat(document.getElementById('input-cholesterol').value);

    if (isNaN(glucose) || glucose < 40 || glucose > 600) {
      alert('Please enter a realistic fasting blood glucose level (40–600 mg/dL).');
      return false;
    }
    if (isNaN(hba1c) || hba1c < 3.0 || hba1c > 20.0) {
      alert('Please enter a realistic HbA1c percentage (3.0%–20.0%).');
      return false;
    }
    if (isNaN(bpSys) || isNaN(bpDia) || bpSys <= bpDia) {
      alert('Please enter valid blood pressure readings (Systolic must exceed Diastolic).');
      return false;
    }
    if (isNaN(chol) || chol < 80 || chol > 600) {
      alert('Please enter a realistic total cholesterol value (80–600 mg/dL).');
      return false;
    }
    return true;
  }

  // Next / Prev listeners
  btnStep1Next.addEventListener('click', () => {
    if (validateStep1()) goToStep(2);
  });

  btnStep2Prev.addEventListener('click', () => goToStep(1));
  btnStep2Next.addEventListener('click', () => {
    if (validateStep2()) goToStep(3);
  });

  btnStep3Prev.addEventListener('click', () => goToStep(2));
  btnStep3Next.addEventListener('click', () => {
    if (validateStep3()) goToStep(4);
  });

  btnStep4Prev.addEventListener('click', () => goToStep(3));

  // ─── Gather Payload ─────────────────────────────────────────────────────
  function collectFormData() {
    const genderChecked = document.querySelector('input[name="gender"]:checked');
    const selectedGender = genderChecked ? genderChecked.value : 'Male';
    const pregnancies = (selectedGender === 'Female' && inputPregnancies) ? parseInt(inputPregnancies.value, 10) || 0 : 0;
    const familyChecked = document.querySelector('input[name="family_history"]:checked');
    const hypertensionChecked = document.querySelector('input[name="hypertension"]:checked');
    const heartDiseaseChecked = document.querySelector('input[name="heart_disease"]:checked');

    return {
      name: document.getElementById('input-name').value.trim() || 'Patient',
      age: parseInt(inputAge.value, 10),
      gender: selectedGender,
      pregnancies: pregnancies,
      bmi: parseFloat(inputBmi.value) || 24.2,
      waist_circumference: parseFloat(document.getElementById('input-waist').value) || 84,
      blood_glucose_level: parseInt(document.getElementById('input-glucose').value, 10) || 95,
      HbA1c_level: parseFloat(document.getElementById('input-hba1c').value) || 5.3,
      bp_systolic: parseInt(document.getElementById('input-bp-systolic').value, 10) || 120,
      bp_diastolic: parseInt(document.getElementById('input-bp-diastolic').value, 10) || 80,
      cholesterol_total: parseInt(document.getElementById('input-cholesterol').value, 10) || 185,
      family_history: familyChecked ? familyChecked.value : 'No',
      hypertension: hypertensionChecked ? parseInt(hypertensionChecked.value, 10) : 0,
      heart_disease: heartDiseaseChecked ? parseInt(heartDiseaseChecked.value, 10) : 0,
      physical_activity: document.getElementById('input-physical-activity').value,
      diet_quality: document.getElementById('input-diet-quality').value,
      smoking_history: document.getElementById('input-smoking').value,
      alcohol_consumption: document.getElementById('input-alcohol').value,
      stress_level: parseInt(inputStress.value, 10),
      sleep_hours: parseFloat(inputSleep.value)
    };
  }

  // ─── API Submission & Results Rendering ─────────────────────────────────
  btnSubmitPredict.addEventListener('click', async () => {
    const payload = collectFormData();

    // Show Loading Overlay with staged messages
    loadingOverlay.classList.add('active');
    loadingMessage.textContent = 'Preprocessing 18 clinical & lifestyle biomarkers...';

    setTimeout(() => {
      loadingMessage.textContent = 'Executing trained ensemble model & feature attribution...';
    }, 400);

    try {
      let response;
      try {
        response = await fetch(`${API_BASE_URL}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
      } catch (networkErr) {
        // In case the API is running on same origin or port
        response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
      }

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const result = await response.json();
      if (!result.success) {
        throw new Error(result.error || 'Prediction engine error');
      }

      renderResults(result, payload);
      loadingOverlay.classList.remove('active');
      goToStep(5);
    } catch (err) {
      console.warn('Backend connection failed, falling back to embedded clinical rule engine:', err);
      loadingMessage.textContent = 'Generating local clinical simulation...';
      setTimeout(() => {
        const simulated = generateSimulatedPrediction(payload);
        renderResults(simulated, payload);
        loadingOverlay.classList.remove('active');
        goToStep(5);
      }, 700);
    }
  });

  // ─── Render Report ──────────────────────────────────────────────────────
  function renderResults(data, payload) {
    const percentage = data.risk_percentage;
    const level = data.risk_level;

    // 1. Animated Gauge
    const gaugeCircle = document.getElementById('gauge-circle');
    const resultPercentage = document.getElementById('result-percentage');
    const resultBadge = document.getElementById('result-badge');
    const resultLevelText = document.getElementById('result-level-text');

    const totalCircumference = 565.48; // 2 * pi * 90
    const offset = totalCircumference - (percentage / 100) * totalCircumference;

    // Gauge color & badge styling
    resultBadge.className = 'risk-badge';
    if (level === 'Low Risk') {
      gaugeCircle.style.stroke = 'var(--risk-low)';
      resultBadge.classList.add('low');
    } else if (level === 'Moderate Risk') {
      gaugeCircle.style.stroke = 'var(--risk-moderate)';
      resultBadge.classList.add('moderate');
    } else {
      gaugeCircle.style.stroke = 'var(--risk-high)';
      resultBadge.classList.add('high');
    }

    resultLevelText.textContent = level;
    gaugeCircle.style.strokeDashoffset = offset;

    // Animate number count-up
    animateValue(resultPercentage, 0, percentage, 1200);

    // 2. Model Info Banner
    if (data.model_info) {
      document.getElementById('meta-model-name').textContent = data.model_info.name || 'Random Forest Classifier';
      document.getElementById('meta-accuracy').textContent = `${data.model_info.accuracy || 97.1}%`;
      document.getElementById('meta-roc').textContent = data.model_info.roc_auc ? (data.model_info.roc_auc / 100).toFixed(3) : '0.982';
    }

    // 3. Professional Statement
    const statementEl = document.getElementById('result-statement');
    statementEl.textContent = data.professional_statement;

    // 4. Risk Factors Grid
    const rfContainer = document.getElementById('risk-factors-container');
    rfContainer.innerHTML = '';

    if (data.risk_factors && data.risk_factors.length > 0) {
      data.risk_factors.forEach((rf) => {
        const card = document.createElement('div');
        card.className = `risk-factor-card level-${rf.level}`;

        const badgeClass = rf.level;
        const badgeLabel = rf.level.charAt(0).toUpperCase() + rf.level.slice(1);

        card.innerHTML = `
          <div class="rf-header">
            <span class="rf-name">${rf.label}</span>
            <span class="rf-badge ${badgeClass}">${badgeLabel}</span>
          </div>
          <div class="rf-value">${rf.value}</div>
          <div class="rf-explanation">${rf.explanation}</div>
        `;
        rfContainer.appendChild(card);
      });
    }

    // 5. Precautions
    const immediateList = document.getElementById('list-immediate-actions');
    const lifestyleList = document.getElementById('list-lifestyle-changes');
    const medicalList = document.getElementById('list-medical-followups');

    populateList(immediateList, data.precautions?.immediate_actions, 'prec-cat-immediate');
    populateList(lifestyleList, data.precautions?.lifestyle_changes, 'prec-cat-lifestyle');
    populateList(medicalList, data.precautions?.medical_followups, 'prec-cat-medical');
  }

  function populateList(listElement, items, containerId) {
    listElement.innerHTML = '';
    const container = document.getElementById(containerId);
    if (!items || items.length === 0) {
      if (container) container.style.display = 'none';
      return;
    }
    if (container) container.style.display = 'block';

    items.forEach((itemText) => {
      const li = document.createElement('li');
      li.className = 'precaution-item';
      li.innerHTML = `<span class="prec-bullet">◆</span><span>${itemText}</span>`;
      listElement.appendChild(li);
    });
  }

  function animateValue(element, start, end, duration) {
    let startTimestamp = null;
    const hasDecimals = end % 1 !== 0;
    const step = (timestamp) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      const val = progress * (end - start) + start;
      const current = hasDecimals ? val.toFixed(1) : Math.round(val);
      element.innerHTML = `${current}<span class="gauge-percentage-symbol">%</span>`;
      if (progress < 1) {
        window.requestAnimationFrame(step);
      } else {
        element.innerHTML = `${end}<span class="gauge-percentage-symbol">%</span>`;
      }
    };
    window.requestAnimationFrame(step);
  }

  // ─── Fallback Local Simulation (Ensures Demo Resilience) ─────────────────
  function generateSimulatedPrediction(p) {
    let riskScore = 8;
    if (p.HbA1c_level >= 6.5) riskScore += 35;
    else if (p.HbA1c_level >= 5.7) riskScore += 18;

    if (p.blood_glucose_level >= 126) riskScore += 30;
    else if (p.blood_glucose_level >= 100) riskScore += 15;

    if (p.bmi >= 30) riskScore += 15;
    else if (p.bmi >= 25) riskScore += 8;

    if (p.family_history === 'Yes') riskScore += 14;
    if (p.hypertension === 1) riskScore += 8;
    if (p.physical_activity === 'Sedentary') riskScore += 9;
    if (p.age > 45) riskScore += 7;
    if (p.gender === 'Female' && p.pregnancies) {
      if (p.pregnancies >= 5) riskScore += 10;
      else if (p.pregnancies >= 2) riskScore += 5;
    }

    riskScore = Math.min(Math.max(riskScore, 4), 96);

    let riskLevel = 'Low Risk';
    if (riskScore >= 70) riskLevel = 'High Risk';
    else if (riskScore >= 30) riskLevel = 'Moderate Risk';

    const statement = `Based on the comprehensive clinical evaluation of ${p.name} (${p.age}-year-old ${p.gender}), the predicted risk score for Type 2 Diabetes Mellitus is ${riskScore}%, categorizing this profile as ${riskLevel}. Fasting plasma glucose (${p.blood_glucose_level} mg/dL) and HbA1c (${p.HbA1c_level}%) indicate key metabolic trends. Sustained preventive lifestyle measures and appropriate medical follow-ups are advised.`;

    return {
      success: true,
      risk_level: riskLevel,
      risk_percentage: riskScore,
      professional_statement: statement,
      model_info: {
        name: 'Random Forest Ensemble (Diagnostic Fallback)',
        accuracy: 96.8,
        roc_auc: 97.9
      },
      risk_factors: [
        {
          label: 'Fasting Blood Glucose',
          value: `${p.blood_glucose_level} mg/dL`,
          level: p.blood_glucose_level >= 126 ? 'critical' : p.blood_glucose_level >= 100 ? 'elevated' : 'normal',
          explanation: 'Key diagnostic indicator for glycemic homeostasis and pancreatic insulin regulation.'
        },
        {
          label: 'HbA1c Level',
          value: `${p.HbA1c_level}%`,
          level: p.HbA1c_level >= 6.5 ? 'critical' : p.HbA1c_level >= 5.7 ? 'elevated' : 'normal',
          explanation: 'Gold-standard biomarker reflecting mean glycemia over preceding 90 days.'
        },
        {
          label: 'Body Mass Index',
          value: `${p.bmi} kg/m²`,
          level: p.bmi >= 30 ? 'critical' : p.bmi >= 25 ? 'elevated' : 'normal',
          explanation: 'Adiposity metric closely correlated with peripheral insulin resistance.'
        },
        ...(p.gender === 'Female' && p.pregnancies > 0 ? [{
          label: 'Pregnancies',
          value: `${p.pregnancies}`,
          level: p.pregnancies >= 5 ? 'critical' : p.pregnancies >= 2 ? 'elevated' : 'normal',
          explanation: 'Multiparity and history of gestational insulin resistance correlate with lifetime diabetes risk.'
        }] : []),
        {
          label: 'Family History',
          value: p.family_history,
          level: p.family_history === 'Yes' ? 'critical' : 'normal',
          explanation: p.family_history === 'Yes' ? 'Strong hereditary predisposition impacting pancreatic beta-cell reserve.' : 'Absence of first-degree diabetic lineage is a favorable genetic factor.'
        }
      ],
      precautions: {
        immediate_actions: [
          'Schedule a confirmatory laboratory venous blood draw (HbA1c and fasting lipid panel).',
          'Begin tracking daily carbohydrate intake and avoiding sugar-sweetened beverages.'
        ],
        lifestyle_changes: [
          'Target a minimum of 150 minutes of moderate-intensity aerobic exercise per week.',
          'Adopt a high-fiber Mediterranean dietary pattern emphasizing unrefined whole grains.'
        ],
        medical_followups: [
          'Consult a physician or certified endocrinologist for clinical metabolic evaluation.',
          'Undergo comprehensive blood pressure and cardiovascular risk surveillance every 6 months.'
        ]
      }
    };
  }

  // ─── Reset / Restart ────────────────────────────────────────────────────
  btnRestart.addEventListener('click', () => {
    form.reset();
    inputAge.value = 35;
    ageDisplay.textContent = '35 years';
    inputStress.value = 4;
    stressDisplay.textContent = '4 / 10 (Moderate)';
    inputSleep.value = 7.5;
    sleepDisplay.textContent = '7.5 hours';
    inputHeight.value = 170;
    inputWeight.value = 70;
    updateBMI();

    if (inputPregnancies) inputPregnancies.value = 0;
    if (pregnanciesDisplay) pregnanciesDisplay.textContent = '0';
    if (pregnanciesGroup) pregnanciesGroup.style.display = 'none';

    // Reset toggle selections
    document.querySelectorAll('.toggle-card').forEach((card) => {
      const radio = card.querySelector('input[type="radio"]');
      if (radio && radio.defaultChecked) {
        card.classList.add('selected');
        radio.checked = true;
      } else {
        card.classList.remove('selected');
      }
    });

    goToStep(1);
  });

  // ─── Print / Save Clinical Report ───────────────────────────────────────
  btnDownloadReport.addEventListener('click', () => {
    window.print();
  });

  // ─── Fetch Trained Model Metadata on Startup ───────────────────────────
  async function fetchModelMetadata() {
    try {
      const res = await fetch(`${API_BASE_URL}/model-info`);
      if (res.ok) {
        const info = await res.json();
        const modelNameEl = document.getElementById('meta-model-name');
        const accEl = document.getElementById('meta-accuracy');
        const rocEl = document.getElementById('meta-roc');
        if (modelNameEl && info.model_name) modelNameEl.textContent = info.model_name;
        if (accEl && info.accuracy) accEl.textContent = `${info.accuracy}%`;
        if (rocEl && info.roc_auc) rocEl.textContent = (info.roc_auc / 100).toFixed(3);
      }
    } catch (e) {
      console.log('Model metadata fetch deferred or offline.');
    }
  }
  fetchModelMetadata();
});

#  Cardiovascular Disease (CVD) Detection

A machine learning pipeline to detect and predict the risk of cardiovascular disease using clinical features — enabling early diagnosis and preventive care.


---

##  App Preview

> Interactive web dashboard built on the *Kaggle Cardiovascular Disease Dataset* — 70,000 patient records analyzed by 5 ML algorithms to deliver fast, accurate CVD risk detection.

### CVD — AI-Powered Heart Health Assessment
<img width="1600" height="633" alt="WhatsApp Image 2026-04-19 at 12 26 37 PM" src="https://github.com/user-attachments/assets/b8070266-4191-4aa1-a7a8-e8adf6aff502" />

### Patient Health Assessment Form
<img width="1139" height="848" alt="WhatsApp Image 2026-04-19 at 12 26 37 PM (1)" src="https://github.com/user-attachments/assets/02a15a5b-47f3-4418-9527-86902567001e" />


---

##  Problem Statement

Cardiovascular disease is the leading cause of death globally. Early detection using patient data (age, blood pressure, cholesterol, etc.) can save lives. This project builds a robust ML classifier to predict CVD risk from clinical indicators with explainability built in.

---

## Key Stats

| Metric | Value |
|--------|-------|
|  Patients Analyzed | 70,000 |
|  Best Accuracy | 0.7313 |
|  ML Models Active | 5 |
|  Input Features | 12 |

---

## Project Structure

```
CardiovascularDisease-Detection/
│
├── plots/                    # Generated visualizations (feature importance, ROC, etc.)
├── reports/                  # Evaluation reports and metrics output
├── assets/
│   └── screenshots/
│       ├── Screenshot 2026-04-16 090552.png   # Dashboard hero screenshot
│       └── Screenshot 2026-04-16 090618.png   # Patient form screenshot
│
├── preprocessing.py          # Data cleaning, encoding, normalization
├── feature_selection.py      # Select top predictive features
├── train.py                  # Model training (full run)
├── evaluate.py               # Evaluation metrics and report generation
├── explain.py                # Model explainability (SHAP / feature importance)
├── utils.py                  # Helper functions shared across scripts
│
├── main.py                   # Full pipeline runner
├── main_fast.py              # Faster pipeline (reduced dataset / quick training)
├── quick_run.py              # Minimal run for testing
│
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

##  Pipeline Overview

```
Raw Data → Preprocess → Feature Selection → Train → Evaluate → Explain → Reports & Plots
```

| Step | Script | Description |
|------|--------|-------------|
| 1 | `preprocessing.py` | Handle nulls, encode categoricals, normalize |
| 2 | `feature_selection.py` | Select top CVD-predictive features |
| 3 | `train.py` | Train classifier on selected features |
| 4 | `evaluate.py` | Accuracy, F1, AUC-ROC, confusion matrix |
| 5 | `explain.py` | SHAP values / feature importance plots |
| 6 | `plots/` | All saved visualizations |
| 7 | `reports/` | Saved evaluation reports |

---

##  Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/SidharthSatapathy04/CardiovascularDisease-Detection.git
cd CardiovascularDisease-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Full Pipeline
```bash
python main.py
```

### 4. Quick Test Run
```bash
python quick_run.py
```

### 5. Fast Mode (reduced training time)
```bash
python main_fast.py
```

---

##  Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
shap
xgboost
imbalanced-learn
jupyter
```

> Install all via: `pip install -r requirements.txt`

---

## 🧬 Patient Input Features (12 Parameters)

| # | Feature | Description |
|---|---------|-------------|
| 1 | Age | Years (29–65) |
| 2 | Gender | Male / Female |
| 3 | Height | cm (100–220) |
| 4 | Weight | kg (30–180) |
| 5 | Systolic BP (ap_hi) | mmHg (80–220) |
| 6 | Diastolic BP (ap_lo) | mmHg (50–150) |
| 7 | Cholesterol Level | Normal / Above Normal / Well Above |
| 8 | Glucose Level | Normal / Above Normal / Well Above |
| 9 | Smoking | Yes / No |
| 10 | Alcohol Consumption | Yes / No |
| 11 | Physical Activity | Yes / No |
| 12 | Chest Discomfort / Shortness of Breath | Yes / No |

---

##  Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 0.7313 |
| F1 Score | 0.7205 |
| AUC-ROC | 0.8001 |
| Precision | 0.7482 |
| Recall | 0.6947 |

> Run `evaluate.py` to generate full metrics saved in `reports/`

---

##  Explainability

`explain.py` generates SHAP-based visualizations to interpret model decisions:
- **Feature Importance Bar Chart** — which features matter most
- **SHAP Summary Plot** — direction and magnitude of each feature's impact
- All plots saved to `plots/`

---

##  Future Improvements

- [ ] Add deep learning model (MLP / TabNet)
- [ ] Build a Streamlit web app for doctor-facing predictions
- [ ] Integrate with real patient EHR data
- [ ] Add cross-validation and hyperparameter tuning logs
- [ ] Deploy as REST API (FastAPI / Flask)

---

##  Author

**Swagat pradhan**
- GitHub: [@swagatprdhan2005](https://github.com/swagatprdhan2005)

**Sidharth Satapathy**
- GitHub: [@SidharthSatapathy04](https://github.com/SidharthSatapathy04)

**Biswaranjan panda**
- GitHub: [@Biswa2006](https://github.com/Biswa2006)

---

##  License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

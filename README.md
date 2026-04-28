# Telco Customer Churn Prediction

Predicting which telecom customers are likely to churn using machine learning, enabling proactive retention strategies before customers leave.

---

## Live Demo
[View the live dashboard](https://your-username-telco-churn-prediction.streamlit.app)

---

## Problem Statement

Customer churn is a critical business problem in the telecom industry. Acquiring a new customer costs significantly more than retaining an existing one. This project builds a binary classification model on the IBM Telco Customer Churn dataset to identify at-risk customers, and provides interpretable insights into the key drivers of churn.

---

## Project Structure

```
telco-churn/
│
├── src/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
│
├── models/
│   ├── lgbm_final.pkl                          # Trained LightGBM model
│   ├── rf_final.pkl                            # Trained Random Forest model
│   ├── lr_final.pkl                            # Trained Logistic Regression model
│   ├── scaler.pkl                              # Fitted StandardScaler
│   └── feature_columns.pkl                    # Feature column names/order
│
├── Notebook/
│   ├──Notebook.ipynb                              # Full analysis and modelling notebook
├── requirements.txt                            # Python dependencies
└── README.md
```

---

## Dataset

- **Source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 features
- **Target:** `Churn` — whether a customer left within the last month
- **Class balance:** ~26% churn (imbalanced)

---

## Approach

### 1. Exploratory Data Analysis
- Class imbalance assessment
- Churn rate breakdown across contract type, internet service, payment method, tenure, and monthly charges
- Correlation heatmap across all features

### 2. Data Cleaning
- Converted `TotalCharges` from string to numeric; imputed 11 missing values with median
- Consolidated `No internet service` / `No phone service` to `No` to reduce cardinality
- Dropped `customerID` (no predictive value)
- Encoded binary columns and target variable

### 3. Feature Engineering
| Feature | Description |
|---|---|
| `charges_per_tenure` | Monthly charges divided by tenure — captures cost efficiency |
| `total_services` | Count of add-on services subscribed — proxy for switching cost |

### 4. Modelling
Three models trained with **5-fold Stratified Cross-Validation**:
- Logistic Regression (baseline, `class_weight='balanced'`)
- Random Forest (`max_depth=10`, `class_weight='balanced'`)
- LightGBM (early stopping, `scale_pos_weight` for imbalance)

Classification threshold optimised using the OOF Precision-Recall curve (F1 maximisation) rather than defaulting to 0.5.

---

## Results

| Model | CV ROC-AUC | CV Std | Test ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.844194 | 0.012966 | 0.862154 |
| Random Forest | 0.839443 | 0.015726 | 0.863163 |
| **LightGBM** | **0.841184** | **0.017926** | **0.862148** |

> All three models performed comparably (~0.862 Test ROC-AUC), suggesting the dataset has a performance ceiling near this level without further feature work or hyperparameter tuning.

---

## Key Findings

| Driver | Finding | Recommended Action |
|---|---|---|
| Contract type | Month-to-month customers churn at ~3x the rate of 2-year contracts | Offer discounts to upgrade to annual contracts |
| Tenure | 50% of churners leave within the first 12 months | Prioritise onboarding experience for new customers |
| Monthly Charges | Churners pay ~$15/month more on average | Flag high-charge, low-tenure customers for proactive outreach |
| Internet Service | Fibre optic customers churn at ~42% vs DSL at ~19% | Investigate fibre service quality issues |

---

## Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/telco-churn.git
cd telco-churn
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook Notebook.ipynb
```

### 4. Load the saved model
```python
import joblib
import pandas as pd

# Load model and preprocessing artifacts
lgbm_model = joblib.load('models/lgbm_final.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_cols = joblib.load('models/feature_columns.pkl')

# Prepare new data
new_data = new_data[feature_cols]  # enforce correct column order

# Predict
churn_probs = lgbm_model.predict_proba(new_data)[:, 1]
```

---

## Requirements

Key dependencies:
- `pandas`, `numpy`
- `scikit-learn`
- `lightgbm`
- `shap`
- `matplotlib`, `seaborn`
- `joblib`

Full list in `requirements.txt`.

---

## Limitations

- Dataset is static — a production model would need periodic retraining as customer behaviour shifts
- No hyperparameter tuning was performed; GridSearchCV or Optuna could improve results further
- SHAP interpretability is applied to LightGBM only

---

## Next Steps

- [ ] Hyperparameter tuning with Optuna
- [ ] Deploy model as a REST API using FastAPI
- [ ] Explore customer lifetime value weighting in the loss function
- [ ] Build a simple Streamlit dashboard for churn probability scoring

---

## Author

Daniel Jacobus Robbertse
[LinkedIn](https://www.linkedin.com/in/daniel-robbertse-a9695b335/) · [GitHub](https://github.com/Daniel577320)

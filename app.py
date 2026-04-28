import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
import re
matplotlib.use('Agg')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
) #000009

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    section[data-testid="stSidebar"] { background: #880009; border-right: 1px solid #e0e0e0; } 
    section[data-testid="stSidebar"] * { color: #fff6ff !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label {
        color: #c9d0db !important; font-size: 0.78rem !important;
        text-transform: uppercase; letter-spacing: 0.06em;
    }

    .main { background: #090c12; }

    .metric-card {
        background: #1e1e1e; border: 1px solid #1e2540;
        border-radius: 12px; padding: 1.4rem 1.6rem; text-align: center;
    }
    .metric-label { color: #5a606e; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.4rem; }
    .metric-value { font-size: 2rem; font-weight: 600; font-family: 'DM Mono', monospace; }

    .risk-low    { color: #4caf50; }
    .risk-medium { color: #ff9800; }
    .risk-high   { color: #f44336; }

    .section-header {
        color: #c9d0db; font-size: 1.4rem; text-transform: uppercase;
        letter-spacing: 0.01em; margin-bottom: 0.4rem;
        padding-bottom: 0.2rem; border-bottom: 1px solid #1e2540;
    }

    .prob-bar-wrap { background: #1a1e2b; border-radius: 999px; height: 10px; margin: 0.6rem 0 1.2rem; overflow: hidden; }
    .prob-bar-fill { height: 100%; border-radius: 999px; transition: width 0.5s ease; }

    button[data-baseweb="tab"] { color: #5a606e !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #c9d0db !important; border-bottom-color: #c9d0db !important; }

    .info-box { background: #131a28; border-left: 4px solid #1e2540; border-radius: 0 8px 8px 0; padding: 0.9rem 1.1rem; font-size: 0.85rem; color: #c9d0db; margin-bottom: 1.2rem; }

    .styled-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .styled-table th { background: #1e2540; color: #c9d0db; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06rem; padding: 0.7rem 1rem; text-align: left; border-bottom: 1px solid #1e2540; }
    .styled-table td { padding: 0.7rem 1rem; color: #c9d1e0; border-bottom: 1px solid #1a1e2b; }
    .styled-table tr:hover td { background: #1a1e2b; }
    .highlight { color: #4caf50; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    models_dir = os.path.join(base, "models")
    lgbm   = joblib.load(os.path.join(models_dir, "lgbm_final.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    lr     = joblib.load(os.path.join(models_dir, "lr_final.pkl"))
    rf     = joblib.load(os.path.join(models_dir, "rf_final.pkl"))
    try:
        feat_cols = joblib.load(os.path.join(models_dir, "feature_columns.pkl"))
    except FileNotFoundError:
        feat_cols = None
    return lgbm, scaler, lr, rf, feat_cols

lgbm_model, scaler, lr_model, rf_model, feat_cols = load_models()

# ── Feature pipeline ──────────────────────────────────────────────────────────
INTERNET_COLS = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']

def build_features(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    yes_no_cols = ['Partner','Dependents','PhoneService','MultipleLines',
                   'OnlineSecurity','OnlineBackup','DeviceProtection',
                   'TechSupport','StreamingTV','StreamingMovies','PaperlessBilling']
    for c in yes_no_cols:
        if c in df.columns:
            df[c] = df[c].map({'Yes': 1, 'No': 0})

    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['charges_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['total_services'] = df[INTERNET_COLS + ['PhoneService','MultipleLines']].sum(axis=1)

    df = pd.get_dummies(df, drop_first=True)

    # Align to training columns BEFORE sanitising (scaler expects original names)
    if feat_cols is not None:
        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0
        df = df[feat_cols]

    return df

BEST_THRESHOLD = 0.456

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Customer Information")
    st.markdown("<div class='section-header'>Demographics</div>", unsafe_allow_html=True)
    gender     = st.selectbox("Gender", ["Male", "Female"])
    senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner    = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    st.markdown("<div class='section-header'>Account Details</div>", unsafe_allow_html=True)
    tenure    = st.slider("Tenure (months)", 0, 72, 12)
    contract  = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment   = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
    monthly   = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total     = st.number_input("Total Charges ($)", 0.0, 9000.0, float(monthly * max(tenure, 1)), step=1.0)

    st.markdown("<div class='section-header'>Services</div>", unsafe_allow_html=True)
    phone      = st.selectbox("Phone Service", ["Yes", "No"])
    multi_line = st.selectbox("Multiple Lines", ["Yes", "No"])
    internet   = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

    col1, col2 = st.columns(2)
    with col1:
        online_sec   = st.selectbox("Online Security",   ["Yes", "No"], key="os")
        device_prot  = st.selectbox("Device Protection", ["Yes", "No"], key="dp")
        streaming_tv = st.selectbox("Streaming TV",      ["Yes", "No"], key="stv")
    with col2:
        online_bk    = st.selectbox("Online Backup",     ["Yes", "No"], key="ob")
        tech_sup     = st.selectbox("Tech Support",      ["Yes", "No"], key="ts")
        streaming_mv = st.selectbox("Streaming Movies",  ["Yes", "No"], key="sm")

    predict_btn = st.button("Predict Churn Risk", use_container_width=True, type="primary")

# ── Build input & run predictions ─────────────────────────────────────────────
raw_input = {
    "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner, "Dependents": dependents,
    "tenure": tenure, "PhoneService": phone, "MultipleLines": multi_line,
    "InternetService": internet, "OnlineSecurity": online_sec,
    "OnlineBackup": online_bk, "DeviceProtection": device_prot,
    "TechSupport": tech_sup, "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_mv, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly, "TotalCharges": total,
}

features = build_features(raw_input)

# LR uses scaler — needs original column names (spaces intact)
lr_prob = lr_model.predict_proba(scaler.transform(features.values))[:, 1][0]

# LightGBM and RF need sanitised column names (underscores)
features_sanitised = features.copy()
features_sanitised.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in features_sanitised.columns]

lgbm_prob = lgbm_model.predict_proba(features_sanitised)[:, 1][0]
rf_prob   = rf_model.predict_proba(features_sanitised)[:, 1][0]
ensemble  = np.mean([lgbm_prob, lr_prob, rf_prob])

def risk_label(p):
    if p < 0.35: return "Low Risk",    "risk-low",    "✅"
    if p < 0.60: return "Medium Risk", "risk-medium", "⚠️"
    return              "High Risk",   "risk-high",   "❌"

def bar_color(p):
    if p < 0.35: return "#4caf50"
    if p < 0.60: return "#ff9800"
    return "#f44336"

label, css_class, icon = risk_label(ensemble)

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("# Telco Customer Churn Predictor")
st.markdown("<div class='info-box'>Enter the customer details in the sidebar and click the button to see the churn risk prediction.</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 EDA & Insights", "📈 Model Performance"])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Ensemble Probability</div><div class='metric-value {css_class}'>{ensemble:.1%}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Risk Level</div><div class='metric-value {css_class}'>{icon} {label}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>LightGBM</div><div class='metric-value' style='color:#7b9cff'>{lgbm_prob:.1%}</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>Random Forest</div><div class='metric-value' style='color:#ff7b7b'>{rf_prob:.1%}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_bar, col_info = st.columns([2, 1])

    with col_bar:
        st.markdown("<div class='section-header'>Churn Probability</div>", unsafe_allow_html=True)
        fill_color = bar_color(ensemble)
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between; font-size:0.75rem; color:#5a606e; margin-bottom:4px;'>
            <span>0%</span><span>50%</span><span>100%</span>
        </div>
        <div class='prob-bar-wrap'>
            <div class='prob-bar-fill' style='width:{ensemble*100:.1f}%; background:{fill_color}'></div>
        </div>
        <div style='font-family: DM Mono, monospace; font-size:0.78rem; color:#5a6a8a;'>
            Threshold: {BEST_THRESHOLD:.2f} &nbsp;|&nbsp;
            Prediction: <span style='color:{fill_color}; font-weight:600;'>{"CHURN" if ensemble >= BEST_THRESHOLD else "RETAIN"}</span>
        </div>""", unsafe_allow_html=True)

    with col_info:
        st.markdown("<div class='section-header'>Model Breakdown</div>", unsafe_allow_html=True)
        for name, prob in [("LightGBM", lgbm_prob), ("Logistic Reg.", lr_prob), ("Random Forest", rf_prob)]:
            fc = bar_color(prob)
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem; font-size:0.82rem;'>
                <span style='color:#8a9abf'>{name}</span>
                <span style='font-family: DM Mono, monospace; color:{fc}; font-weight:600;'>{prob:.1%}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Key Risk Factors for This Customer</div>", unsafe_allow_html=True)

    risk_factors = []
    if contract == "Month-to-month":
        risk_factors.append(("Month-to-month contract", "Highest churn rate of any contract type (~42%)"))
    if tenure < 12:
        risk_factors.append(("Low tenure", f"Only {tenure} months — 50% of churners leave in the first year"))
    if monthly > 70:
        risk_factors.append(("High monthly charges", f"${monthly:.0f}/month — churners average ~$15 more than retained customers"))
    if internet == "Fiber optic":
        risk_factors.append(("Fibre optic subscriber", "Fibre customers churn at ~42% vs DSL at ~19%"))
    if online_sec == "No" and internet != "No":
        risk_factors.append(("No online security", "Add-on absence correlates with higher churn"))
    if not risk_factors:
        risk_factors.append(("Low risk profile", "No major churn risk factors detected for this customer"))

    for factor, explanation in risk_factors:
        st.markdown(f"""
        <div style='background:#13172a; border:1px solid #1e2540; border-radius:8px; padding:0.8rem 1rem; margin-bottom:0.5rem;'>
            <span style='color:#127775; font-weight:500'>{factor}</span><br>
            <span style='color:#5a6a8a; font-size:0.8rem'>{explanation}</span>
        </div>""", unsafe_allow_html=True)

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-header'>Churn Drivers</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>The plots below show churn rates across key customer segments. Percentage labels show the proportion of customers in each category who churned.</div>", unsafe_allow_html=True)

    figures_dir = os.path.join(os.path.dirname(__file__), "outputs", "figures")
    fig_map = {
        "Churn Rate by Category":          "churn_rate_categorical.png",
        "Numerical Feature Distributions": "Numerical_Features.png",
    }
    for title, fname in fig_map.items():
        path = os.path.join(figures_dir, fname)
        if os.path.exists(path):
            st.markdown(f"**{title}**")
            st.image(path, use_container_width=True)
        else:
            st.info(f"Figure not found: `outputs/figures/{fname}` — run notebook to generate it.")

    st.markdown("<br><div class='section-header'>Business Insights</div>", unsafe_allow_html=True)
    st.markdown("""
    <table class='styled-table'>
        <thead><tr><th>Driver</th><th>Finding</th><th>Recommended Action</th></tr></thead>
        <tbody>
        <tr><td><span class='highlight'>Contract Type</span></td><td>Month-to-month customers churn at ~3× the rate of 2-year contracts</td><td>Offer discounts to upgrade to annual contracts</td></tr>
        <tr><td><span class='highlight'>Tenure</span></td><td>50% of churners leave within the first 12 months</td><td>Prioritise onboarding experience for new customers</td></tr>
        <tr><td><span class='highlight'>Monthly Charges</span></td><td>Churners pay ~$15/month more on average</td><td>Flag high-charge, low-tenure customers for proactive outreach</td></tr>
        <tr><td><span class='highlight'>Internet Service</span></td><td>Fibre optic customers churn at ~42% vs DSL at ~19%</td><td>Investigate fibre service quality issues</td></tr>
        </tbody>
    </table>""", unsafe_allow_html=True)

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-header'>Model Comparison</div>", unsafe_allow_html=True)
    st.markdown("""
    <table class='styled-table'>
        <thead><tr><th>Model</th><th>CV ROC-AUC</th><th>CV Std</th><th>Test ROC-AUC</th></tr></thead>
        <tbody>
        <tr><td>Logistic Regression</td><td>~0.841</td><td>low</td><td>0.8622</td></tr>
        <tr><td>Random Forest</td><td>~0.841</td><td>low</td><td>0.8632</td></tr>
        <tr style='background:#13172a'><td><span class='highlight'>LightGBM ✓</span></td><td><span class='highlight'>~0.841</span></td><td><span class='highlight'>low</span></td><td><span class='highlight'>0.8621</span></td></tr>
        </tbody>
    </table>
    <p style='color:#5a6a8a; font-size:0.78rem; margin-top:0.6rem;'>LightGBM selected as primary model. Ensemble used for live predictions.</p>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    figures_dir = os.path.join(os.path.dirname(__file__), "outputs", "figures")

    with col_l:
        st.markdown("<div class='section-header'>ROC & Precision-Recall Curves</div>", unsafe_allow_html=True)
        roc_path = os.path.join(figures_dir, "ROC_&_PR_Curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
        else:
            st.info("Figure not found: `outputs/figures/ROC_&_PR_Curves.png`")

    with col_r:
        st.markdown("<div class='section-header'>Confusion Matrix (Test Set)</div>", unsafe_allow_html=True)
        cm_path = os.path.join(figures_dir, "LightGBM_ConfusionMatrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.info("Figure not found: `outputs/figures/LightGBM_ConfusionMatrix.png`")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>SHAP Feature Importance</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>SHAP (SHapley Additive exPlanations) shows each feature's contribution to individual predictions. Red = pushes toward churn, blue = pushes away from churn.</div>", unsafe_allow_html=True)

    shap_path = os.path.join(figures_dir, "ShapSummary.png")
    if os.path.exists(shap_path):
        st.image(shap_path, use_container_width=True)
    else:
        st.info("Figure not found: `outputs/figures/ShapSummary.png`")

    fi_path = os.path.join(figures_dir, "Feature_Importance.png")
    if os.path.exists(fi_path):
        st.markdown("<div class='section-header'>LightGBM Feature Importance</div>", unsafe_allow_html=True)
        st.image(fi_path, use_container_width=True)

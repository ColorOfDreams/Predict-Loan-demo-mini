import streamlit as st
import pandas as pd
import joblib

def feature_engineering(df, is_train=True):
    df = df.copy()

    # ========================
    # 1. Ratio / Burden features
    # ========================

    eps = 1e-6
    df["loan_to_income"] = df["loan_amount"] / (df["annual_income"] + eps)

    df["loan_per_credit"] = df["loan_amount"] / (df["credit_score"] + eps)

    # ========================
    # 2. Interest burden
    # ========================
    df["interest_burden"] = df["loan_amount"] * df["interest_rate"] / 100

    df["interest_income_ratio"] = (
        df["interest_burden"] / (df["annual_income"] + eps)
    )


    # =======================
    # 2. Grade risk mapping
    # ========================
    grade_risk_map = {
        'A1': 1.0, 'A2': 1.2, 'A3': 1.4, 'A4': 1.6, 'A5': 1.8,
        'B1': 2.0, 'B2': 2.2, 'B3': 2.4, 'B4': 2.6, 'B5': 2.8,
        'C1': 3.0, 'C2': 3.2, 'C3': 3.4, 'C4': 3.6, 'C5': 3.8,
        'D1': 4.0, 'D2': 4.2, 'D3': 4.4, 'D4': 4.6, 'D5': 4.8,
        'E1': 5.0, 'E2': 5.2, 'E3': 5.4, 'E4': 5.6, 'E5': 5.8,
        'F1': 6.0, 'F2': 6.2, 'F3': 6.4, 'F4': 6.6, 'F5': 6.8
    }
    df["grade_risk"] = df["grade_subgrade"].map(grade_risk_map)
    df["grade_risk"] = df["grade_risk"].astype("float32")
    purpose_map = {
        "Home": "low",
        "Business": "low",

        "Car": "medium",
        "Other": "medium",
        "Debt consolidation": "medium",
        "Vacation": "medium",

        "Education": "high",
        "Medical": "high"
    }

    df["loan_purpose_group"] = df["loan_purpose"].map(purpose_map)


    return df


# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Credit Risk Demo",
    layout="centered"
)

st.title("💳 Credit Risk Prediction Demo")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = joblib.load("lgbm_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model()

# ---------------- FEATURE LIST ----------------
num_features = [
    "annual_income",
    "debt_to_income_ratio",
    "credit_score",
    "loan_amount",
    "interest_rate",
    "loan_to_income",
    "interest_burden",
    "loan_per_credit",
    "interest_income_ratio",
    "grade_risk",
]

cat_features = [
    "employment_status",
    "loan_purpose_group",
    "education_level",
    "marital_status"
]

features = num_features + cat_features

# ---------------- INPUT MODE ----------------
mode = st.radio("Chọn cách nhập dữ liệu", ["Nhập tay", "Upload CSV"])

# ================= MANUAL INPUT =================
if mode == "Nhập tay":
    with st.form("input_form"):
        annual_income = st.number_input("Annual Income", 1000, 1_000_000, 50000)
        loan_amount = st.number_input("Loan Amount", 1000, 1_000_000, 10000)
        interest_rate = st.number_input("Interest Rate (%)", 0.0, 50.0, 10.0)
        credit_score = st.number_input("Credit Score", 300, 850, 650)
        debt_to_income_ratio = st.slider("Debt to Income Ratio", 0.0, 1.0, 0.3)

        grade_subgrade = st.selectbox(
            "Grade Subgrade",
            [
                "A1","A2","A3","A4","A5",
                "B1","B2","B3","B4","B5",
                "C1","C2","C3","C4","C5",
                "D1","D2","D3","D4","D5",
                "E1","E2","E3","E4","E5",
                "F1","F2","F3","F4","F5"
            ]
        )
        employment_status = st.selectbox(
            "Employment Status",
            ["Employed", "Self-employed", "Unemployed", "Student", "Retired"]
        )

        loan_purpose = st.selectbox(
            "Loan Purpose",
            [
                "Home",
                "Business",
                "Car",
                "Other",
                "Debt consolidation",
                "Vacation",
                "Education",
                "Medical"
            ]
        )

        education_level = st.selectbox(
            "Education Level",
            ["High School", "Bachelor's", "Master's", "PhD", "Other"]
        )

        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced", "Widowed"]
        )

        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([{
            "annual_income": annual_income,
            "loan_amount": loan_amount,
            "interest_rate": interest_rate,
            "credit_score": credit_score,
            "debt_to_income_ratio": debt_to_income_ratio,
            "grade_subgrade": grade_subgrade,
            "employment_status": employment_status,
            "loan_purpose": loan_purpose,
            "education_level": education_level,
            "marital_status": marital_status,
        }])

        df_fe = feature_engineering(input_df)
        X = df_fe[features]
        X_proc = preprocessor.transform(X)

        proba = model.predict_proba(X_proc)[0, 1]

        st.success(f"✅ Xác suất trả nợ: **{proba:.2%}**")

# ================= CSV UPLOAD =================
else:
    file = st.file_uploader("Upload CSV test file", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df_fe = feature_engineering(df)

        X = df_fe[features]
        X_proc = preprocessor.transform(X)
        proba = model.predict_proba(X_proc)[:, 1]

        df["loan_paid_back"] = proba

        st.dataframe(df.head())

        st.download_button(
            label="📥 Download Result CSV",
            data=df.to_csv(index=False),
            file_name="prediction.csv",
            mime="text/csv"
        )

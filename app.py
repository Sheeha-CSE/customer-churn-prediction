import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------

model = joblib.load("customer_churn_model.pkl")

# Model accuracy obtained during testing
MODEL_ACCURACY = 78.54

# -----------------------------
# CUSTOM CSS
# -----------------------------

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 25px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------

st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict whether a telecom customer is likely to churn.</div>',
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# CUSTOMER INPUTS
# -----------------------------

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No phone service", "No", "Yes"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=800.0
    )

with col3:

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No internet service", "No", "Yes"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No internet service", "No", "Yes"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No internet service", "No", "Yes"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No internet service", "No", "Yes"]
    )

col4, col5 = st.columns(2)

with col4:

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No internet service", "No", "Yes"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No internet service", "No", "Yes"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col5:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

st.divider()

# -----------------------------
# CREATE INPUT DATA
# -----------------------------

input_data = pd.DataFrame({

    "gender": [
        1 if gender == "Male" else 0
    ],

    "SeniorCitizen": [
        senior_citizen
    ],

    "Partner": [
        1 if partner == "Yes" else 0
    ],

    "Dependents": [
        1 if dependents == "Yes" else 0
    ],

    "tenure": [
        tenure
    ],

    "PhoneService": [
        1 if phone_service == "Yes" else 0
    ],

    "PaperlessBilling": [
        1 if paperless_billing == "Yes" else 0
    ],

    "MonthlyCharges": [
        monthly_charges
    ],

    "TotalCharges": [
        total_charges
    ],

    "MultipleLines_No phone service": [
        1 if multiple_lines == "No phone service" else 0
    ],

    "MultipleLines_Yes": [
        1 if multiple_lines == "Yes" else 0
    ],

    "InternetService_Fiber optic": [
        1 if internet_service == "Fiber optic" else 0
    ],

    "InternetService_No": [
        1 if internet_service == "No" else 0
    ],

    "OnlineSecurity_No internet service": [
        1 if online_security == "No internet service" else 0
    ],

    "OnlineSecurity_Yes": [
        1 if online_security == "Yes" else 0
    ],

    "OnlineBackup_No internet service": [
        1 if online_backup == "No internet service" else 0
    ],

    "OnlineBackup_Yes": [
        1 if online_backup == "Yes" else 0
    ],

    "DeviceProtection_No internet service": [
        1 if device_protection == "No internet service" else 0
    ],

    "DeviceProtection_Yes": [
        1 if device_protection == "Yes" else 0
    ],

    "TechSupport_No internet service": [
        1 if tech_support == "No internet service" else 0
    ],

    "TechSupport_Yes": [
        1 if tech_support == "Yes" else 0
    ],

    "StreamingTV_No internet service": [
        1 if streaming_tv == "No internet service" else 0
    ],

    "StreamingTV_Yes": [
        1 if streaming_tv == "Yes" else 0
    ],

    "StreamingMovies_No internet service": [
        1 if streaming_movies == "No internet service" else 0
    ],

    "StreamingMovies_Yes": [
        1 if streaming_movies == "Yes" else 0
    ],

    "Contract_One year": [
        1 if contract == "One year" else 0
    ],

    "Contract_Two year": [
        1 if contract == "Two year" else 0
    ],

    "PaymentMethod_Credit card (automatic)": [
        1 if payment_method == "Credit card (automatic)" else 0
    ],

    "PaymentMethod_Electronic check": [
        1 if payment_method == "Electronic check" else 0
    ],

    "PaymentMethod_Mailed check": [
        1 if payment_method == "Mailed check" else 0
    ]
})

# Make sure columns are in exactly the same order as training
if hasattr(model, "feature_names_in_"):
    input_data = input_data[
        list(model.feature_names_in_)
    ]

# -----------------------------
# PREDICTION BUTTON
# -----------------------------

st.markdown("### 🔮 Prediction")

predict_button = st.button(
    "🔮 Predict Churn",
    use_container_width=True,
    type="primary"
)

if predict_button:

    # -----------------------------
    # PREDICTION
    # -----------------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    churn_probability = probability[1] * 100

    stay_probability = probability[0] * 100

    st.divider()

    # -----------------------------
    # RESULT
    # -----------------------------

    if prediction == 1:

        st.error(
            "⚠️ Customer is likely to CHURN"
        )

        risk_level = "HIGH RISK"

        recommendation = (
            "Consider contacting this customer with a retention offer, "
            "better support, or a suitable plan."
        )

    else:

        st.success(
            "✅ Customer is likely to STAY"
        )

        if churn_probability < 30:
            risk_level = "LOW RISK"
        else:
            risk_level = "MEDIUM RISK"

        recommendation = (
            "The customer currently shows a lower likelihood of churn. "
            "Continue providing good service and engagement."
        )

    # -----------------------------
    # RESULT METRICS
    # -----------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Churn Probability",
            f"{churn_probability:.2f}%"
        )

    with col2:

        st.metric(
            "Stay Probability",
            f"{stay_probability:.2f}%"
        )

    with col3:

        st.metric(
            "Model Accuracy",
            f"{MODEL_ACCURACY:.2f}%"
        )

    st.write("### 🎯 Risk Level")

    if churn_probability >= 70:

        st.error(
            f"🔴 {risk_level}"
        )

    elif churn_probability >= 30:

        st.warning(
            f"🟡 {risk_level}"
        )

    else:

        st.success(
            f"🟢 {risk_level}"
        )

    # -----------------------------
    # PROBABILITY BAR
    # -----------------------------

    st.write("### 📈 Churn Probability")

    st.progress(
        min(churn_probability / 100, 1.0)
    )

    st.write(
        f"**{churn_probability:.2f}%** probability that this customer will churn."
    )

    # -----------------------------
    # RECOMMENDATION
    # -----------------------------

    st.write("### 💡 Recommendation")

    st.info(
        recommendation
    )

    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------

    if hasattr(model, "feature_importances_"):

        st.divider()

        st.write("### 🌳 Top Factors Used by the Model")

        feature_importance = pd.DataFrame({
            "Feature": input_data.columns,
            "Importance": model.feature_importances_
        })

        feature_importance = (
            feature_importance
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(10)
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.barh(
            feature_importance["Feature"][::-1],
            feature_importance["Importance"][::-1]
        )

        ax.set_xlabel(
            "Importance"
        )

        ax.set_title(
            "Top 10 Features Influencing the Model"
        )

        plt.tight_layout()

        st.pyplot(fig)

        st.caption(
            "Feature importance shows which variables the Random Forest "
            "model relied on most overall. It does not mean that every "
            "individual customer is affected by each feature in the same way."
        )

    # -----------------------------
    # MODEL INFORMATION
    # -----------------------------

    st.divider()

    st.write("### 🤖 Model Information")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.write("**Algorithm**")
        st.write("Random Forest Classifier")

    with info_col2:
        st.write("**Training Samples**")
        st.write("5,625")

    with info_col3:
        st.write("**Testing Samples**")
        st.write("1,407")
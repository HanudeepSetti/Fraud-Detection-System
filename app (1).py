
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="JP Morgan Fraud Detector", layout="wide")
st.title("🏦 JP Morgan Fraud Detection")
st.write("Enter transaction details to check whether it is fraudulent.")

MODEL_PATH = "/content/drive/MyDrive/fraud_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ── SIDEBAR INPUTS ────────────────────────────────────────────────────────────
st.sidebar.header("Enter Transaction Details")

transaction_amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=5000.0)
account_balance    = st.sidebar.number_input("Account Balance", min_value=0.0, value=50000.0)
risk_score         = st.sidebar.slider("Risk Score", min_value=0, max_value=100, value=50)
credit_rating      = st.sidebar.slider("Credit Rating", min_value=1, max_value=10, value=5)
tenure_months      = st.sidebar.number_input("Tenure Months", min_value=0, value=24)
account_type       = st.sidebar.selectbox("Account Type", ["Savings", "Current", "Business"])
transaction_type   = st.sidebar.selectbox("Transaction Type", ["Deposit", "Withdrawal", "Payment"])
region             = st.sidebar.selectbox("Region", ["North", "South", "East", "West"])
activity_level     = st.sidebar.selectbox("Activity Level", ["Low", "Medium", "High"])
volume_group       = st.sidebar.selectbox("Volume Group", ["Low Volume", "High Volume"])
seg_customer       = st.sidebar.selectbox("Customer Segment", ["Low", "Medium", "Highest"])
manager            = st.sidebar.selectbox("Manager", ["Yes", "No"])
product            = st.sidebar.selectbox("Product", ["Savings", "Current", "Business"])
firm               = st.sidebar.selectbox("Firm", ["Retail", "Corporate", "SME"])

if st.sidebar.button("Check for Fraud", type="primary"):

    try:
        # Derived values
        total_credit     = float(transaction_amount) if transaction_type == "Deposit" else 0.0
        total_debit      = float(transaction_amount) if transaction_type in ["Withdrawal", "Payment"] else 0.0
        net_transaction  = total_credit - total_debit
        avg_balance      = float(account_balance)
        txn_volume       = total_credit + total_debit
        txn_count        = 5
        gap              = 2.0
        z                = float(risk_score)

        input_data = pd.DataFrame([{
            # int64
            "TransactionID":              np.int64(1001),
            "transaction_count":          np.int64(txn_count),
            # object (string) — must stay as str, not numeric
            "CustomerID":                 "C_001",
            "AccountID":                  "A_001",
            "fraud_risk_flag":            "Low",
            # object categorical
            "AccountType":                str(account_type),
            "TransactionType":            str(transaction_type),
            "Product":                    str(product),
            "Firm":                       str(firm),
            "Region":                     str(region),
            "Manager":                    str(manager),
            "Activity_Level":             str(activity_level),
            "Seg_Customer":               str(seg_customer),
            "Volume_Group":               str(volume_group),
            # int32
            "Year":                       np.int32(2026),
            "Month":                      np.int32(5),
            # int64
            "CreditRating":               np.int64(credit_rating),
            "TenureMonths":               np.int64(tenure_months),
            # float64
            "TransactionAmount":          np.float64(transaction_amount),
            "AccountBalance":             np.float64(account_balance),
            "RiskScore":                  np.float64(risk_score),
            "total_credit":               np.float64(total_credit),
            "total_debit":                np.float64(total_debit),
            "Net_transaction":            np.float64(net_transaction),
            "gap_days":                   np.float64(gap),
            "Account_Transaction_Volume": np.float64(txn_volume),
            "Account_Avg_Balance":        np.float64(avg_balance),
            "z_score":                    np.float64(z),
            # bool — must be actual Python bool
            "Dormant_Flag":               bool(gap >= 60),
            "High_net_inflow":            bool(net_transaction > 0 and account_balance > avg_balance),
            "High_Freq-low_bal":          bool(txn_count > 4 and account_balance < avg_balance),
            "Nil_neg_acc":                bool(account_balance <= 0),
        }])

        # ── Enforce exact dtypes to match training data ───────────────────────
        input_data["TransactionID"]              = input_data["TransactionID"].astype(np.int64)
        input_data["transaction_count"]          = input_data["transaction_count"].astype(np.int64)
        input_data["CreditRating"]               = input_data["CreditRating"].astype(np.int64)
        input_data["TenureMonths"]               = input_data["TenureMonths"].astype(np.int64)
        input_data["Year"]                       = input_data["Year"].astype(np.int32)
        input_data["Month"]                      = input_data["Month"].astype(np.int32)
        input_data["Dormant_Flag"]               = input_data["Dormant_Flag"].astype(bool)
        input_data["High_net_inflow"]            = input_data["High_net_inflow"].astype(bool)
        input_data["High_Freq-low_bal"]          = input_data["High_Freq-low_bal"].astype(bool)
        input_data["Nil_neg_acc"]                = input_data["Nil_neg_acc"].astype(bool)

        for col in ["CustomerID", "AccountID", "fraud_risk_flag", "AccountType",
                    "TransactionType", "Product", "Firm", "Region", "Manager",
                    "Activity_Level", "Seg_Customer", "Volume_Group"]:
            input_data[col] = input_data[col].astype(str)

        for col in ["TransactionAmount", "AccountBalance", "RiskScore", "total_credit",
                    "total_debit", "Net_transaction", "gap_days",
                    "Account_Transaction_Volume", "Account_Avg_Balance", "z_score"]:
            input_data[col] = input_data[col].astype(np.float64)

        # ── Reorder columns to match X_train exactly ─────────────────────────
        expected_cols = [
            "TransactionID", "CustomerID", "AccountID", "AccountType",
            "TransactionType", "Product", "Firm", "Region", "Manager",
            "TransactionAmount", "AccountBalance", "RiskScore", "CreditRating",
            "TenureMonths", "fraud_risk_flag", "Year", "Month",
            "total_credit", "total_debit", "Net_transaction", "gap_days",
            "Dormant_Flag", "transaction_count", "Activity_Level",
            "Account_Transaction_Volume", "Account_Avg_Balance", "Seg_Customer",
            "High_net_inflow", "High_Freq-low_bal", "Nil_neg_acc",
            "z_score", "Volume_Group"
        ]
        input_data = input_data[expected_cols]

        # ── Predict ───────────────────────────────────────────────────────────
        prediction  = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else float(prediction)

        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.error("🚨 FRAUD DETECTED")
            else:
                st.success("✅ Transaction Looks Legitimate")
        with col2:
            st.metric("Fraud Probability", f"{probability:.2%}")

        st.progress(float(probability))
        st.subheader("Input Summary")
        st.dataframe(input_data)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        if 'input_data' in dir():
            st.write("**Columns:**", list(input_data.columns))
            st.write("**Dtypes:**", input_data.dtypes.to_dict())

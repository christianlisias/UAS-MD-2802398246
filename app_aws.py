"""
app_aws.py
Aplikasi Streamlit yang memanggil SageMaker Endpoint (bukan model lokal) untuk
prediksi Credit Score. Dipakai untuk deployment cloud (soal 2b).

Kredensial AWS dibaca dari st.secrets (saat deploy di Streamlit Community Cloud)
atau dari environment variable / AWS CLI profile (saat dijalankan lokal).

Jalankan dengan:
    streamlit run app_aws.py
"""

import json
import os

import boto3
import streamlit as st

REGION = "us-east-1"
ENDPOINT_NAME = "credit-score-endpoint"


def get_runtime_client():
    """Buat boto3 client sagemaker-runtime.

    Prioritas kredensial:
    1. st.secrets (untuk Streamlit Community Cloud -> Settings -> Secrets)
    2. environment variable / default AWS credential chain (untuk lokal)
    """
    if "aws" in st.secrets:
        return boto3.client(
            "sagemaker-runtime",
            region_name=st.secrets["aws"].get("region", REGION),
            aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
            aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
            aws_session_token=st.secrets["aws"].get("aws_session_token"),
        )
    return boto3.client("sagemaker-runtime", region_name=os.environ.get("AWS_REGION", REGION))


runtime = get_runtime_client()

st.title("Credit Score Prediction (AWS SageMaker Endpoint)")

age = st.number_input("Age", min_value=14, max_value=100, value=30)
occupation = st.text_input("Occupation", value="Engineer")
annual_income = st.number_input("Annual Income", min_value=0.0, value=50000.0)
monthly_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=4000.0)
num_bank_accounts = st.number_input("Num Bank Accounts", min_value=0, max_value=20, value=3)
num_credit_card = st.number_input("Num Credit Card", min_value=0, max_value=20, value=4)
interest_rate = st.number_input("Interest Rate (%)", min_value=0, max_value=100, value=15)
num_of_loan = st.number_input("Num of Loan", min_value=0, max_value=20, value=2)
type_of_loan = st.text_input("Type of Loan (pisahkan dengan koma)", value="Auto Loan, Personal Loan")
delay_from_due = st.number_input("Delay from Due Date (hari)", min_value=0, max_value=200, value=10)
num_delayed_payment = st.number_input("Num of Delayed Payment", min_value=0, max_value=50, value=5)
changed_credit_limit = st.number_input("Changed Credit Limit", value=5.0)
num_credit_inquiries = st.number_input("Num Credit Inquiries", min_value=0.0, value=4.0)
credit_mix = st.selectbox("Credit Mix", ["Good", "Standard", "Bad"], index=1)
outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=1000.0)
credit_utilization = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=30.0)
credit_history_age = st.text_input("Credit History Age (format: 'X Years and Y Months')", value="10 Years and 0 Months")
payment_min_amount = st.selectbox("Payment of Min Amount", ["Yes", "No", "NM"], index=0)
total_emi = st.number_input("Total EMI per month", min_value=0.0, value=100.0)
amount_invested = st.number_input("Amount Invested Monthly", min_value=0.0, value=150.0)
payment_behaviour = st.text_input("Payment Behaviour", value="Low_spent_Small_value_payments")
monthly_balance = st.number_input("Monthly Balance", value=300.0)

if st.button("Prediksi Credit Score"):
    record = {
        "Age": age,
        "Occupation": occupation,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": num_of_loan,
        "Type_of_Loan": type_of_loan,
        "Delay_from_due_date": delay_from_due,
        "Num_of_Delayed_Payment": num_delayed_payment,
        "Changed_Credit_Limit": changed_credit_limit,
        "Num_Credit_Inquiries": num_credit_inquiries,
        "Credit_Mix": credit_mix,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_utilization,
        "Credit_History_Age": credit_history_age,
        "Payment_of_Min_Amount": payment_min_amount,
        "Total_EMI_per_month": total_emi,
        "Amount_invested_monthly": amount_invested,
        "Payment_Behaviour": payment_behaviour,
        "Monthly_Balance": monthly_balance,
    }

    payload = json.dumps({"instances": [record]})

    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=payload,
        )
        result = json.loads(response["Body"].read().decode("utf-8"))

        st.write("Hasil Prediksi:", result["labels"][0])

        st.write("Probabilitas tiap kelas:")
        for cls, prob in zip(result["classes"], result["probabilities"][0]):
            st.write(f"{cls}: {prob:.4f}")

    except Exception as e:
        st.error(f"Gagal memanggil SageMaker endpoint: {e}")

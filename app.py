import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="AI Cost Estimator",
    page_icon="💰",
    layout="wide"
)

st.title("💰 AI-Based Project Cost Estimator")

# ------------------ LOAD DATA ------------------
df = pd.read_csv("3.data/projects.csv")
model = joblib.load("model.pkl")

# ------------------ DATA PREVIEW ------------------
st.subheader("📊 Dataset Overview")
st.dataframe(df)

# ------------------ SELECT PROJECT ------------------
st.subheader("📌 Select Existing Project")

selected_index = st.selectbox("Choose Project", df.index)
selected_project = df.loc[selected_index]

st.write("Selected Project:", selected_project)

# ------------------ INPUTS ------------------
st.subheader("⚙️ Modify Project Inputs")

col1, col2 = st.columns(2)

with col1:
    team_exp = st.number_input("Team Experience", value=int(selected_project["team_experience"]))
    manager_exp = st.number_input("Manager Experience", value=int(selected_project["manager_experience"]))
    duration = st.number_input("Duration (months)", value=int(selected_project["duration_months"]))
    transactions = st.number_input("Transactions", value=int(selected_project["transactions"]))

with col2:
    entities = st.number_input("Entities", value=int(selected_project["entities"]))
    adjustment = st.number_input("Adjustment Factor", value=int(selected_project["adjustment"]))
    language = st.number_input("Language Type (1/2/3)", value=int(selected_project["language"]))
    hourly_rate = st.number_input("Hourly Rate (₹)", value=500)

# ------------------ PREDICTION ------------------
if st.button("💰 Estimate Cost"):

    input_data = [[
        team_exp,
        manager_exp,
        duration,
        transactions,
        entities,
        adjustment,
        language
    ]]

    predicted_hours = model.predict(input_data)[0]

    # Cost & duration
    estimated_cost = predicted_hours * hourly_rate
    estimated_months = predicted_hours / (160 * 5)

    # Uncertainty
    uncertainty = predicted_hours * 0.15

    st.subheader("📈 AI Prediction Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Effort", f"{predicted_hours:,.0f} hrs")
    c2.metric("Cost", f"₹{estimated_cost:,.0f}")
    c3.metric("Duration", f"{estimated_months:.1f} months")

    st.info(f"📊 Confidence Range: ± {uncertainty:.0f} hrs")

    # ------------------ COCOMO ------------------
    cocomo_effort = 2.94 * (transactions ** 1.1)

    st.subheader("📐 COCOMO Comparison")
    st.write(f"COCOMO Estimated Effort: {cocomo_effort:.0f} hrs")

    # ------------------ SCENARIOS ------------------
    st.subheader("⚖️ Scenario Analysis")

    low_cost = predicted_hours * 400
    high_cost = predicted_hours * 800

    st.write("Low Budget Scenario: ₹", int(low_cost))
    st.write("High Budget Scenario: ₹", int(high_cost))

# ------------------ VISUALIZATION ------------------
st.subheader("📊 Data Insights")

st.bar_chart(df["effort_hours"])
st.line_chart(df["duration_months"])

# ------------------ REPORT ------------------
if st.button("📄 Generate Report"):
    report = f"""
    AI Cost Estimation Report

    Predicted Effort: {predicted_hours:.0f} hrs
    Estimated Cost: ₹{estimated_cost:.0f}
    Duration: {estimated_months:.1f} months
    """

    st.download_button("Download Report", report, file_name="report.txt")
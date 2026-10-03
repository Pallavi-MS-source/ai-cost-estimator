import streamlit as st

st.set_page_config(
    page_title="AI Project Cost Estimator",
    page_icon="💰",
    layout="wide"
)

st.title("AI-Based Project Cost Estimator")
st.write("Enter software project details to estimate effort, cost, and duration.")

st.subheader("Project Details")

project_name = st.text_input("Project name", "Sample Web Application")

col1, col2 = st.columns(2)

with col1:
    project_size = st.number_input(
        "Estimated project size (Function Points)",
        min_value=1,
        value=100
    )

    complexity = st.selectbox(
        "Project complexity",
        ["Low", "Medium", "High"]
    )

    team_size = st.number_input(
        "Available team members",
        min_value=1,
        value=5
    )

with col2:
    experience = st.selectbox(
        "Average team experience",
        ["Beginner", "Intermediate", "Experienced"]
    )

    methodology = st.selectbox(
        "Development methodology",
        ["Agile", "Waterfall", "DevOps", "Hybrid"]
    )

    hourly_rate = st.number_input(
        "Average hourly cost per employee (₹)",
        min_value=1,
        value=500
    )

if st.button("Estimate Project Cost"):
    st.success("Your input form is working successfully.")

    st.write("Project:", project_name)
    st.write("Size:", project_size, "Function Points")
    st.write("Complexity:", complexity)
    st.write("Team size:", team_size)
    st.write("Hourly rate: ₹", hourly_rate)

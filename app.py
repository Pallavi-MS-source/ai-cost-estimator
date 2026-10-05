import streamlit as st
import pandas as pd
import joblib
import os

# ------------------ CONFIG ------------------
st.set_page_config(page_title="AI Cost Estimator", layout="wide")

# ------------------ STYLING ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f3e8ff, #e9d5ff);
}
html, body {
    color: #1e1b4b !important;
}
h1, h2, h3 {
    color: #4c1d95 !important;
}
.stButton>button {
    background: linear-gradient(90deg, #9333ea, #7e22ce);
    color: white;
    border-radius: 10px;
    font-weight: bold;
}
label {
    color: #1e1b4b !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------ SESSION ------------------
if "page" not in st.session_state:
    st.session_state.page = "login"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ------------------ LOGIN ------------------
def login_page():
    st.title("🔐 Login")

    username = st.text_input("Username", placeholder="Enter username")
    password = st.text_input("Password", type="password", placeholder="Enter password")

    if st.button("Login"):
        if username and password:
            st.session_state.logged_in = True
            st.session_state.page = "input"
            st.rerun()
        else:
            st.error("Please enter username and password")

# ------------------ LOAD DATA ------------------
DATA_PATH = os.path.join("3.data", "projects.csv")
df = pd.read_csv(DATA_PATH)
model = joblib.load("model.pkl")

# ------------------ INPUT PAGE ------------------
def input_page():
    st.title("📊 Enter Project Details")

    if st.button("🔓 Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.rerun()

    col1, col2 = st.columns(2)

    with col1:
        team_exp = st.number_input("Team Experience (0-10)", 0, 10, value=None, placeholder="e.g. 3")
        manager_exp = st.number_input("Manager Experience (0-10)", 0, 10, value=None, placeholder="e.g. 5")
        duration = st.number_input("Duration (months)", 1, 60, value=None, placeholder="e.g. 12")
        transactions = st.number_input("Transactions", 1, 1000, value=None, placeholder="e.g. 200")

    with col2:
        entities = st.number_input("Entities", 1, 500, value=None, placeholder="e.g. 100")
        adjustment = st.number_input("Adjustment Factor", 1, 100, value=None, placeholder="e.g. 30")
        language = st.selectbox("Language Type", [None, 1, 2, 3],
                                format_func=lambda x: "Select language" if x is None else f"Type {x}")
        hourly_rate = st.number_input("Hourly Rate (₹)", 100, 5000, value=None, placeholder="e.g. 500")

    if st.button("➡️ Estimate"):
        if None in [team_exp, manager_exp, duration, transactions, entities, adjustment, language, hourly_rate]:
            st.error("⚠️ Please fill all fields")
        else:
            input_data = [[team_exp, manager_exp, duration, transactions,
                           entities, adjustment, language]]

            predicted_hours = model.predict(input_data)[0]

            st.session_state.result = {
                "hours": predicted_hours,
                "cost": predicted_hours * hourly_rate,
                "months": predicted_hours / (160 * 5),
                "transactions": transactions,
                "team_exp": team_exp,
                "manager_exp": manager_exp
            }

            st.session_state.page = "result"
            st.rerun()

# ------------------ RESULT PAGE ------------------
def result_page():
    st.title("📈 Results")

    res = st.session_state.result

    c1, c2, c3 = st.columns(3)
    c1.metric("Effort", f"{res['hours']:.0f} hrs")
    c2.metric("Cost", f"₹{res['cost']:.0f}")
    c3.metric("Duration", f"{res['months']:.1f} months")

    # AI Explanation
    st.subheader("🤖 AI Explanation")

    explanation = f"""
    The model predicts **{res['hours']:.0f} hours** based on your inputs.

    • Higher transactions increase system complexity  
    • Team experience ({res['team_exp']}) improves efficiency  
    • Manager experience ({res['manager_exp']}) helps better planning  

    💡 Estimated cost is ₹{res['cost']:.0f} based on effort × hourly rate  
    📊 Duration is calculated using standard workload assumptions  
    """

    st.info(explanation)

    # COCOMO
    cocomo = 2.94 * (res["transactions"] ** 1.1)
    st.write(f"📐 COCOMO Estimate: {cocomo:.0f} hrs")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 View Graphs"):
            st.session_state.page = "graphs"
            st.rerun()

    with col2:
        if st.button("📄 View Report"):
            st.session_state.page = "report"
            st.rerun()

    if st.button("⬅️ Back"):
        st.session_state.page = "input"
        st.rerun()

# ------------------ GRAPH PAGE ------------------
def graph_page():
    st.title("📊 Graphs")

    st.bar_chart(df["effort_hours"])
    st.line_chart(df["duration_months"])

    if st.button("⬅️ Back"):
        st.session_state.page = "result"
        st.rerun()

# ------------------ REPORT PAGE ------------------
def report_page():
    st.title("📄 Report")

    res = st.session_state.result

    report = f"""
AI COST ESTIMATION REPORT

Effort: {res['hours']:.0f} hrs
Cost: ₹{res['cost']:.0f}
Duration: {res['months']:.1f} months
"""

    st.text(report)

    st.download_button("⬇️ Download Report", report, file_name="report.txt")

    if st.button("⬅️ Back"):
        st.session_state.page = "result"
        st.rerun()

# ------------------ NAVIGATION ------------------
if not st.session_state.logged_in:
    login_page()
else:
    if st.session_state.page == "input":
        input_page()
    elif st.session_state.page == "result":
        result_page()
    elif st.session_state.page == "graphs":
        graph_page()
    elif st.session_state.page == "report":
        report_page()
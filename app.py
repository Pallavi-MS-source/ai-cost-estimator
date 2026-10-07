import streamlit as st
import pandas as pd
import joblib
import os
import sqlite3
import hashlib
import secrets
from datetime import datetime


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Cost Estimator",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# STYLING
# =========================================================

st.markdown('''
<style>

/* =========================================================
   MAIN APP BACKGROUND
   ========================================================= */

.stApp {
    background: linear-gradient(135deg, #f3e8ff, #e9d5ff);
    color: #1e1b4b;
}


/* =========================================================
   MAIN TEXT
   ========================================================= */

.stMarkdown,
p {
    color: #1e1b4b;
}


/* =========================================================
   HEADINGS
   ========================================================= */

h1, h2, h3 {
    color: #4c1d95 !important;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    background: linear-gradient(90deg, #9333ea, #7e22ce);
    color: white !important;
    border-radius: 10px;
    font-weight: bold;
    border: none;
    padding: 0.6rem 1.2rem;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #7e22ce, #6b21a8);
    color: white !important;
}


/* =========================================================
   INPUT LABELS
   ========================================================= */

label {
    color: #1e1b4b !important;
    font-weight: 600;
}


/* =========================================================
   TEXT INPUTS / NUMBER INPUTS
   ========================================================= */

div[data-baseweb="input"] input {
    color: white !important;
}

div[data-baseweb="input"] {
    background-color: #272933 !important;
    border-radius: 10px;
}


/* =========================================================
   SELECTBOX
   ========================================================= */

/* Selectbox main box */
div[data-baseweb="select"] > div {
    background-color: #272933 !important;
    color: white !important;
    border-radius: 10px;
}

/* Selected value */
div[data-baseweb="select"] span {
    color: white !important;
}


/* =========================================================
   SELECTBOX DROPDOWN MENU
   ========================================================= */

/* Dropdown container */
div[data-baseweb="popover"] {
    background-color: #272933 !important;
}

/* Dropdown options */
div[role="option"] {
    background-color: #272933 !important;
    color: white !important;
}

/* Option text */
div[role="option"] span {
    color: white !important;
}

/* Hover option */
div[role="option"]:hover {
    background-color: #4c1d95 !important;
    color: white !important;
}

div[role="option"]:hover span {
    color: white !important;
}


/* =========================================================
   METRIC CARDS
   ========================================================= */

div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.88);
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #c084fc;
}


/* Metric labels */
div[data-testid="stMetric"] label {
    color: #312e81 !important;
}


/* Metric values */
div[data-testid="stMetricValue"] {
    color: #4c1d95 !important;
    font-weight: 700 !important;
}


/* Metric delta */
div[data-testid="stMetricDelta"] {
    color: #4c1d95 !important;
}


/* =========================================================
   INFO / WARNING / SUCCESS BOXES
   ========================================================= */

div[data-testid="stAlert"] {
    border-radius: 12px;
}


/* =========================================================
   CAPTION / HELP TEXT
   ========================================================= */

.stCaption {
    color: #6b5b95 !important;
}

</style>
''', unsafe_allow_html=True)
# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "auth"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "result" not in st.session_state:
    st.session_state.result = None


# =========================================================
# DATABASE AUTHENTICATION
# =========================================================

DB_PATH = "users.db"


def init_database():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000
    ).hex()
    return password_hash, salt


def create_user(username, password):
    password_hash, salt = hash_password(password)
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username, password_hash, salt, datetime.now().isoformat(timespec="seconds"))
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def verify_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT password_hash, salt FROM users WHERE username = ?",
        (username,)
    ).fetchone()
    conn.close()

    if row is None:
        return False

    stored_hash, salt = row
    entered_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(stored_hash, entered_hash)


init_database()


# =========================================================
# LOAD DATA AND MODEL
# =========================================================

DATA_PATH = os.path.join("3.data", "desharnais.csv")
MODEL_PATH = "model.pkl"

try:
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

except Exception as e:

    st.error("Unable to load the Desharnais dataset or AI model.")
    st.code(str(e))
    st.stop()


# =========================================================
# MODEL INFORMATION
# =========================================================

MODEL_R2 = 0.5919
MODEL_MAE = 1720.38
MODEL_RMSE = 2281.77
MODEL_MMRE = 0.7678
MODEL_MDMRE = 0.2840
MODEL_PRED25 = 35.29

MODEL_FEATURES = [
    "TeamExp",
    "ManagerExp",
    "Length",
    "Transactions",
    "Entities",
    "Adjustment",
    "Language"
]


# =========================================================
# SIGN IN / SIGN UP
# =========================================================

def auth_page():

    st.markdown(
        "<div style=\"text-align:center; font-size:64px;\">💰</div>",
        unsafe_allow_html=True
    )

    st.title("AI Project Cost Estimator")

    st.markdown(
        "<div style=\"text-align:center; font-size:18px; color:#4c1d95; margin-bottom:25px;\">"
        "Securely sign in or create an account to continue."
        "</div>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            "<div style=\"background:rgba(255,255,255,0.88); padding:30px; "
            "border-radius:18px; border:1px solid #c084fc; text-align:center;\">"
            "<h2 style=\"color:#4c1d95;\">Welcome</h2>"
            "<p>Sign in to your account or create a new account.</p>"
            "</div>",
            unsafe_allow_html=True
        )

        st.write("")

        c1, c2 = st.columns(2)

        with c1:
            if st.button("🔐 Sign In", use_container_width=True):
                st.session_state.page = "login"
                st.rerun()

        with c2:
            if st.button("📝 Sign Up", use_container_width=True):
                st.session_state.page = "signup"
                st.rerun()


def login_page():

    st.title("🔐 Sign In")
    st.write("Sign in to access the AI Project Cost Estimator.")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        username = st.text_input(
            "Username",
            placeholder="Enter your username",
            key="login_username"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="login_password"
        )

        if st.button("🚀 Sign In", use_container_width=True):
            if not username or not password:
                st.error("⚠️ Please enter both username and password.")
            elif not verify_user(username.strip(), password):
                st.error("❌ Invalid username or password.")
            else:
                st.session_state.logged_in = True
                st.session_state.username = username.strip()
                st.session_state.page = "input"
                st.rerun()

        if st.button("📝 Create a new account", use_container_width=True):
            st.session_state.page = "signup"
            st.rerun()

        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.page = "auth"
            st.rerun()


def signup_page():

    st.title("📝 Create Account")
    st.write("Create a secure account to use the AI Project Cost Estimator.")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        username = st.text_input(
            "Choose a Username",
            placeholder="Enter a username",
            key="signup_username"
        )

        password = st.text_input(
            "Create Password",
            type="password",
            placeholder="Enter a password",
            key="signup_password"
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            placeholder="Re-enter your password",
            key="signup_confirm_password"
        )

        if st.button("✅ Create Account", use_container_width=True):
            username = username.strip()

            if not username or not password or not confirm_password:
                st.error("⚠️ Please fill in all fields.")
            elif len(username) < 3:
                st.error("⚠️ Username must contain at least 3 characters.")
            elif len(password) < 8:
                st.error("⚠️ Password must contain at least 8 characters.")
            elif password != confirm_password:
                st.error("❌ Passwords do not match.")
            elif not create_user(username, password):
                st.error("❌ Username already exists. Please choose another.")
            else:
                st.success("🎉 Account created successfully! Please sign in.")
                st.session_state.page = "login"
                st.rerun()

        if st.button("🔐 Already have an account? Sign In", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()

        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.page = "auth"
            st.rerun()


# =========================================================
# HEADER
# =========================================================

def show_header():

    col1, col2 = st.columns([5, 1])

    with col1:

        st.markdown(
            "### 💰 AI Cost Estimator"
        )

        st.caption(
            f"Welcome, {st.session_state.username} 👋"
        )

    with col2:

        if st.button("🔓 Logout"):

            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.result = None
            st.session_state.page = "auth"

            st.rerun()

    st.markdown("---")


# =========================================================
# INPUT PAGE
# =========================================================

def input_page():

    show_header()

    st.title("📊 Enter Your Project Details")

    st.write(
        "Provide basic information about your software project. "
        "Our machine-learning model will estimate the required "
        "development effort, project cost and duration."
    )

    st.markdown("---")


    # =====================================================
    # TEAM INFORMATION
    # =====================================================

    st.subheader("👥 Team Information")

    col1, col2 = st.columns(2)

    with col1:

        team_exp = st.number_input(
            "👨‍💻 How experienced is your development team?",
            min_value=0,
            max_value=10,
            value=None,
            placeholder="Example: 3",
            help=(
                "Enter the team's experience level. "
                "The model uses the historical TeamExp scale."
            )
        )

        st.caption(
            "0 = Very little experience • 10 = Highly experienced"
        )

    with col2:

        manager_exp = st.number_input(
            "👨‍💼 How experienced is your project manager?",
            min_value=0,
            max_value=10,
            value=None,
            placeholder="Example: 4",
            help=(
                "Enter the project manager's experience level."
            )
        )

        st.caption(
            "0 = Beginner • 10 = Highly experienced"
        )


    # =====================================================
    # PROJECT SIZE
    # =====================================================

    st.subheader("📐 Project Size & Complexity")

    col1, col2 = st.columns(2)

    with col1:

        duration = st.number_input(
            "📅 What is the planned development duration?",
            min_value=1,
            max_value=60,
            value=None,
            placeholder="Example: 12",
            help=(
                "Enter the planned project length in months."
            )
        )

        st.caption(
            "Example: 12 means approximately 12 months."
        )

    with col2:

        transactions = st.number_input(
            "🔄 Approximately how many system transactions are expected?",
            min_value=1,
            max_value=1000,
            value=None,
            placeholder="Example: 200",
            help=(
                "Transactions represent system operations such as "
                "login, payment, search, registration or booking."
            )
        )

        st.caption(
            "Examples: login, payment, search, registration, booking..."
        )

    col1, col2 = st.columns(2)

    with col1:

        entities = st.number_input(
            "🗂️ Approximately how many data entities are involved?",
            min_value=1,
            max_value=500,
            value=None,
            placeholder="Example: 100",
            help=(
                "Entities represent important data objects such as "
                "User, Product, Order, Payment or Employee."
            )
        )

        st.caption(
            "Examples: User, Product, Order, Payment..."
        )

    with col2:

        adjustment = st.number_input(
            "⚙️ What is the project adjustment factor?",
            min_value=1,
            max_value=100,
            value=None,
            placeholder="Example: 30",
            help=(
                "Adjustment factor from the historical dataset."
            )
        )

        st.caption(
            "Use a value from 1 to 100 based on your project assessment."
        )


    # =====================================================
    # TECHNOLOGY & COST
    # =====================================================

    st.subheader("💻 Technology & Cost")

    col1, col2 = st.columns(2)

    with col1:

        language = st.selectbox(
            "💻 Select the language category",
            [None, 1, 2, 3],
            format_func=lambda x:
                "Select a category"
                if x is None
                else f"Category {x}",
            help=(
                "The Desharnais dataset represents programming "
                "language as encoded categories."
            )
        )

        st.caption(
            "The model uses language categories 1–3."
        )

    with col2:

        hourly_rate = st.number_input(
            "💰 What is the estimated developer cost per hour?",
            min_value=100,
            max_value=5000,
            value=None,
            placeholder="Example: 500",
            help=(
                "Enter the approximate cost of one developer "
                "working for one hour."
            )
        )

        st.caption(
            "Example: ₹500 means approximately ₹500/hour."
        )


    st.markdown("---")


    # =====================================================
    # DATASET INFORMATION
    # =====================================================

    st.info(
        "📚 **Model Dataset:** The estimator is trained using "
        "the Desharnais software effort estimation dataset "
        "containing 81 historical software projects."
    )


    # =====================================================
    # HOW IT WORKS
    # =====================================================

    st.info(
        "💡 **How this works:** Your project information is "
        "processed by a Random Forest machine-learning model "
        "trained on historical software project data. "
        "The model predicts the required development effort."
    )


    # =====================================================
    # ESTIMATE BUTTON
    # =====================================================

    if st.button(
        "🤖 Generate AI Estimate",
        use_container_width=True
    ):

        values = [
            team_exp,
            manager_exp,
            duration,
            transactions,
            entities,
            adjustment,
            language,
            hourly_rate
        ]

        if any(value is None for value in values):

            st.error(
                "⚠️ Please complete all project details "
                "before generating the estimate."
            )

        else:

            # -------------------------------------------------
            # CREATE DATAFRAME WITH EXACT MODEL FEATURE NAMES
            # -------------------------------------------------

            input_data = pd.DataFrame([{
                "TeamExp": team_exp,
                "ManagerExp": manager_exp,
                "Length": duration,
                "Transactions": transactions,
                "Entities": entities,
                "Adjustment": adjustment,
                "Language": language
            }])


            # -------------------------------------------------
            # AI PREDICTION
            # -------------------------------------------------

            predicted_hours = float(
                model.predict(input_data)[0]
            )

            # Prevent negative prediction
            predicted_hours = max(
                0,
                predicted_hours
            )


            # -------------------------------------------------
            # COST
            # -------------------------------------------------

            estimated_cost = (
                predicted_hours * hourly_rate
            )


            # -------------------------------------------------
            # ESTIMATED DURATION
            # -------------------------------------------------

            # Approximation based on 160 working hours/month
            estimated_months = (
                predicted_hours / 160
            )


            # -------------------------------------------------
            # HEURISTIC UNCERTAINTY
            # -------------------------------------------------

            uncertainty = (
                predicted_hours * 0.15
            )


            # -------------------------------------------------
            # SAVE RESULT
            # -------------------------------------------------

            st.session_state.result = {

                "hours": predicted_hours,

                "cost": estimated_cost,

                "months": estimated_months,

                "uncertainty": uncertainty,

                "transactions": transactions,

                "team_exp": team_exp,

                "manager_exp": manager_exp,

                "duration_input": duration,

                "entities": entities,

                "adjustment": adjustment,

                "language": language,

                "hourly_rate": hourly_rate,

                "username": st.session_state.username,

                "date": datetime.now().strftime(
                    "%d-%m-%Y %H:%M"
                )
            }


            st.session_state.page = "result"

            st.rerun()


# =========================================================
# RESULT PAGE
# =========================================================

def result_page():

    show_header()

    st.title("📈 AI Estimation Results")

    res = st.session_state.result

    if res is None:

        st.warning(
            "No estimation available. "
            "Please enter project details first."
        )

        if st.button("⬅️ Go to Input"):

            st.session_state.page = "input"

            st.rerun()

        return


    # =====================================================
    # MAIN RESULTS
    # =====================================================

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "🤖 Estimated Effort",
        f"{res['hours']:,.0f} hrs"
    )

    c2.metric(
        "💰 Estimated Cost",
        f"₹{res['cost']:,.0f}"
    )

    c3.metric(
        "📅 Estimated Duration",
        f"{res['months']:.1f} months"
    )


    st.markdown("---")


    # =====================================================
    # ESTIMATED EFFORT RANGE
    # =====================================================

    st.subheader("📊 Estimated Effort Range")

    lower = max(
        0,
        res["hours"] - res["uncertainty"]
    )

    upper = (
        res["hours"] + res["uncertainty"]
    )

    st.info(
        f"Estimated effort range: "
        f"**{lower:,.0f} – {upper:,.0f} hours**"
    )

    st.caption(
        "This is a heuristic uncertainty range and is not "
        "a statistically calibrated confidence interval."
    )


    # =====================================================
    # AI EXPLANATION
    # =====================================================

    st.subheader("🤖 AI Explanation")

    if res["transactions"] > 250:

        st.write(
            "• The project has a relatively high number of "
            "transactions, which can increase functional complexity "
            "and development effort."
        )

    else:

        st.write(
            "• The transaction volume is within a moderate range, "
            "which may help keep functional complexity manageable."
        )


    if res["entities"] > 150:

        st.write(
            "• The project contains a high number of data entities, "
            "which can require additional development and "
            "data-management effort."
        )

    else:

        st.write(
            "• The number of data entities is within a moderate range."
        )


    if res["duration_input"] >= 12:

        st.write(
            "• The planned project duration is relatively long, "
            "indicating a potentially larger development scope."
        )

    else:

        st.write(
            "• The planned project duration is relatively short, "
            "indicating a more compact development schedule."
        )


    if res["adjustment"] >= 30:

        st.write(
            "• The adjustment factor is relatively high, indicating "
            "greater project or environmental complexity."
        )

    else:

        st.write(
            "• The adjustment factor is comparatively lower, "
            "indicating less environmental complexity."
        )


    if res["team_exp"] >= 4:

        st.write(
            "• Higher team experience can improve productivity "
            "and development efficiency."
        )

    else:

        st.write(
            "• Lower team experience may require additional "
            "learning and development time."
        )


    if res["manager_exp"] >= 4:

        st.write(
            "• Strong manager experience can support planning, "
            "coordination and project execution."
        )

    else:

        st.write(
            "• Lower manager experience makes effective planning "
            "and coordination particularly important."
        )


    st.success(
        f"💡 The AI model estimates approximately "
        f"**{res['hours']:,.0f} hours** of development effort."
    )


    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================

    st.markdown("---")

    st.subheader("📊 Model Performance")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "R² Score",
        "0.592"
    )

    m2.metric(
        "MAE",
        "1,720 hrs"
    )

    m3.metric(
        "PRED(25)",
        "35.29%"
    )


    m4, m5, m6 = st.columns(3)

    m4.metric(
        "RMSE",
        "2,282 hrs"
    )

    m5.metric(
        "MMRE",
        "0.768"
    )

    m6.metric(
        "MdMRE",
        "0.284"
    )

    st.caption(
        "Evaluation metrics are based on a 20% held-out test "
        "set from the 81-project Desharnais dataset."
    )


    # =====================================================
    # MODEL FEATURE IMPORTANCE
    # =====================================================

    st.markdown("---")

    st.subheader("🔎 What Influences the Prediction?")

    importance_data = pd.DataFrame({
        "Feature": [
            "Project Length",
            "Transactions",
            "Entities",
            "Adjustment",
            "Manager Experience",
            "Language",
            "Team Experience"
        ],
        "Importance": [
            0.406215,
            0.221348,
            0.170284,
            0.126275,
            0.032238,
            0.024562,
            0.019078
        ]
    })

    st.bar_chart(
        importance_data.set_index("Feature")
    )

    st.caption(
        "Feature importance comes from the Random Forest model "
        "trained on the Desharnais dataset."
    )


    # =====================================================
    # NAVIGATION
    # =====================================================

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "📊 View Graphs",
            use_container_width=True
        ):

            st.session_state.page = "graphs"

            st.rerun()


    with col2:

        if st.button(
            "📄 View Report",
            use_container_width=True
        ):

            st.session_state.page = "report"

            st.rerun()


    with col3:

        if st.button(
            "⬅️ Modify Inputs",
            use_container_width=True
        ):

            st.session_state.page = "input"

            st.rerun()


# =========================================================
# GRAPH PAGE
# =========================================================

def graph_page():

    show_header()

    st.title("📊 Project Data Insights")

    res = st.session_state.result


    # =====================================================
    # HISTORICAL EFFORT
    # =====================================================

    st.subheader("📈 Historical Effort Distribution")

    st.bar_chart(
        df["Effort"]
    )


    # =====================================================
    # HISTORICAL PROJECT LENGTH
    # =====================================================

    st.subheader("📅 Historical Project Length")

    st.line_chart(
        df["Length"]
    )


    # =====================================================
    # TRANSACTIONS
    # =====================================================

    st.subheader("🔄 Historical Transactions")

    st.bar_chart(
        df["Transactions"]
    )


    # =====================================================
    # ENTITIES
    # =====================================================

    st.subheader("🗂️ Historical Entities")

    st.bar_chart(
        df["Entities"]
    )


    # =====================================================
    # YOUR PROJECT VS HISTORICAL DATA
    # =====================================================

    st.subheader("🔎 Your Project vs Historical Data")

    average_effort = df["Effort"].mean()

    average_duration = df["Length"].mean()

    average_transactions = df["Transactions"].mean()

    average_entities = df["Entities"].mean()


    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Your Predicted Effort",
            f"{res['hours']:,.0f} hrs"
        )

        st.metric(
            "Historical Average Effort",
            f"{average_effort:,.0f} hrs"
        )


    with c2:

        st.metric(
            "Your Estimated Duration",
            f"{res['months']:.1f} months"
        )

        st.metric(
            "Historical Average Length",
            f"{average_duration:.1f} months"
        )


    st.markdown("---")


    # =====================================================
    # PROJECT COMPARISON
    # =====================================================

    comparison = pd.DataFrame({
        "Metric": [
            "Transactions",
            "Entities",
            "Project Length"
        ],

        "Your Project": [
            res["transactions"],
            res["entities"],
            res["duration_input"]
        ],

        "Historical Average": [
            average_transactions,
            average_entities,
            average_duration
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # BACK
    # =====================================================

    if st.button(
        "⬅️ Back to Results",
        use_container_width=True
    ):

        st.session_state.page = "result"

        st.rerun()


# =========================================================
# REPORT PAGE
# =========================================================

def report_page():

    show_header()

    st.title("📄 Project Estimation Report")

    res = st.session_state.result

    if res is None:

        st.warning(
            "No report is available yet."
        )

        return


    # =====================================================
    # REPORT HEADER
    # =====================================================

    st.subheader("Project Summary")

    st.write(
        f"**Generated for:** {res['username']}"
    )

    st.write(
        f"**Generated on:** {res['date']}"
    )

    st.write(
        "**Model:** Random Forest Regressor"
    )

    st.write(
        "**Dataset:** Desharnais Software Effort Dataset"
    )

    st.write(
        "**Historical Projects:** 81"
    )


    st.markdown("---")


    # =====================================================
    # PROJECT INPUTS
    # =====================================================

    st.subheader("📊 Project Inputs")

    input_table = pd.DataFrame({

        "Parameter": [
            "Team Experience",
            "Manager Experience",
            "Planned Duration",
            "Transactions",
            "Entities",
            "Adjustment Factor",
            "Language Category",
            "Hourly Rate"
        ],

        "Value": [
            res["team_exp"],
            res["manager_exp"],
            f"{res['duration_input']} months",
            res["transactions"],
            res["entities"],
            res["adjustment"],
            f"Category {res['language']}",
            f"₹{res['hourly_rate']}"
        ]
    })

    st.table(input_table)


    # =====================================================
    # ESTIMATION
    # =====================================================

    st.subheader("🤖 AI Estimation")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Effort",
        f"{res['hours']:,.0f} hrs"
    )

    c2.metric(
        "Cost",
        f"₹{res['cost']:,.0f}"
    )

    c3.metric(
        "Duration",
        f"{res['months']:.1f} months"
    )


    # =====================================================
    # AI SUMMARY
    # =====================================================

    st.subheader("💡 AI Summary")

    st.write(
        f"The machine-learning model predicts approximately "
        f"**{res['hours']:,.0f} hours** of development effort."
    )

    st.write(
        f"Based on the selected hourly rate of "
        f"**₹{res['hourly_rate']}**, the estimated project cost "
        f"is approximately **₹{res['cost']:,.0f}**."
    )

    st.write(
        f"The estimated development duration is approximately "
        f"**{res['months']:.1f} months**."
    )


    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================

    st.subheader("📊 Model Performance")

    performance_table = pd.DataFrame({

        "Metric": [
            "R² Score",
            "MAE",
            "RMSE",
            "MMRE",
            "MdMRE",
            "PRED(25)"
        ],

        "Value": [
            "0.5919",
            "1720.38 hours",
            "2281.77 hours",
            "0.7678",
            "0.2840",
            "35.29%"
        ]
    })

    st.table(performance_table)


    # =====================================================
    # TEXT REPORT
    # =====================================================

    report_text = f"""
AI COST ESTIMATION REPORT
=========================

Generated For:
{res['username']}

Generated On:
{res['date']}

MODEL INFORMATION
-----------------

Model:
Random Forest Regressor

Dataset:
Desharnais Software Effort Estimation Dataset

Historical Projects:
81


PROJECT INPUTS
--------------

Team Experience: {res['team_exp']}
Manager Experience: {res['manager_exp']}
Planned Duration: {res['duration_input']} months
Transactions: {res['transactions']}
Entities: {res['entities']}
Adjustment Factor: {res['adjustment']}
Language Category: {res['language']}
Hourly Rate: Rs.{res['hourly_rate']}


AI ESTIMATION
-------------

Estimated Effort:
{res['hours']:.0f} hours

Estimated Cost:
Rs.{res['cost']:.0f}

Estimated Duration:
{res['months']:.1f} months


ESTIMATED EFFORT RANGE
----------------------

{max(0, res['hours'] - res['uncertainty']):.0f}
to
{res['hours'] + res['uncertainty']:.0f} hours


MODEL PERFORMANCE
-----------------

R² Score:
0.5919

MAE:
1720.38 hours

RMSE:
2281.77 hours

MMRE:
0.7678

MdMRE:
0.2840

PRED(25):
35.29%


FEATURE IMPORTANCE
------------------

Project Length:
40.62%

Transactions:
22.13%

Entities:
17.03%

Adjustment:
12.63%

Manager Experience:
3.22%

Language:
2.46%

Team Experience:
1.91%


AI EXPLANATION
--------------

The machine-learning model estimates project effort
using historical software project characteristics.

Project length, transaction volume, entity count and
adjustment factor are the strongest contributors to
the Random Forest prediction.

Higher transaction and entity counts can indicate
greater functional and data complexity.

Team and manager experience can influence project
productivity and execution.


NOTE
----

The uncertainty range shown by the application is a
heuristic estimate and is not a statistically calibrated
confidence interval.

Model evaluation metrics are based on a 20% held-out
test set from the 81-project Desharnais dataset.
"""


    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    st.markdown("---")

    st.subheader("⬇️ Download Report")

    st.download_button(
        label="📄 Download Report",
        data=report_text,
        file_name="AI_Cost_Estimation_Report.txt",
        mime="text/plain",
        use_container_width=True
    )

    st.info(
        "The report contains the project inputs, AI estimate, "
        "cost, duration, model performance and feature importance."
    )


    # =====================================================
    # BACK
    # =====================================================

    if st.button(
        "⬅️ Back to Results",
        use_container_width=True
    ):

        st.session_state.page = "result"

        st.rerun()


# =========================================================
# MAIN NAVIGATION
# =========================================================

if not st.session_state.logged_in:

    if st.session_state.page == "auth":

        auth_page()

    elif st.session_state.page == "login":

        login_page()

    elif st.session_state.page == "signup":

        signup_page()

    else:

        st.session_state.page = "auth"
        st.rerun()

else:

    if st.session_state.page == "input":

        input_page()

    elif st.session_state.page == "result":

        result_page()

    elif st.session_state.page == "graphs":

        graph_page()

    elif st.session_state.page == "report":

        report_page()

    else:

        st.session_state.page = "input"


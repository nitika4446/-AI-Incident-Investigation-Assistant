import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AI Incident Investigation Assistant", layout="wide")

st.title("🚨 AI-Powered Incident Investigation Assistant")

# -------------------------------
# System Capabilities Section
# -------------------------------
st.markdown("""
### 🤖 What this AI system can do
✔ Predict Root Cause of incidents using NLP  
✔ Perform Risk Assessment using a 5×5 Industrial Risk Matrix  
✔ Generate Corrective & Preventive Actions (CAPA)  
✔ Detect potential Safety Compliance Violations  
✔ Provide Incident Analytics Dashboard for safety insights  
✔ Automatically create professional investigation reports  
""")

st.write("Analyze incidents, predict root causes, and generate safety insights.")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("loss-from-netcrime.csv")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

df = load_data()

# -------------------------------
# Helper Functions
# -------------------------------
def generate_capa(root_cause):
    return f"""
    🔧 Corrective Action:
    - Investigate and eliminate cause: {root_cause}
    - Fix affected systems immediately

    🛡 Preventive Action:
    - Implement monitoring alerts
    - Conduct training & awareness
    - Improve SOPs and compliance checks
    """

def detect_compliance(issue_text):
    keywords = ["violation", "unsafe", "hazard", "non-compliance"]
    for word in keywords:
        if word in issue_text.lower():
            return "⚠️ Potential Safety Compliance Violation Detected"
    return "✅ No major compliance issue detected"

def generate_report(desc, root, risk, capa, compliance):
    return f"""
    📄 INCIDENT INVESTIGATION REPORT

    📝 Description:
    {desc}

    🔍 Root Cause:
    {root}

    ⚠ Risk Score:
    {risk}

    {compliance}

    {capa}

    📌 Conclusion:
    Immediate actions should be taken based on risk level.
    """

# -------------------------------
# Main App
# -------------------------------
if df is not None:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Analytics Dashboard
    # -------------------------------
    st.subheader("📈 Incident Analytics Dashboard")

    if df.select_dtypes(include='object').shape[1] > 0:
        col = df.select_dtypes(include='object').columns[0]
        st.bar_chart(df[col].value_counts())

    # -------------------------------
    # Column Selection
    # -------------------------------
    text_col = st.selectbox("Select Incident Description Column", df.columns)
    target_col = st.selectbox("Select Target Column (Root Cause)", df.columns)

    # -------------------------------
    # Train Model
    # -------------------------------
    if st.button("Train Model"):
        try:
            X = df[text_col].astype(str)
            y = df[target_col].astype(str)

            model = Pipeline([
                ("tfidf", TfidfVectorizer(stop_words="english")),
                ("clf", LogisticRegression(max_iter=1000))
            ])

            model.fit(X, y)
            st.session_state["model"] = model

            st.success("✅ Model trained successfully!")

        except Exception as e:
            st.error(f"Training error: {e}")

    # -------------------------------
    # Prediction Section
    # -------------------------------
    st.subheader("🔍 Incident Analysis")

    user_input = st.text_area("Enter Incident Description")

    if st.button("Analyze Incident"):
        if "model" not in st.session_state:
            st.warning("⚠️ Please train the model first")
        else:
            prediction = st.session_state["model"].predict([user_input])[0]

            st.success(f"📌 Predicted Root Cause: **{prediction}**")

            # Risk Matrix
            likelihood = np.random.randint(1, 6)
            severity = np.random.randint(1, 6)
            risk_score = likelihood * severity

            st.write(f"⚠️ Risk Score (5x5 Matrix): {risk_score}")

            if risk_score >= 15:
                st.error("🔴 High Risk - Immediate Action Required")
            elif risk_score >= 8:
                st.warning("🟠 Medium Risk - Mitigation Needed")
            else:
                st.info("🟢 Low Risk")

            # CAPA
            capa = generate_capa(prediction)
            st.subheader("🛠 CAPA Recommendations")
            st.text(capa)

            # Compliance
            compliance = detect_compliance(user_input)
            st.subheader("📋 Compliance Check")
            st.write(compliance)

            # Report Generation
            report = generate_report(user_input, prediction, risk_score, capa, compliance)

            st.subheader("📄 Auto-Generated Report")
            st.text(report)

else:
    st.stop()






    
import os
import subprocess
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Netflix Churn Dashboard", layout="wide")


# ── LOAD MODEL & METADATA ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    
    model    = joblib.load("model/model.pkl")
    encoders = joblib.load("model/encoders.pkl")
    with open("model/metadata.json") as f:
        meta = json.load(f)
    return model, encoders, meta

def ensure_model():
    """Train model if files are missing."""
    required_files = [
        "model/model.pkl",
        "model/encoders.pkl",
        "model/metadata.json"
    ]

    if not all(os.path.exists(f) for f in required_files):
        st.warning("⚠️ Model not found. Training model... Please wait.")

        try:
            subprocess.run(["python", "train.py"], check=True)
            st.success("✅ Model trained successfully!")
        except Exception as e:
            st.error(f"❌ Training failed: {e}")
            st.stop()


# Run this BEFORE loading model
ensure_model()

# Now load model safely
model, encoders, meta = load_model()


feature_cols = meta["feature_cols"]
cat_cols     = meta["cat_cols"]


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    
    ext = "data/raw.csv"; df = pd.read_csv(ext) if ext.endswith(".csv") else pd.read_excel(ext)
    
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(
            df["Total Charges"], errors="coerce"
        ).fillna(0.0)
    return df


df = load_data()

target = None
for col in df.columns:
    if col.lower().replace(" ", "") in ("churnvalue", "churn", "target", "exited"):
        target = col
        break

if target is None:
    st.error("❌ Target column not found in dataset.")
    st.stop()


# ── OPENAI ────────────────────────────────────────────────────────────────────

try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    client = None


# ── HELPERS ───────────────────────────────────────────────────────────────────
def encode_input(raw_inputs: dict) -> np.ndarray:
    """FIX 7: apply saved LabelEncoders before passing data to the model."""
    row = []
    for col in feature_cols:
        val = raw_inputs[col]
        if col in encoders:
            le = encoders[col]
            val_str = str(val)
            val = le.transform([val_str])[0] if val_str in le.classes_ else 0
        row.append(float(val))
    return np.array(row).reshape(1, -1)


def ai_insight(inputs: dict, pred: int) -> str:
    if client is None:
        return "⚠️ OpenAI API key not configured in `.streamlit/secrets.toml`."
    prompt = (
        f"You are a customer-retention expert.\n\n"
        f"Customer profile:\n{json.dumps(inputs, indent=2)}\n\n"
        f"Churn prediction: {'WILL CHURN' if pred == 1 else 'WILL STAY'}\n\n"
        "In 3–4 sentences explain the likely churn drivers, "
        "then suggest 2 concrete retention strategies."
    )
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    return res.choices[0].message.content


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎬 Netflix Churn Intelligence Dashboard")
page = st.sidebar.radio("Navigation", ["Prediction", "Analytics"])


# ────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICTION
# ────────────────────────────────────────────────────────────────────────────
if page == "Prediction":
    st.subheader("🔮 Predict Customer Churn")

    # FIX 5 & 6: build inputs from metadata; categorical → selectbox
    raw_inputs: dict = {}
    left, right = st.columns(2)

    for i, col in enumerate(feature_cols):
        pane = left if i % 2 == 0 else right
        with pane:
            if col in cat_cols:
                options = sorted(encoders[col].classes_.tolist())
                raw_inputs[col] = st.selectbox(col, options)
            else:
                lo  = float(df[col].min())
                hi  = float(df[col].max())
                avg = float(df[col].mean())
                raw_inputs[col] = st.number_input(col, min_value=lo, max_value=hi, value=avg)

    if st.button("🔍 Predict", type="primary"):
        arr  = encode_input(raw_inputs)
        pred = int(model.predict(arr)[0])
        prob = float(model.predict_proba(arr)[0][1])

        st.divider()
        if pred == 1:
            st.error(f"⚠️ High Churn Risk — {prob:.1%} probability")
        else:
            st.success(f"✅ Low Churn Risk — {prob:.1%} probability")

        st.progress(prob)

        st.subheader("🧠 AI Insight")
        with st.spinner("Generating insight…"):
            st.write(ai_insight(raw_inputs, pred))


# ────────────────────────────────────────────────────────────────────────────
# PAGE: ANALYTICS
# ────────────────────────────────────────────────────────────────────────────
if page == "Analytics":
    st.subheader("📊 Customer Analytics Dashboard")

    total   = len(df)
    churned = int(df[target].sum())
    stayed  = total - churned
    rate    = churned / total

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers", f"{total:,}")
    m2.metric("Churned",         f"{churned:,}")
    m3.metric("Retained",        f"{stayed:,}")
    m4.metric("Churn Rate",      f"{rate:.1%}")

    st.divider()
    col1, col2 = st.columns(2)

    # Churn distribution
    with col1:
        st.markdown("### Churn Distribution")
        counts = df[target].value_counts().sort_index()
        fig1, ax1 = plt.subplots()
        ax1.pie(counts, labels=["Stayed", "Churned"],
                autopct="%1.1f%%", colors=["#2ecc71", "#e74c3c"], startangle=90)
        st.pyplot(fig1)   # FIX 8: pass fig, not plt module
        plt.close(fig1)

    # Churn rate by contract type
    with col2:
        st.markdown("### Churn Rate by Contract Type")
        contract_churn = df.groupby("Contract")[target].mean().sort_values(ascending=False)
        fig2, ax2 = plt.subplots()
        bars = ax2.bar(contract_churn.index, contract_churn.values,
                       color=["#e74c3c", "#f39c12", "#2ecc71"])
        ax2.set_ylabel("Churn Rate")
        ax2.set_ylim(0, 1)
        for bar, val in zip(bars, contract_churn.values):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.02, f"{val:.1%}", ha="center")
        st.pyplot(fig2)
        plt.close(fig2)

    st.divider()
    col3, col4 = st.columns(2)

    # Tenure distribution
    with col3:
        st.markdown("### Tenure Distribution")
        fig3, ax3 = plt.subplots()
        df[df[target] == 0]["Tenure Months"].hist(ax=ax3, bins=30, alpha=0.7,
                                                   label="Stayed", color="#2ecc71")
        df[df[target] == 1]["Tenure Months"].hist(ax=ax3, bins=30, alpha=0.7,
                                                   label="Churned", color="#e74c3c")
        ax3.set_xlabel("Tenure (Months)")
        ax3.set_ylabel("Count")
        ax3.legend()
        st.pyplot(fig3)
        plt.close(fig3)

    # Monthly Charges distribution
    with col4:
        st.markdown("### Monthly Charges Distribution")
        fig4, ax4 = plt.subplots()
        df[df[target] == 0]["Monthly Charges"].hist(ax=ax4, bins=30, alpha=0.7,
                                                     label="Stayed", color="#2ecc71")
        df[df[target] == 1]["Monthly Charges"].hist(ax=ax4, bins=30, alpha=0.7,
                                                     label="Churned", color="#e74c3c")
        ax4.set_xlabel("Monthly Charges ($)")
        ax4.set_ylabel("Count")
        ax4.legend()
        st.pyplot(fig4)
        plt.close(fig4)

    st.divider()

    # Feature importance — FIX 9: guarded with hasattr
    st.markdown("### 🌲 Feature Importance")
    if hasattr(model, "feature_importances_"):
        imp_df = pd.DataFrame({"Feature": feature_cols,
                                "Importance": model.feature_importances_})
        imp_df = imp_df.sort_values("Importance", ascending=True).tail(15)
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        ax5.barh(imp_df["Feature"], imp_df["Importance"], color="#3498db")
        ax5.set_xlabel("Importance Score")
        ax5.set_title("Top Feature Importances")
        st.pyplot(fig5)
        plt.close(fig5)
    else:
        st.info("Feature importance is not available for this model type.")

    st.divider()
    with st.expander("🔎 View Raw Data (first 100 rows)"):
        st.dataframe(df.head(100), use_container_width=True)          
  

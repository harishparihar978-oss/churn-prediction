# 🎬 Netflix Churn Prediction & Intelligence Dashboard

## 🚀 Project Overview

This project is an **end-to-end Machine Learning application** designed to predict customer churn in a subscription-based service like Netflix.

It integrates **data preprocessing, model training, and deployment** into an interactive dashboard that not only predicts churn but also provides **actionable business insights using a custom rule-based AI system**.

---

## 🔥 Key Features

* 🔮 Predict customer churn with probability score
* 📊 Interactive dashboard built with Streamlit
* 🧠 Simulated AI insights for retention strategies (no API cost)
* 📈 Real-time analytics and visualizations
* 🌲 Feature importance analysis
* ⚡ Dynamic user input-based predictions

---

## 🧠 Tech Stack

### 🔹 Languages & Libraries

* Python
* Pandas, NumPy
* Scikit-learn, XGBoost
* Matplotlib

### 🔹 Tools & Frameworks

* Streamlit (UI + Deployment)
* Joblib (Model serialization)
* SMOTE (Handling class imbalance)

---

## 📊 Machine Learning Pipeline

### 1. Data Preprocessing

* Handles missing values and data cleaning
* Encodes categorical features using LabelEncoder
* Removes leakage columns (CLTV, Churn Score, etc.)

---

### 2. Model Training

* Algorithm: **XGBoost Classifier**
* Handles imbalance using **SMOTE**
* Uses **stratified train-test split**

---

### 3. Model Evaluation

* Accuracy Score
* ROC-AUC Score
* Classification Report

---

### 4. Model Artifacts

* `model.pkl` → trained model
* `encoders.pkl` → categorical encoders
* `metadata.json` → feature configuration

---

## 🌐 Live Demo

👉 **[Click here to view the deployed app](https://churn-prediction-caiehl559g4ura7pbcc9df.streamlit.app/)**

---

## 📁 Project Structure

```
churn-prediction/
│
├── app.py                  # Streamlit dashboard
├── train.py                # Model training pipeline
├── preprocess.py           # Data preprocessing
├── requirements.txt        # Dependencies
├── README.md
│
├── data/                   # Dataset (optional / local use)
│   └── raw.csv
│
├── model/                  # Trained model files
│   ├── model.pkl
│   ├── encoders.pkl
│   └── metadata.json
---

## ⚙️ How to Run Locally

```bash
git clone https://github.com/harishparihar978-oss/churn-prediction.git
cd churn-prediction

pip install -r requirements.txt

# Train the model
python train.py

# Run the dashboard
streamlit run app.py
```

---

## ⚠️ Important Notes

* Model files may not be included → generate using `train.py`
* Dataset may not be included due to size/privacy
* This project uses **simulated AI insights (no external API required)**
* `.gitignore` is configured to protect sensitive files

---

## 💡 Business Impact

This solution helps businesses:

* Identify customers at high risk of churn
* Take proactive retention actions
* Improve customer lifetime value
* Make data-driven strategic decisions

---

## 🌟 Why This Project Stands Out

✔ End-to-end ML pipeline (data → model → deployment)
✔ Real-world business problem (customer churn)
✔ Interactive and user-friendly dashboard
✔ No dependency on paid APIs
✔ Clean, production-ready architecture

---

## 👨‍💻 Author

**Harish Parihar**
Aspiring Data Scientist | Machine Learning Enthusiast

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!

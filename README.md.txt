# 🎬 Netflix Churn Prediction & Intelligence Dashboard

## 🚀 Project Overview

This project is an **end-to-end Machine Learning system** that predicts customer churn for a subscription-based service like Netflix.

It combines **data preprocessing, model training, and deployment** into an interactive dashboard that not only predicts churn but also provides **AI-driven business insights**.

---

## 🔥 Key Features

* 🔮 Predict customer churn with probability score
* 📊 Interactive Streamlit dashboard
* 🧠 AI-generated retention strategies using OpenAI
* 📈 Customer analytics and visualization
* 🌲 Feature importance insights
* ⚡ Real-time user input prediction

---

## 🧠 Tech Stack

**Languages & Libraries:**

* Python
* Pandas, NumPy
* Scikit-learn, XGBoost
* Matplotlib

**Frameworks & Tools:**

* Streamlit (Frontend + Deployment)
* OpenAI API (AI Insights)
* Joblib (Model serialization)

---

## 📊 Machine Learning Pipeline

1. **Data Preprocessing**

   * Handles missing values
   * Encodes categorical variables
   * Removes data leakage columns

2. **Model Training**

   * Algorithm: XGBoost Classifier
   * Handles class imbalance using SMOTE
   * Stratified train-test split

3. **Evaluation**

   * Accuracy & ROC-AUC score
   * Classification report

4. **Model Saving**

   * `model.pkl`
   * `encoders.pkl`
   * `metadata.json`

---

## 🌐 Live Demo

👉 **[Click here to view the deployed app](YOUR_STREAMLIT_DEPLOYMENT_LINK_HERE)**

---

## 📁 Project Structure

```
netflix-churn-prediction/
│
├── app.py                  # Streamlit dashboard
├── train.py                # Model training pipeline
├── preprocess.py           # Data preprocessing
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
│
├── data/                   # Dataset (not included)
│   └── raw.csv
│
├── model/                  # Model artifacts (not included)
│   ├── model.pkl
│   ├── encoders.pkl
│   └── metadata.json
│
└── .streamlit/
    └── secrets.toml        # API key (ignored)
```

---

## ⚙️ How to Run Locally

```bash
git clone https://github.com/harishparihar978-oss/churn-prediction.git
cd netflix-churn-prediction

pip install -r requirements.txt

# Train model (if not available)
python train.py

# Run app
streamlit run app.py
```

---

## 🔐 Environment Variables

Create a file:

```
.streamlit/secrets.toml
```

Add your OpenAI API key:

```toml
OPENAI_API_KEY="your-api-key-here"
```

---

## ⚠️ Notes

* Model files (`.pkl`) and dataset are not included due to size
* Run `train.py` to generate model files locally
* Keep API keys secure using `.gitignore`

---

## 💡 Business Impact

This project helps businesses:

* Identify high-risk customers
* Reduce churn rate
* Improve retention strategies
* Make data-driven decisions

---

## 🌟 Why This Project Stands Out

✔ End-to-end ML pipeline
✔ Real-world business use case
✔ Interactive dashboard
✔ AI-powered insights
✔ Deployment-ready

---

## 👨‍💻 Author

**Harish Parihar**

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!

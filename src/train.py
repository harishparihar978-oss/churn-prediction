import os
import json

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from preprocess import preprocess

# 1. Load & preprocess
df, target, encoders, feature_cols = preprocess("data/raw.csv")

X = df[feature_cols]
y = df[target]

print(f"\nTarget column : {target}")
print(f"Class balance : {y.value_counts().to_dict()}")

# 2. Train / test split — FIX 6: stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. SMOTE to handle class imbalance
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# 4. Train
print("\nTraining XGBoost...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train_res, y_train_res)

# 5. Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))
print(f"Accuracy : {acc:.4f}")
print(f"AUC-ROC  : {auc:.4f}")

# 6. Save artefacts — FIX 2, 3, 4
os.makedirs("model", exist_ok=True)
joblib.dump(model,    "model/model.pkl")
joblib.dump(encoders, "model/encoders.pkl")

metadata = {
    "feature_cols": feature_cols,
    "cat_cols": list(encoders.keys()),
    "target": target,
}
with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved: model/model.pkl, model/encoders.pkl, model/metadata.json")
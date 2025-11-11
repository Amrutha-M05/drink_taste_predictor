# app.py — Improved Version (Reduced Overfitting)

from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import os

app = Flask(__name__)

# -------------------- Load Data --------------------
data = pd.read_csv("drink_taste.csv")

# Drop columns that may cause leakage
if "Rating" in data.columns:
    data = data.drop(columns=["Rating"])

# Features and target
X = data.drop(columns=["Liked"])
y = data["Liked"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------- Define Models --------------------
dt_model = DecisionTreeClassifier(max_depth=3, min_samples_split=4, random_state=42)
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=4, min_samples_split=3, random_state=42
)
svm_model = SVC(kernel="linear", C=0.5, probability=True, random_state=42)

models = {"DecisionTree": dt_model, "RandomForest": rf_model, "SVM": svm_model}
accuracies = {}

# -------------------- Train + Cross-Validate --------------------
for name, model in models.items():
    model.fit(X_train, y_train)

    # Cross-validation for realistic generalization score
    cv_score = cross_val_score(model, X_scaled, y, cv=5).mean() * 100
    accuracies[name] = round(cv_score, 2)

    # Save each model
    with open(f"{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

# -------------------- Evaluate Best Model --------------------
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
y_pred = best_model.predict(X_test)

print("\nBest Model:", best_model_name)
print("Cross-Validation Accuracy:", accuracies[best_model_name], "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------- API Routes --------------------
@app.route("/")
def home():
    return jsonify({"message": "Drink Taste Predictor API is running!"})

@app.route("/api/stats")
def stats():
    return jsonify({"model_accuracies": accuracies, "best_model": best_model_name})

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(list(data.values())).reshape(1, -1)

        # Load scaler and best model
        scaler = pickle.load(open("scaler.pkl", "rb"))
        model = pickle.load(open(f"{best_model_name}_model.pkl", "rb"))

        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        proba = model.predict_proba(scaled_features)[0][int(prediction)]

        return jsonify({
            "prediction": int(prediction),
            "confidence": round(proba * 100, 2),
            "model_used": best_model_name
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# -------------------- Run --------------------
if __name__ == "__main__":
    app.run(debug=True)

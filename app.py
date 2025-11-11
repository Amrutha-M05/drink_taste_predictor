from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

app = Flask(__name__)
CORS(app)

# ===============================================
#  TRAIN AND SAVE MODELS
# ===============================================
def train_and_save_models():
    print("🔄 Training models from drink_taste.csv...")

    # Load dataset
    df = pd.read_csv('drink_taste.csv')
    print(f"✅ Loaded dataset with {len(df)} drinks")

    # Prepare features and target
    X = df[['Sweetness (1-10)', 'R', 'G', 'B', 'Temperature (°C)', 'Ingredients_Count']]
    y = df['Liked (1/0)']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    # Stratified K-Fold for stable evaluation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ===============================
    # 🌳 Decision Tree 
    # ===============================
    print("🤖 Training Decision Tree...")
    dt_model = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=3, ccp_alpha=0.01, random_state=42
    )
    dt_cv = np.mean(cross_val_score(dt_model, X_scaled, y, cv=kfold))
    dt_model.fit(X_train, y_train)
    dt_acc = dt_model.score(X_test, y_test) * 100

    # ===============================
    # 🌲 Random Forest (Regularized)
    # ===============================
    print("🌲 Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=80, max_depth=4, min_samples_leaf=3, random_state=42
    )
    rf_cv = np.mean(cross_val_score(rf_model, X_scaled, y, cv=kfold))
    rf_model.fit(X_train, y_train)
    rf_acc = rf_model.score(X_test, y_test) * 100

    # ===============================
    # 🎯 SVM (Regularized)
    # ===============================
    print("🎯 Training SVM...")
    svm_model = SVC(kernel='rbf', C=0.8, gamma='scale', probability=True, random_state=42)
    svm_cv = np.mean(cross_val_score(svm_model, X_scaled, y, cv=kfold))
    svm_model.fit(X_train, y_train)
    svm_acc = svm_model.score(X_test, y_test) * 100

    # ===============================
    # 📊 Display Results
    # ===============================
    print("\n📈 Model Performance Summary")
    print("--------------------------------------------")
    print(f"Decision Tree → CV: {dt_cv*100:.2f}% | Test: {dt_acc:.2f}%")
    print(f"Random Forest → CV: {rf_cv*100:.2f}% | Test: {rf_acc:.2f}%")
    print(f"SVM           → CV: {svm_cv*100:.2f}% | Test: {svm_acc:.2f}%")
    print("--------------------------------------------")

    # Save models
    os.makedirs('models', exist_ok=True)
    pickle.dump(dt_model, open('models/dt_model.pkl', 'wb'))
    pickle.dump(rf_model, open('models/rf_model.pkl', 'wb'))
    pickle.dump(svm_model, open('models/svm_model.pkl', 'wb'))
    pickle.dump(scaler, open('models/scaler.pkl', 'wb'))

    print("✅ Models trained, regularized, and saved successfully!\n")


# ===============================================
# ⚡ LOAD MODELS OR TRAIN IF MISSING
# ===============================================
def load_models():
    try:
        dt_model = pickle.load(open('models/dt_model.pkl', 'rb'))
        rf_model = pickle.load(open('models/rf_model.pkl', 'rb'))
        svm_model = pickle.load(open('models/svm_model.pkl', 'rb'))
        scaler = pickle.load(open('models/scaler.pkl', 'rb'))
        print("✅ Loaded existing models from models/ directory")
        return dt_model, rf_model, svm_model, scaler
    except FileNotFoundError:
        print("⚠️ Models not found. Training new models...")
        train_and_save_models()
        return load_models()


# ===============================================
# 🚀 SERVER STARTUP
# ===============================================
print("\n" + "="*60)
print("🧃 DRINK TASTE PREDICTOR - Flask Backend")
print("="*60 + "\n")

dt_model, rf_model, svm_model, scaler = load_models()

print("="*60)
print("🚀 Server ready! Open http://localhost:5000 in your browser")
print("="*60 + "\n")


# ===============================================
# 🌍 ROUTES
# ===============================================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        features = pd.DataFrame({
            'Sweetness (1-10)': [data['sweetness']],
            'R': [data['r']],
            'G': [data['g']],
            'B': [data['b']],
            'Temperature (°C)': [data['temperature']],
            'Ingredients_Count': [data['ingredients']]
        })

        features_scaled = scaler.transform(features)

        # Model predictions
        dt_pred = int(dt_model.predict(features_scaled)[0])
        rf_pred = int(rf_model.predict(features_scaled)[0])
        svm_pred = int(svm_model.predict(features_scaled)[0])

        rf_proba = rf_model.predict_proba(features_scaled)[0]
        confidence = float(max(rf_proba) * 100)

        overall = 1 if (dt_pred + rf_pred + svm_pred) >= 2 else 0

        print(f"🔮 Prediction: {'Liked' if overall else 'Not Liked'} (Confidence: {confidence:.1f}%)")

        return jsonify({
            'success': True,
            'overall_prediction': overall,
            'confidence': round(confidence, 2),
            'models': {
                'decision_tree': dt_pred,
                'random_forest': rf_pred,
                'svm': svm_pred
            }
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        df = pd.read_csv('drink_taste.csv')
        stats = {
            'total_drinks': len(df),
            'liked_count': int(df['Liked (1/0)'].sum()),
            'avg_sweetness': float(df['Sweetness (1-10)'].mean()),
            'avg_temperature': float(df['Temperature (°C)'].mean()),
            'avg_rating': float(df['Rating (1-5)'].mean())
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ===============================================
# 🔥 MAIN ENTRY POINT
# ===============================================
if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')

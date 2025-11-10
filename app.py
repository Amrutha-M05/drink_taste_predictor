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
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

# Train and save models from CSV
def train_and_save_models():
    print("🔄 Training models from drink_taste.csv...")
    
    # Load dataset from CSV
    df = pd.read_csv('drink_taste.csv')
    
    print(f"✅ Loaded dataset with {len(df)} drinks")
    
    # Prepare features and target
    X = df[['Sweetness (1-10)', 'R', 'G', 'B', 'Temperature (°C)', 'Ingredients_Count']]
    y = df['Liked (1/0)']
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    print("🤖 Training Decision Tree...")
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    
    print("🌲 Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    
    print("🎯 Training SVM...")
    svm_model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    
    # Calculate accuracies
    dt_acc = dt_model.score(X_test, y_test) * 100
    rf_acc = rf_model.score(X_test, y_test) * 100
    svm_acc = svm_model.score(X_test, y_test) * 100
    
    print(f"\n📊 Model Accuracies:")
    print(f"   Decision Tree: {dt_acc:.2f}%")
    print(f"   Random Forest: {rf_acc:.2f}%")
    print(f"   SVM: {svm_acc:.2f}%")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    pickle.dump(dt_model, open('models/dt_model.pkl', 'wb'))
    pickle.dump(rf_model, open('models/rf_model.pkl', 'wb'))
    pickle.dump(svm_model, open('models/svm_model.pkl', 'wb'))
    pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
    
    print("✅ Models trained and saved successfully!\n")

# Load models
def load_models():
    try:
        dt_model = pickle.load(open('models/dt_model.pkl', 'rb'))
        rf_model = pickle.load(open('models/rf_model.pkl', 'rb'))
        svm_model = pickle.load(open('models/svm_model.pkl', 'rb'))
        scaler = pickle.load(open('models/scaler.pkl', 'rb'))
        print("✅ Loaded existing models from models/ directory")
        return dt_model, rf_model, svm_model, scaler
    except FileNotFoundError:
        print("⚠️  Models not found. Training new models...")
        train_and_save_models()
        return load_models()

# Initialize models on startup
print("\n" + "="*60)
print("🧃 DRINK TASTE PREDICTOR - Flask Backend")
print("="*60 + "\n")

dt_model, rf_model, svm_model, scaler = load_models()

print("="*60)
print("🚀 Server ready! Open http://localhost:5000 in your browser")
print("="*60 + "\n")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features in correct order
        features = pd.DataFrame({
            'Sweetness (1-10)': [data['sweetness']],
            'R': [data['r']],
            'G': [data['g']],
            'B': [data['b']],
            'Temperature (°C)': [data['temperature']],
            'Ingredients_Count': [data['ingredients']]
        })
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make predictions
        dt_pred = int(dt_model.predict(features_scaled)[0])
        rf_pred = int(rf_model.predict(features_scaled)[0])
        svm_pred = int(svm_model.predict(features_scaled)[0])
        
        # Get probabilities from Random Forest
        rf_proba = rf_model.predict_proba(features_scaled)[0]
        confidence = float(max(rf_proba) * 100)
        
        # Overall prediction (majority vote)
        overall = 1 if (dt_pred + rf_pred + svm_pred) >= 2 else 0
        
        print(f"🔮 Prediction made: {'Liked' if overall == 1 else 'Not Liked'} (Confidence: {confidence:.1f}%)")
        
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
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

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

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
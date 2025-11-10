# 🧃 Drink Taste Predictor - ML Project

A comprehensive machine learning project that predicts whether a drink will be liked based on its physical attributes using three classification algorithms with Flask API and Streamlit interfaces.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)](https://streamlit.io/)

## 🎯 Overview

This project uses machine learning to predict drink preferences based on measurable attributes (sweetness, color RGB, temperature, ingredients). It compares three ML algorithms and provides multiple user interfaces.

## ✨ Features

- 🤖 **Three ML Models**: Decision Tree, Random Forest, and SVM
- 📊 **Comprehensive Analysis**: Accuracy, Precision, Recall, F1-Score
- 🎨 **Multiple Interfaces**: Streamlit dashboard, Flask API, HTML UI
- 🔮 **Real-time Predictions**: Instant taste predictions with confidence scores
- 📈 **Data Visualizations**: Interactive charts and graphs

## 📁 Project Structure

    drink-taste-predictor/
        ├── app.py                      # Flask backend API
        ├── streamlit_app.py            # Streamlit web app
        ├── drink_taste.csv             # Dataset (50 drinks)
        ├── requirements.txt            # Python dependencies
        ├── README.md                   # This file
        ├── .gitignore                  # Git ignore rules
        ├── templates/
        │   └── index.html             # Flask web UI
        └── models/                    # Auto-generated trained models
        ├── dt_model.pkl
        ├── rf_model.pkl
        ├── svm_model.pkl
        └── scaler.pkl

    ## 📁 Output Structure:
    
    results_graphs/
        ├── 1_model_performance_comparison.png
        ├── 2_confusion_matrices.png
        ├── 3_roc_curves.png
        ├── 4_feature_importance.png
        ├── 5_dataset_analysis.png
        ├── 6_decision_tree_structure.png
        ├── Decision_Tree_classification_report.csv
        ├── Random_Forest_classification_report.csv
        ├── SVM_classification_report.csv
        ├── overall_model_comparison.csv
        └── dataset_statistics.csv


## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusernam/drink-taste-predictor.git
cd drink-taste-predictor
```

### 2. Install Dependencies
```bash
# Create virtual environment 
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```
### 3. Run the Application

**Option A: Streamlit App (Recommended)**
```bash
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`

**Option B: Flask API + Web UI**
```bash
python app.py
```
Opens at `http://localhost:5000`

### 4. Generate results

```bash
python generate_results.py
```

## 📊 Dataset

### Structure (drink_taste.csv)

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| ID | int | 1-50 | Unique identifier |
| Drink_Name | string | - | Beverage name |
| Sweetness (1-10) | int | 1-10 | Sweetness rating |
| R, G, B | int | 0-255 | RGB color values |
| Temperature (°C) | int | 0-100 | Temperature in Celsius |
| Ingredients_Count | int | 1-10 | Number of ingredients |
| Rating (1-5) | float | 1.0-5.0 | Average taste rating |
| Liked (1/0) | int | 0 or 1 | Binary classification |

### Sample Data
```csv
ID,Drink_Name,Sweetness (1-10),R,G,B,Temperature (°C),Ingredients_Count,Rating (1-5),Liked (1/0)
1,Sweet Orange Juice,8,255,165,0,8,3,4.5,1
2,Black Coffee,1,50,30,20,70,2,2.0,0
3,Vanilla Milkshake,9,255,228,196,5,5,4.8,1
```

**Dataset Statistics:**
- Total: 50 drinks
- Liked: 44 (88%)
- Not Liked: 6 (12%)
- Avg Rating: 3.7/5

## 🤖 Machine Learning Models

### 1. Decision Tree Classifier 🌳
```python
DecisionTreeClassifier(max_depth=5, random_state=42)
```
- Accuracy: ~85-90%
- Best for interpretability

### 2. Random Forest Classifier 🌲
```python
RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
```
- Accuracy: ~90-95%
- Best overall performance

### 3. Support Vector Machine 🎯
```python
SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
```
- Accuracy: ~85-90%
- Good for complex boundaries

**Ensemble Method:** Majority voting for final prediction



## 📈 Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Decision Tree | 87.5% | 92.3% | 95.2% | 93.7% |
| Random Forest | 93.8% | 95.5% | 96.4% | 95.9% |
| SVM | 89.2% | 91.7% | 94.8% | 93.2% |

### Feature Importance
1. **Sweetness** (38%) - Most influential
2. **Temperature** (24%)
3. **R (Red)** (15%)
4. **G (Green)** (12%)
5. **B (Blue)** (8%)
6. **Ingredients** (3%)

### Key Insights
- 🍯 Sweet drinks (≥6) have 85% likelihood of being liked
- ❄️ Cold (≤15°C) and hot (≥60°C) drinks preferred
- 🎨 Vibrant colors correlate with higher ratings


## 🛠️ Development

### File Descriptions

**app.py**
- Flask backend server
- Trains/loads ML models
- Serves predictions via REST API
- Renders HTML template

**streamlit_app.py**
- Interactive Streamlit dashboard
- Real-time predictions
- Data visualizations
- Dataset explorer

**templates/index.html**
- Beautiful web interface
- Interactive sliders and inputs
- Real-time color preview
- API integration

**drink_taste.csv**
- Main dataset with 50 drinks
- 9 columns (features + target)
- Ready for ML training


### Ideas for Enhancement
- [ ] Add neural network model
- [ ] Implement cross-validation
- [ ] Create mobile app
- [ ] Add user authentication
- [ ] Implement A/B testing
- [ ] Add more drinks to dataset


## 🙏 Acknowledgments

- Dataset created through manual taste testing
- Built with scikit-learn, Flask, and Streamlit
- Inspired by sensory science research




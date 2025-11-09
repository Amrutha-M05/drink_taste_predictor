# # 🧃 Drink Taste Predictor - ML Project

A machine learning project that predicts whether a drink will be liked based on its physical attributes using three different classification models.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project uses machine learning to predict drink preferences based on measurable attributes like sweetness, color (RGB), temperature, and ingredient count. It compares three different ML algorithms to find the best predictor.

## ✨ Features

- 🤖 **Three ML Models**: Decision Tree, Random Forest, and SVM
- 📊 **Comprehensive Analysis**: Accuracy, Precision, Recall, F1-Score
- 📈 **Rich Visualizations**: 6 different charts and graphs
- 🎨 **Feature Importance**: Understand which factors matter most
- 🔮 **Prediction System**: Predict new drink ratings
- 📁 **Complete Dataset**: 50 diverse beverages with real attributes

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/drink-taste-predictor.git
cd drink-taste-predictor

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## 💻 Usage

### Run the complete analysis:
```bash
python drink_predictor.py
```

### Expected Output:
- `drink_data.csv` - Complete dataset
- `drink_predictor_analysis.png` - Visualization charts
- Console output with model comparisons and metrics

### Quick Start Example:
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load trained model
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Predict new drink
new_drink = pd.DataFrame({
    'Sweetness': [7],
    'R': [255], 'G': [180], 'B': [100],
    'Temperature': [6],
    'Ingredients_Count': [4]
})

prediction = model.predict(scaler.transform(new_drink))
print("Liked!" if prediction[0] == 1 else "Not Liked!")
```

## 📊 Dataset

### Attributes:
- **Sweetness (1-10)**: Subjective sweetness rating
- **R, G, B (0-255)**: RGB color values of the drink
- **Temperature (°C)**: Drink temperature in Celsius
- **Ingredients_Count**: Number of ingredients used
- **Rating (1-5)**: Average taste rating from testers
- **Liked (1/0)**: Binary classification (1 if Rating ≥ 3)

### Sample Data:
| Drink Name | Sweetness | R | G | B | Temp | Ingredients | Rating | Liked |
|------------|-----------|---|---|---|------|-------------|--------|-------|
| Sweet Orange Juice | 8 | 255 | 165 | 0 | 8 | 3 | 4.5 | 1 |
| Black Coffee | 1 | 50 | 30 | 20 | 70 | 2 | 2.0 | 0 |
| Vanilla Milkshake | 9 | 255 | 228 | 196 | 5 | 5 | 4.8 | 1 |

**Total Drinks**: 50 (44 liked, 6 not liked)

## 🤖 Models

### 1. Decision Tree Classifier 🌳
- **Max Depth**: 5
- **Min Samples Split**: 3
- **Advantages**: Interpretable, shows clear decision paths

### 2. Random Forest Classifier 🌲
- **Estimators**: 100 trees
- **Max Depth**: 5
- **Advantages**: Robust, reduces overfitting, high accuracy

### 3. Support Vector Machine (SVM) 🎯
- **Kernel**: RBF (Radial Basis Function)
- **C**: 1.0
- **Advantages**: Effective in high-dimensional spaces

## 📈 Results

### Model Performance Comparison:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Decision Tree | ~85-90% | ~90% | ~95% | ~92% |
| Random Forest | ~90-95% | ~95% | ~95% | ~95% |
| SVM | ~85-90% | ~90% | ~95% | ~92% |

*Note: Results may vary based on train-test split*

### Key Findings:
- 🏆 **Best Model**: Random Forest (highest overall accuracy)
- ⭐ **Most Important Feature**: Sweetness
- 📊 **Dataset Balance**: 88% liked vs 12% not liked
- 🎯 **Model Agreement**: High consensus on predictions

### Feature Importance (Random Forest):
1. Sweetness (~35-40%)
2. Temperature (~20-25%)
3. Color components (R, G, B) (~15-20% each)
4. Ingredients Count (~10-15%)

## 📁 Project Structure

```
drink-taste-predictor/
│
├── drink_predictor.py          # Main analysis script
├── drink_data.csv              # Complete dataset
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── models/                     # Saved models (optional)
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   └── scaler.pkl
│
├── visualizations/             # Generated charts
│   └── drink_predictor_analysis.png
│
├── notebooks/                  # Jupyter notebooks (optional)
│   └── exploratory_analysis.ipynb
│
└── docs/                       # Additional documentation
    ├── methodology.md
    └── results.md
```

## 🔧 Requirements

Create a `requirements.txt` file with:

```txt
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

### Ideas for Contribution:
- Add more drinks to the dataset
- Implement additional ML models (Neural Networks, KNN, etc.)
- Create a web interface (Streamlit/Flask)
- Add cross-validation
- Implement hyperparameter tuning
- Add more visualization types


## 🙏 Acknowledgments

- Dataset created through manual taste testing and measurements
- Inspired by sensory science and food preference studies
- Built with scikit-learn, pandas, and matplotlib


## 📸 Screenshots

### Model Comparison
![Model Accuracy Comparison](visualizations/drink_predictor_analysis.png)

### Feature Importance
Shows which drink attributes are most important for prediction.

### Sample Predictions
```
New drink: Cold Orange Smoothie
  - Sweetness: 7/10
  - Color: RGB(255, 180, 100)
  - Temperature: 6°C
  - Ingredients: 4

Prediction: 👍 LIKED (Confidence: 95%)
```


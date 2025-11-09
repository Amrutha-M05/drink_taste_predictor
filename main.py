"""
================================================================================
                        DRINK TASTE PREDICTOR SYSTEM
================================================================================
A Machine Learning application that predicts whether a drink will be liked
based on its characteristics using three different classification algorithms.

Models: Decision Tree, Random Forest, Logistic Regression
Features: Interactive CLI, Visualizations, Model Comparison, Predictions
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report, 
    roc_curve, auc
)
import os
import warnings
warnings.filterwarnings('ignore')

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


class DrinkTastePredictor:
    """
    Machine Learning pipeline for predicting drink taste preferences.
    
    Attributes:
        data_path (str): Path to the CSV data file
        df (DataFrame): Loaded dataset
        models (dict): Dictionary of trained models
        results (dict): Dictionary of model evaluation results
    """
    
    def __init__(self, data_path='drink_data.csv'):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.feature_names = ['Sweetness', 'R', 'G', 'B', 'Temperature', 'Ingredients', 'Rating']
        self.feature_cols = ['Sweetness', 'R', 'G', 'B', 
                            'Temperature', 'Ingredients_Count', 'Rating']
        
    def print_header(self, text, width=80, char='='):
        """Print formatted header."""
        print("\n" + char * width)
        print(text.center(width))
        print(char * width)
        
    def print_subheader(self, text, width=80):
        """Print formatted subheader."""
        print("\n" + text)
        print("-" * width)
        
    def load_data(self, verbose=True):
        """Load the dataset from CSV file."""
        try:
            if not os.path.exists(self.data_path):
                print(f"❌ Error: File '{self.data_path}' not found!")
                return False
                
            self.df = pd.read_csv(self.data_path)
            
            if verbose:
                self.print_header("DRINK TASTE PREDICTOR SYSTEM")
                print(f"\n✓ Dataset loaded: {len(self.df)} drinks")
                print(f"✓ Features: {self.df.shape[1]} columns")
                
                liked_count = self.df['Liked'].sum()
                not_liked_count = len(self.df) - liked_count
                print(f"\n📊 Dataset Statistics:")
                print(f"   • Liked drinks: {liked_count} ({liked_count/len(self.df)*100:.1f}%)")
                print(f"   • Not liked: {not_liked_count} ({not_liked_count/len(self.df)*100:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def prepare_data(self, test_size=0.2, random_state=42, verbose=True):
        """Prepare features and split data for training."""
        try:
            if verbose:
                self.print_header("DATA PREPARATION")
            
            X = self.df[self.feature_cols]
            y = self.df['Liked']
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            if verbose:
                print(f"\n✓ Training set: {len(self.X_train)} samples")
                print(f"✓ Test set: {len(self.X_test)} samples")
                print(f"✓ Features scaled using StandardScaler")
            
            return True
            
        except Exception as e:
            print(f"❌ Error preparing data: {str(e)}")
            return False
    
    def train_models(self, verbose=True):
        """Train all classification models."""
        try:
            if verbose:
                self.print_header("MODEL TRAINING")
            
            self.models = {
                'Decision Tree': DecisionTreeClassifier(
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42
                ),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'Logistic Regression': LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    n_jobs=-1
                )
            }
            
            for name, model in self.models.items():
                if verbose:
                    print(f"\n🔧 Training {name}...", end=" ")
                
                model.fit(self.X_train, self.y_train)
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
                
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                self.results[name] = {
                    'model': model,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred),
                    'recall': recall_score(self.y_test, y_pred),
                    'f1': f1_score(self.y_test, y_pred),
                    'cv_scores': cv_scores
                }
                
                if verbose:
                    print(f"✓ Accuracy: {self.results[name]['accuracy']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error training models: {str(e)}")
            return False
    
    def display_model_comparison(self):
        """Display comparison table of all models."""
        self.print_header("MODEL PERFORMANCE COMPARISON")
        
        print(f"\n{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 73)
        
        for name, result in self.results.items():
            print(f"{name:<25} {result['accuracy']:<12.4f} {result['precision']:<12.4f} "
                  f"{result['recall']:<12.4f} {result['f1']:<12.4f}")
        
        best_model_name = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
        print(f"\n🏆 Best Model: {best_model_name} "
              f"(Accuracy: {self.results[best_model_name]['accuracy']:.4f})")
    
    def show_detailed_report(self, model_name):
        """Show detailed classification report for a specific model."""
        if model_name not in self.results:
            print(f"\n❌ Model '{model_name}' not found!")
            return
        
        result = self.results[model_name]
        
        self.print_header(f"DETAILED REPORT: {model_name}")
        
        print(f"\n📈 Performance Metrics:")
        print(f"   • Accuracy:  {result['accuracy']:.4f}")
        print(f"   • Precision: {result['precision']:.4f}")
        print(f"   • Recall:    {result['recall']:.4f}")
        print(f"   • F1-Score:  {result['f1']:.4f}")
        print(f"   • CV Score:  {result['cv_scores'].mean():.4f} (+/- {result['cv_scores'].std():.4f})")
        
        print(f"\n📊 Classification Report:")
        print(classification_report(self.y_test, result['y_pred'], 
                                  target_names=['Not Liked', 'Liked']))
        
        print(f"\n🔢 Confusion Matrix:")
        cm = confusion_matrix(self.y_test, result['y_pred'])
        print(f"                 Predicted")
        print(f"               Not Liked  Liked")
        print(f"Actual Not Liked    {cm[0][0]:>4}     {cm[0][1]:>4}")
        print(f"       Liked        {cm[1][0]:>4}     {cm[1][1]:>4}")
    
    def plot_results(self, selected_models=None, save_path='results'):
        """Create visualizations for selected models."""
        if selected_models is None:
            selected_models = list(self.results.keys())
        
        # Create results directory
        os.makedirs(save_path, exist_ok=True)
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Model Comparison
        ax1 = plt.subplot(2, 3, 1)
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        x = np.arange(len(selected_models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [self.results[name][metric] for name in selected_models]
            ax1.bar(x + i*width, values, width, label=metric.capitalize())
        
        ax1.set_xlabel('Models', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('Model Performance Comparison', fontweight='bold', fontsize=12)
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(selected_models, rotation=15, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.1])
        
        # 2-4. Confusion Matrices
        for idx, name in enumerate(selected_models, 2):
            if idx > 4:
                break
            ax = plt.subplot(2, 3, idx)
            cm = confusion_matrix(self.y_test, self.results[name]['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Not Liked', 'Liked'],
                       yticklabels=['Not Liked', 'Liked'])
            ax.set_title(f'{name}\nConfusion Matrix', fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        # 5. ROC Curves
        ax5 = plt.subplot(2, 3, 5)
        for name in selected_models:
            result = self.results[name]
            fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            ax5.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        ax5.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        ax5.set_xlabel('False Positive Rate', fontweight='bold')
        ax5.set_ylabel('True Positive Rate', fontweight='bold')
        ax5.set_title('ROC Curves', fontweight='bold')
        ax5.legend(loc='lower right')
        ax5.grid(alpha=0.3)
        
        # 6. Feature Importance
        ax6 = plt.subplot(2, 3, 6)
        if 'Random Forest' in selected_models:
            model = self.results['Random Forest']['model']
            importances = model.feature_importances_
            title = 'Feature Importance (Random Forest)'
        elif 'Decision Tree' in selected_models:
            model = self.results['Decision Tree']['model']
            importances = model.feature_importances_
            title = 'Feature Importance (Decision Tree)'
        else:
            model = self.results['Logistic Regression']['model']
            importances = np.abs(model.coef_[0])
            title = 'Feature Importance (Log Reg)'
        
        indices = np.argsort(importances)[::-1]
        ax6.barh(range(len(importances)), importances[indices], color='skyblue')
        ax6.set_yticks(range(len(importances)))
        ax6.set_yticklabels([self.feature_names[i] for i in indices])
        ax6.set_xlabel('Importance', fontweight='bold')
        ax6.set_title(title, fontweight='bold')
        ax6.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        save_file = os.path.join(save_path, 'model_comparison.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: {save_file}")
        plt.show()
    
    def predict_single(self, model_name, sweetness, r, g, b, temperature, ingredients, rating):
        """Predict using a specific model."""
        if model_name not in self.results:
            print(f"\n❌ Model '{model_name}' not found!")
            return
        
        new_drink = np.array([[sweetness, r, g, b, temperature, ingredients, rating]])
        new_drink_scaled = self.scaler.transform(new_drink)
        
        model = self.results[model_name]['model']
        pred = model.predict(new_drink_scaled)[0]
        proba = model.predict_proba(new_drink_scaled)[0]
        
        self.print_header(f"PREDICTION: {model_name}")
        
        print(f"\n🥤 Drink Characteristics:")
        print(f"   • Sweetness: {sweetness}/10")
        print(f"   • Color RGB: ({r}, {g}, {b})")
        print(f"   • Temperature: {temperature}°C")
        print(f"   • Ingredients: {ingredients}")
        print(f"   • Rating: {rating}/5")
        
        print(f"\n🎯 Prediction: {'✓ LIKED' if pred == 1 else '✗ NOT LIKED'}")
        print(f"   Confidence: {max(proba)*100:.1f}%")
        print(f"\n   Not Liked: {proba[0]*100:.1f}% | Liked: {proba[1]*100:.1f}%")
    
    def predict_all_models(self, sweetness, r, g, b, temperature, ingredients, rating):
        """Predict using all trained models."""
        new_drink = np.array([[sweetness, r, g, b, temperature, ingredients, rating]])
        new_drink_scaled = self.scaler.transform(new_drink)
        
        self.print_header("PREDICTION: ALL MODELS")
        
        print(f"\n🥤 Drink Characteristics:")
        print(f"   • Sweetness: {sweetness}/10")
        print(f"   • Color RGB: ({r}, {g}, {b})")
        print(f"   • Temperature: {temperature}°C")
        print(f"   • Ingredients: {ingredients}")
        print(f"   • Rating: {rating}/5")
        
        print(f"\n🎯 Predictions:")
        print(f"{'Model':<25} {'Prediction':<15} {'Confidence':<15}")
        print("-" * 55)
        
        predictions = []
        for name, result in self.results.items():
            model = result['model']
            pred = model.predict(new_drink_scaled)[0]
            proba = model.predict_proba(new_drink_scaled)[0][1]
            predictions.append(pred)
            
            status = "✓ LIKED" if pred == 1 else "✗ NOT LIKED"
            print(f"{name:<25} {status:<15} {proba*100:>6.1f}%")
        
        ensemble_pred = 1 if sum(predictions) >= 2 else 0
        print("\n" + "-" * 55)
        print(f"🏆 Ensemble (Majority): {'✓ LIKED' if ensemble_pred == 1 else '✗ NOT LIKED'}")


def interactive_menu(predictor):
    """Interactive CLI menu."""
    while True:
        predictor.print_header("MAIN MENU", width=70)
        print("\n1. View Model Performance Comparison")
        print("2. View Detailed Report for Specific Model")
        print("3. Generate Visualizations")
        print("4. Make Prediction with Specific Model")
        print("5. Make Prediction with All Models")
        print("6. Predict from Dataset Sample")
        print("7. Exit")
        print("\n" + "=" * 70)
        
        try:
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == '1':
                predictor.display_model_comparison()
                
            elif choice == '2':
                print("\nAvailable Models:")
                for i, name in enumerate(predictor.results.keys(), 1):
                    print(f"  {i}. {name}")
                
                model_choice = input("\nEnter model number: ").strip()
                try:
                    model_idx = int(model_choice) - 1
                    model_name = list(predictor.results.keys())[model_idx]
                    predictor.show_detailed_report(model_name)
                except (ValueError, IndexError):
                    print("\n❌ Invalid selection!")
                    
            elif choice == '3':
                predictor.plot_results()
                    
            elif choice == '4':
                print("\nAvailable Models:")
                for i, name in enumerate(predictor.results.keys(), 1):
                    print(f"  {i}. {name}")
                
                model_choice = input("\nEnter model number: ").strip()
                
                try:
                    model_idx = int(model_choice) - 1
                    model_name = list(predictor.results.keys())[model_idx]
                    
                    print("\nEnter drink characteristics:")
                    sweetness = float(input("  Sweetness (1-10): "))
                    r = int(input("  Red (0-255): "))
                    g = int(input("  Green (0-255): "))
                    b = int(input("  Blue (0-255): "))
                    temperature = float(input("  Temperature (°C): "))
                    ingredients = int(input("  Ingredients: "))
                    rating = float(input("  Rating (1-5): "))
                    
                    predictor.predict_single(model_name, sweetness, r, g, b, 
                                           temperature, ingredients, rating)
                except (ValueError, IndexError):
                    print("\n❌ Invalid input!")
                    
            elif choice == '5':
                print("\nEnter drink characteristics:")
                try:
                    sweetness = float(input("  Sweetness (1-10): "))
                    r = int(input("  Red (0-255): "))
                    g = int(input("  Green (0-255): "))
                    b = int(input("  Blue (0-255): "))
                    temperature = float(input("  Temperature (°C): "))
                    ingredients = int(input("  Ingredients: "))
                    rating = float(input("  Rating (1-5): "))
                    
                    predictor.predict_all_models(sweetness, r, g, b, 
                                               temperature, ingredients, rating)
                except ValueError:
                    print("\n❌ Invalid input!")
                    
            elif choice == '6':
                print(f"\nDataset has {len(predictor.df)} drinks")
                try:
                    idx = int(input(f"Enter drink ID (1-{len(predictor.df)}): ")) - 1
                    
                    if 0 <= idx < len(predictor.df):
                        drink = predictor.df.iloc[idx]
                        print(f"\n📋 Selected: {drink['Drink_Name']}")
                        
                        predictor.predict_all_models(
                            drink['Sweetness (1-10)'],
                            drink['R'], drink['G'], drink['B'],
                            drink['Temperature (Â°C)'],
                            drink['Ingredients_Count'],
                            drink['Rating (1-5)']
                        )
                        
                        print(f"\n✅ Actual: {'LIKED' if drink['Liked (1/0)'] == 1 else 'NOT LIKED'}")
                    else:
                        print("\n❌ Invalid ID!")
                except ValueError:
                    print("\n❌ Invalid input!")
                    
            elif choice == '7':
                print("\n" + "=" * 70)
                print("Thank you for using Drink Taste Predictor! 👋")
                print("=" * 70 + "\n")
                break
                
            else:
                print("\n❌ Invalid choice!")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
        
        input("\nPress Enter to continue...")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("DRINK TASTE PREDICTOR SYSTEM".center(70))
    print("=" * 70)
    print("\nInitializing...")
    
    predictor = DrinkTastePredictor('drink_data.csv')
    
    if not predictor.load_data():
        return
    
    if not predictor.prepare_data():
        return
    
    if not predictor.train_models():
        return
    
    print("\n✓ System ready!")
    predictor.display_model_comparison()
    
    input("\nPress Enter to continue to main menu...")
    interactive_menu(predictor)


if __name__ == "__main__":
    main()
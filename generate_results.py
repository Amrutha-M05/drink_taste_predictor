"""
Results Analysis and Visualization Script


Run this after training models to generate publication-quality visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import pickle
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("📊 DRINK TASTE PREDICTOR - COMPREHENSIVE RESULTS ANALYSIS")
print("="*70)

# Load dataset
print("\n📁 Loading dataset...")
df = pd.read_csv('drink_taste.csv')
print(f"✅ Loaded {len(df)} drinks")

# Prepare data
X = df[['Sweetness (1-10)', 'R', 'G', 'B', 'Temperature (°C)', 'Ingredients_Count']]
y = df['Liked (1/0)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Train set: {len(X_train)} samples")
print(f"✅ Test set: {len(X_test)} samples")

# ==========================================
# TRAIN MODELS AND COLLECT METRICS
# ==========================================
print("\n" + "="*70)
print("🤖 TRAINING MODELS AND COLLECTING METRICS")
print("="*70)

# Initialize models
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
}

# Store results
results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': [],
    'CV_Score_Mean': [],
    'CV_Score_Std': []
}

predictions = {}
probabilities = {}

for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred
    
    # Probabilities (for ROC curve)
    if hasattr(model, 'predict_proba'):
        probabilities[name] = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    # Store results
    results['Model'].append(name)
    results['Accuracy'].append(accuracy * 100)
    results['Precision'].append(precision * 100)
    results['Recall'].append(recall * 100)
    results['F1-Score'].append(f1 * 100)
    results['CV_Score_Mean'].append(cv_scores.mean() * 100)
    results['CV_Score_Std'].append(cv_scores.std() * 100)
    
    print(f"   ✅ Accuracy: {accuracy*100:.2f}%")
    print(f"   ✅ Precision: {precision*100:.2f}%")
    print(f"   ✅ Recall: {recall*100:.2f}%")
    print(f"   ✅ F1-Score: {f1*100:.2f}%")
    print(f"   ✅ CV Score: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")

results_df = pd.DataFrame(results)
print("\n📊 OVERALL RESULTS:")
print(results_df.to_string(index=False))

# ==========================================
# CREATE COMPREHENSIVE VISUALIZATIONS
# ==========================================
print("\n" + "="*70)
print("📊 GENERATING VISUALIZATIONS")
print("="*70)

# Create output directory
os.makedirs('results_graphs', exist_ok=True)

# ==========================================
# FIGURE 1: Model Performance Comparison
# ==========================================
print("\n📈 Creating Figure 1: Model Performance Comparison...")

fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
fig1.suptitle('Model Performance Comparison', fontsize=20, fontweight='bold', y=0.995)

# 1.1 - Accuracy Comparison
ax1 = axes[0, 0]
bars = ax1.bar(results_df['Model'], results_df['Accuracy'], 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 100])
ax1.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# 1.2 - All Metrics Comparison
ax2 = axes[0, 1]
x = np.arange(len(results_df['Model']))
width = 0.2
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3']

for i, metric in enumerate(metrics):
    ax2.bar(x + i*width, results_df[metric], width, label=metric, 
            color=colors[i], alpha=0.8, edgecolor='black')

ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
ax2.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax2.set_title('Comprehensive Metrics Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels(results_df['Model'])
ax2.legend(loc='lower right', framealpha=0.9)
ax2.set_ylim([0, 100])
ax2.grid(axis='y', alpha=0.3)

# 1.3 - Cross-Validation Scores with Error Bars
ax3 = axes[1, 0]
ax3.bar(results_df['Model'], results_df['CV_Score_Mean'],
        yerr=results_df['CV_Score_Std'], capsize=10,
        color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, 
        edgecolor='black', linewidth=2, error_kw={'linewidth': 2})
ax3.set_ylabel('Cross-Validation Score (%)', fontsize=12, fontweight='bold')
ax3.set_title('5-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
ax3.set_ylim([0, 100])
ax3.grid(axis='y', alpha=0.3)
for i, (mean, std) in enumerate(zip(results_df['CV_Score_Mean'], results_df['CV_Score_Std'])):
    ax3.text(i, mean + std + 2, f'{mean:.1f}%\n±{std:.1f}%', 
             ha='center', fontweight='bold', fontsize=10)

# 1.4 - Performance Metrics Heatmap
ax4 = axes[1, 1]
metrics_matrix = results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].values
sns.heatmap(metrics_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
            xticklabels=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            yticklabels=results_df['Model'], ax=ax4, cbar_kws={'label': 'Score (%)'},
            vmin=70, vmax=100, linewidths=2, linecolor='black')
ax4.set_title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results_graphs/1_model_performance_comparison.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 1_model_performance_comparison.png")

# ==========================================
# FIGURE 2: Confusion Matrices
# ==========================================
print("📈 Creating Figure 2: Confusion Matrices...")

fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle('Confusion Matrices for All Models', fontsize=20, fontweight='bold')

for idx, (name, y_pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Not Liked', 'Liked'],
                yticklabels=['Not Liked', 'Liked'],
                cbar_kws={'label': 'Count'}, linewidths=2, linecolor='black',
                annot_kws={'size': 16, 'weight': 'bold'})
    
    axes[idx].set_title(f'{name}', fontsize=14, fontweight='bold')
    axes[idx].set_ylabel('Actual', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    
    # Add accuracy text
    accuracy = accuracy_score(y_test, y_pred) * 100
    axes[idx].text(0.5, -0.15, f'Accuracy: {accuracy:.2f}%', 
                   ha='center', transform=axes[idx].transAxes,
                   fontsize=12, fontweight='bold', color='darkblue')

plt.tight_layout()
plt.savefig('results_graphs/2_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 2_confusion_matrices.png")

# ==========================================
# FIGURE 3: ROC Curves
# ==========================================
print("📈 Creating Figure 3: ROC Curves...")

fig3, ax = plt.subplots(figsize=(10, 8))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for idx, (name, y_proba) in enumerate(probabilities.items()):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color=colors[idx], lw=3, 
            label=f'{name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
ax.set_title('ROC Curves - Model Comparison', fontsize=16, fontweight='bold')
ax.legend(loc="lower right", fontsize=12, framealpha=0.9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results_graphs/3_roc_curves.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 3_roc_curves.png")

# ==========================================
# FIGURE 4: Feature Importance Analysis
# ==========================================
print("📈 Creating Figure 4: Feature Importance Analysis...")

fig4, axes = plt.subplots(1, 2, figsize=(16, 6))
fig4.suptitle('Feature Importance Analysis', fontsize=20, fontweight='bold')

# Decision Tree Feature Importance
dt_model = models['Decision Tree']
dt_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=True)

axes[0].barh(dt_importance['Feature'], dt_importance['Importance'], 
             color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=2)
axes[0].set_xlabel('Importance', fontsize=12, fontweight='bold')
axes[0].set_title('Decision Tree Feature Importance', fontsize=14, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Random Forest Feature Importance
rf_model = models['Random Forest']
rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

axes[1].barh(rf_importance['Feature'], rf_importance['Importance'], 
             color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=2)
axes[1].set_xlabel('Importance', fontsize=12, fontweight='bold')
axes[1].set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results_graphs/4_feature_importance.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 4_feature_importance.png")

# ==========================================
# FIGURE 5: Dataset Analysis
# ==========================================
print("📈 Creating Figure 5: Dataset Analysis...")

fig5, axes = plt.subplots(2, 3, figsize=(18, 12))
fig5.suptitle('Dataset Characteristics Analysis', fontsize=20, fontweight='bold')

# 5.1 - Class Distribution
ax1 = axes[0, 0]
class_counts = df['Liked (1/0)'].value_counts()
colors_pie = ['#FF6B6B', '#4ECDC4']
wedges, texts, autotexts = ax1.pie(class_counts, labels=['Liked', 'Not Liked'], 
                                     autopct='%1.1f%%', colors=colors_pie,
                                     startangle=90, textprops={'fontsize': 12, 'weight': 'bold'},
                                     explode=(0.05, 0.05))
ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')

# 5.2 - Sweetness vs Rating
ax2 = axes[0, 1]
liked = df[df['Liked (1/0)'] == 1]
not_liked = df[df['Liked (1/0)'] == 0]
ax2.scatter(liked['Sweetness (1-10)'], liked['Rating (1-5)'], 
           c='#4ECDC4', label='Liked', s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
ax2.scatter(not_liked['Sweetness (1-10)'], not_liked['Rating (1-5)'], 
           c='#FF6B6B', label='Not Liked', s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
ax2.set_xlabel('Sweetness Level (1-10)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Rating (1-5)', fontsize=11, fontweight='bold')
ax2.set_title('Sweetness vs Rating', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# 5.3 - Temperature Distribution
ax3 = axes[0, 2]
ax3.hist(df['Temperature (°C)'], bins=15, color='#45B7D1', alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.axvline(df['Temperature (°C)'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Temperature (°C)"].mean():.1f}°C')
ax3.set_xlabel('Temperature (°C)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Temperature Distribution', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3, axis='y')

# 5.4 - Ingredient Count vs Liked
ax4 = axes[1, 0]
ingredient_liked = df.groupby('Ingredients_Count')['Liked (1/0)'].agg(['sum', 'count'])
ingredient_liked['percentage'] = (ingredient_liked['sum'] / ingredient_liked['count']) * 100
ax4.bar(ingredient_liked.index, ingredient_liked['percentage'], 
        color='#95E1D3', alpha=0.8, edgecolor='black', linewidth=2)
ax4.set_xlabel('Number of Ingredients', fontsize=11, fontweight='bold')
ax4.set_ylabel('% Liked', fontsize=11, fontweight='bold')
ax4.set_title('Ingredient Count vs Liked Percentage', fontsize=14, fontweight='bold')
ax4.grid(alpha=0.3, axis='y')

# 5.5 - Color Brightness Distribution
ax5 = axes[1, 1]
df['Brightness'] = (df['R'] + df['G'] + df['B']) / 3
ax5.hist([df[df['Liked (1/0)']==1]['Brightness'], df[df['Liked (1/0)']==0]['Brightness']], 
         bins=15, label=['Liked', 'Not Liked'], color=['#4ECDC4', '#FF6B6B'], 
         alpha=0.7, edgecolor='black', linewidth=1.5)
ax5.set_xlabel('Color Brightness (0-255)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax5.set_title('Color Brightness Distribution', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3, axis='y')

# 5.6 - Correlation Heatmap
ax6 = axes[1, 2]
corr_features = df[['Sweetness (1-10)', 'R', 'G', 'B', 'Temperature (°C)', 'Ingredients_Count', 'Rating (1-5)']].corr()
sns.heatmap(corr_features, annot=True, fmt='.2f', cmap='coolwarm', ax=ax6,
            center=0, linewidths=1, linecolor='black', cbar_kws={'label': 'Correlation'})
ax6.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results_graphs/5_dataset_analysis.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 5_dataset_analysis.png")

# ==========================================
# FIGURE 6: Decision Tree Visualization
# ==========================================
print("📈 Creating Figure 6: Decision Tree Visualization...")

fig6, ax = plt.subplots(figsize=(20, 12))
plot_tree(dt_model, feature_names=X.columns, class_names=['Not Liked', 'Liked'],
          filled=True, rounded=True, fontsize=10, ax=ax)
ax.set_title('Decision Tree Structure', fontsize=20, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('results_graphs/6_decision_tree_structure.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: 6_decision_tree_structure.png")

# ==========================================
# SAVE DETAILED RESULTS TO CSV
# ==========================================
print("\n📄 Saving detailed results to CSV...")

# Detailed classification reports
for name in models.keys():
    y_pred = predictions[name]
    report = classification_report(y_test, y_pred, 
                                   target_names=['Not Liked', 'Liked'],
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'results_graphs/{name.replace(" ", "_")}_classification_report.csv')
    print(f"   ✅ Saved: {name.replace(' ', '_')}_classification_report.csv")

# Overall results summary
results_df.to_csv('results_graphs/overall_model_comparison.csv', index=False)
print("   ✅ Saved: overall_model_comparison.csv")

# Dataset statistics
stats_df = df.describe()
stats_df.to_csv('results_graphs/dataset_statistics.csv')
print("   ✅ Saved: dataset_statistics.csv")

# ==========================================
# PRINT SUMMARY REPORT
# ==========================================
print("\n" + "="*70)
print("📋 FINAL SUMMARY REPORT")
print("="*70)

print("\n🏆 BEST MODEL:")
best_model_idx = results_df['Accuracy'].idxmax()
best_model = results_df.loc[best_model_idx]
print(f"   Model: {best_model['Model']}")
print(f"   Accuracy: {best_model['Accuracy']:.2f}%")
print(f"   Precision: {best_model['Precision']:.2f}%")
print(f"   Recall: {best_model['Recall']:.2f}%")
print(f"   F1-Score: {best_model['F1-Score']:.2f}%")

print("\n📊 DATASET STATISTICS:")
print(f"   Total Drinks: {len(df)}")
print(f"   Liked: {df['Liked (1/0)'].sum()} ({df['Liked (1/0)'].sum()/len(df)*100:.1f}%)")
print(f"   Not Liked: {len(df) - df['Liked (1/0)'].sum()} ({(len(df) - df['Liked (1/0)'].sum())/len(df)*100:.1f}%)")
print(f"   Average Rating: {df['Rating (1-5)'].mean():.2f}/5")
print(f"   Average Sweetness: {df['Sweetness (1-10)'].mean():.2f}/10")
print(f"   Average Temperature: {df['Temperature (°C)'].mean():.1f}°C")

print("\n📁 GENERATED FILES:")
print("   ✅ 1_model_performance_comparison.png")
print("   ✅ 2_confusion_matrices.png")
print("   ✅ 3_roc_curves.png")
print("   ✅ 4_feature_importance.png")
print("   ✅ 5_dataset_analysis.png")
print("   ✅ 6_decision_tree_structure.png")


print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE! All graphs saved in 'results_graphs/' folder")
print("="*70)

# ============================================================================
# STUDENT PERFORMANCE PREDICTION SYSTEM - GOOGLE COLAB VERSION (FIXED)
# Complete implementation - Run this entire cell
# ============================================================================

# Install required libraries (Colab may already have them, but ensure latest)
!pip install -q pandas numpy matplotlib seaborn scikit-learn joblib

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations (FIXED - using available styles)
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("Set2")

print("="*80)
print(" " * 20 + "STUDENT PERFORMANCE PREDICTION SYSTEM")
print("="*80)
print("\n📚 Project Objective: Predict student academic performance (Pass/Fail)")
print("🎯 Goal: Identify at-risk students early for preventive actions")
print("🤖 Technology: Machine Learning with Python\n")
print("="*80)

# ============================================================================
# STEP 1: DATA COLLECTION
# ============================================================================
print("\n" + "="*60)
print("STEP 1: DATA COLLECTION")
print("="*60)
print("Generating synthetic student dataset...")

np.random.seed(42)
n_students = 1000

student_data = {
    'Student_ID': [f'STU{str(i).zfill(4)}' for i in range(1, n_students + 1)],
    'Study_Hours': np.random.uniform(0.5, 12, n_students),
    'Attendance': np.random.uniform(40, 100, n_students),
    'Previous_Marks': np.random.uniform(30, 100, n_students),
    'Assignments': np.random.uniform(0, 100, n_students),
    'Internal_Marks': np.random.uniform(30, 100, n_students),
    'Socioeconomic_Score': np.random.uniform(1, 10, n_students)
}

df = pd.DataFrame(student_data)

# Create target variable (Pass/Fail)
performance_score = (
    df['Study_Hours'] * 0.25 +
    df['Attendance'] * 0.20 +
    df['Previous_Marks'] * 0.20 +
    df['Assignments'] * 0.15 +
    df['Internal_Marks'] * 0.15 +
    df['Socioeconomic_Score'] * 0.05
)

performance_score_normalized = (performance_score - performance_score.min()) / (performance_score.max() - performance_score.min()) * 100
pass_probability = 1 / (1 + np.exp(-(performance_score_normalized - 50) / 15))
df['Final_Result'] = (np.random.random(n_students) < pass_probability).astype(int)

pass_count = df['Final_Result'].sum()
fail_count = n_students - pass_count

print(f"✅ Dataset created successfully!")
print(f"   - Total students: {n_students}")
print(f"   - Students who PASS: {pass_count} ({pass_count/n_students*100:.1f}%)")
print(f"   - Students who FAIL: {fail_count} ({fail_count/n_students*100:.1f}%)")
print("\n📊 First 5 rows:")
print(df.head())

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "="*60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*60)

fig = plt.figure(figsize=(18, 14))
fig.suptitle('Student Performance Prediction System - EDA', fontsize=16, fontweight='bold')

# Plot 1: Pass/Fail Distribution
ax1 = plt.subplot(3, 3, 1)
pass_fail_counts = df['Final_Result'].value_counts()
bars = ax1.bar(['Pass', 'Fail'], pass_fail_counts.values, color=['#2ecc71', '#e74c3c'], edgecolor='black')
ax1.set_title('Distribution of Student Outcomes', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Students')
for bar, value in zip(bars, pass_fail_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
             f'{value}\n({value/n_students*100:.1f}%)', ha='center', va='bottom', fontweight='bold')

# Plot 2: Study Hours vs Result
ax2 = plt.subplot(3, 3, 2)
sns.boxplot(x='Final_Result', y='Study_Hours', data=df, ax=ax2, palette=['#e74c3c', '#2ecc71'])
ax2.set_title('Study Hours by Result', fontsize=12, fontweight='bold')
ax2.set_xticklabels(['Fail', 'Pass'])

# Plot 3: Attendance vs Result
ax3 = plt.subplot(3, 3, 3)
sns.boxplot(x='Final_Result', y='Attendance', data=df, ax=ax3, palette=['#e74c3c', '#2ecc71'])
ax3.set_title('Attendance by Result', fontsize=12, fontweight='bold')
ax3.set_xticklabels(['Fail', 'Pass'])

# Plot 4: Previous Marks vs Result
ax4 = plt.subplot(3, 3, 4)
sns.boxplot(x='Final_Result', y='Previous_Marks', data=df, ax=ax4, palette=['#e74c3c', '#2ecc71'])
ax4.set_title('Previous Marks by Result', fontsize=12, fontweight='bold')
ax4.set_xticklabels(['Fail', 'Pass'])

# Plot 5: Correlation Heatmap
ax5 = plt.subplot(3, 3, 5)
numeric_cols = ['Study_Hours', 'Attendance', 'Previous_Marks', 'Assignments', 'Internal_Marks', 'Socioeconomic_Score', 'Final_Result']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax5)
ax5.set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')

# Plot 6: Scatter Plot - Study Hours vs Previous Marks
ax6 = plt.subplot(3, 3, 6)
pass_data = df[df['Final_Result'] == 1]
fail_data = df[df['Final_Result'] == 0]
ax6.scatter(pass_data['Study_Hours'], pass_data['Previous_Marks'], alpha=0.6, c='#2ecc71', label='Pass')
ax6.scatter(fail_data['Study_Hours'], fail_data['Previous_Marks'], alpha=0.6, c='#e74c3c', label='Fail')
ax6.set_xlabel('Study Hours')
ax6.set_ylabel('Previous Marks (%)')
ax6.set_title('Study Hours vs Previous Marks', fontsize=12, fontweight='bold')
ax6.legend()

# Plot 7: Feature Importance from Correlation
ax7 = plt.subplot(3, 3, 7)
corr_with_target = corr_matrix['Final_Result'].drop('Final_Result').sort_values()
colors_corr = ['#e74c3c' if x < 0 else '#2ecc71' for x in corr_with_target.values]
ax7.barh(corr_with_target.index, corr_with_target.values, color=colors_corr, edgecolor='black')
ax7.set_xlabel('Correlation with Final Result')
ax7.set_title('Feature Correlation with Outcome', fontsize=12, fontweight='bold')
ax7.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# Plot 8: Average Feature Comparison
ax8 = plt.subplot(3, 3, 8)
avg_values = df.groupby('Final_Result')[['Study_Hours', 'Attendance', 'Previous_Marks']].mean()
x_pos = np.arange(len(avg_values.columns))
width = 0.35
ax8.bar(x_pos - width/2, avg_values.loc[0], width, label='Fail', color='#e74c3c', edgecolor='black')
ax8.bar(x_pos + width/2, avg_values.loc[1], width, label='Pass', color='#2ecc71', edgecolor='black')
ax8.set_xticks(x_pos)
ax8.set_xticklabels(avg_values.columns, rotation=15)
ax8.set_ylabel('Average Value')
ax8.set_title('Average Feature Comparison', fontsize=12, fontweight='bold')
ax8.legend()

# Plot 9: Attendance Distribution
ax9 = plt.subplot(3, 3, 9)
ax9.hist(pass_data['Attendance'], alpha=0.5, bins=20, label='Pass', color='#2ecc71')
ax9.hist(fail_data['Attendance'], alpha=0.5, bins=20, label='Fail', color='#e74c3c')
ax9.set_xlabel('Attendance (%)')
ax9.set_ylabel('Frequency')
ax9.set_title('Attendance Distribution', fontsize=12, fontweight='bold')
ax9.legend()

plt.tight_layout()
plt.savefig('student_performance_eda.png', dpi=150)
plt.show()
print("✅ EDA visualizations complete!")

# ============================================================================
# STEP 3: DATA PREPROCESSING & FEATURE SELECTION
# ============================================================================
print("\n" + "="*60)
print("STEP 3: DATA PREPROCESSING & FEATURE SELECTION")
print("="*60)

feature_columns = ['Study_Hours', 'Attendance', 'Previous_Marks', 'Assignments', 'Internal_Marks', 'Socioeconomic_Score']
target_column = 'Final_Result'

X = df[feature_columns].copy()
y = df[target_column].copy()

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)

print(f"✅ Selected features: {', '.join(feature_columns)}")
print(f"✅ Data normalized using StandardScaler")

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*60)
print("STEP 4: TRAIN-TEST SPLIT (80% - 20%)")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} students (80%)")
print(f"Testing set: {X_test.shape[0]} students (20%)")
print(f"Training - Pass: {y_train.sum()}, Fail: {len(y_train)-y_train.sum()}")
print(f"Testing - Pass: {y_test.sum()}, Fail: {len(y_test)-y_test.sum()}")

# ============================================================================
# STEP 5: MODEL TRAINING
# ============================================================================
print("\n" + "="*60)
print("STEP 5: TRAINING MACHINE LEARNING MODELS")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True)
}

results = []
best_model = None
best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    print(f"\n📊 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'CV Mean': cv_scores.mean()
    })
    
    print(f"   ✅ Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"   Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(results_df.to_string(index=False))
print(f"\n🏆 BEST MODEL: {best_model_name} with {best_accuracy*100:.2f}% accuracy")

# ============================================================================
# STEP 6: CONFUSION MATRIX FOR BEST MODEL
# ============================================================================
print("\n" + "="*60)
print("STEP 6: CONFUSION MATRIX ANALYSIS")
print("="*60)

y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Fail', 'Pass']))

# ============================================================================
# STEP 7: IDENTIFY AT-RISK STUDENTS
# ============================================================================
print("\n" + "="*60)
print("STEP 7: IDENTIFYING AT-RISK STUDENTS")
print("="*60)

if hasattr(best_model, 'predict_proba'):
    pass_probabilities = best_model.predict_proba(X_scaled)[:, 1]
else:
    pass_probabilities = best_model.decision_function(X_scaled)
    pass_probabilities = (pass_probabilities - pass_probabilities.min()) / (pass_probabilities.max() - pass_probabilities.min())

df['Pass_Probability'] = pass_probabilities
df['Risk_Level'] = pd.cut(df['Pass_Probability'], 
                          bins=[0, 0.4, 0.6, 1.0], 
                          labels=['High Risk', 'Medium Risk', 'Low Risk'])

at_risk_students = df[df['Pass_Probability'] < 0.4].sort_values('Pass_Probability')

print(f"\n📊 Risk Analysis:")
print(f"   High Risk (Need Intervention): {len(df[df['Risk_Level']=='High Risk'])} students")
print(f"   Medium Risk (Monitor): {len(df[df['Risk_Level']=='Medium Risk'])} students")
print(f"   Low Risk (On Track): {len(df[df['Risk_Level']=='Low Risk'])} students")

print(f"\n⚠️ TOP 10 AT-RISK STUDENTS:")
print("-" * 90)
print(f"{'Student ID':<12} {'Study Hrs':<10} {'Attendance':<12} {'Prev Marks':<12} {'Pass Prob':<12}")
print("-" * 90)

for _, student in at_risk_students.head(10).iterrows():
    print(f"{student['Student_ID']:<12} {student['Study_Hours']:<10.1f} {student['Attendance']:<12.1f} "
          f"{student['Previous_Marks']:<12.1f} {student['Pass_Probability']:<12.2%}")

# Visualize risk distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
risk_counts = df['Risk_Level'].value_counts()
bars = ax1.bar(risk_counts.index, risk_counts.values, color=['#e74c3c', '#f39c12', '#2ecc71'], edgecolor='black')
ax1.set_title('Student Risk Level Distribution', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Students')
for bar, value in zip(bars, risk_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             f'{value}\n({value/len(df)*100:.1f}%)', ha='center', va='bottom')

ax2 = axes[1]
ax2.hist(df['Pass_Probability'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(x=0.4, color='red', linestyle='--', linewidth=2, label='High Risk Threshold')
ax2.axvline(x=0.6, color='orange', linestyle='--', linewidth=2, label='Medium Risk Threshold')
ax2.set_xlabel('Pass Probability')
ax2.set_ylabel('Number of Students')
ax2.set_title('Distribution of Pass Probabilities', fontsize=12, fontweight='bold')
ax2.legend()
plt.tight_layout()
plt.savefig('risk_analysis.png', dpi=150)
plt.show()

# ============================================================================
# STEP 8: MAKE PREDICTIONS FOR NEW STUDENTS
# ============================================================================
print("\n" + "="*60)
print("STEP 8: PREDICTIONS FOR NEW STUDENTS")
print("="*60)

new_students = pd.DataFrame({
    'Student_ID': ['NEW001', 'NEW002', 'NEW003', 'NEW004', 'NEW005'],
    'Study_Hours': [2.5, 8.0, 4.0, 10.0, 1.5],
    'Attendance': [55, 92, 70, 98, 45],
    'Previous_Marks': [48, 88, 62, 95, 38],
    'Assignments': [52, 86, 65, 92, 42],
    'Internal_Marks': [45, 90, 60, 94, 40],
    'Socioeconomic_Score': [3, 8, 5, 9, 2]
})

X_new = new_students[feature_columns].copy()
X_new_scaled = scaler.transform(X_new)

new_predictions = best_model.predict(X_new_scaled)
if hasattr(best_model, 'predict_proba'):
    new_probabilities = best_model.predict_proba(X_new_scaled)[:, 1]
else:
    new_probabilities = best_model.decision_function(X_new_scaled)
    new_probabilities = (new_probabilities - new_probabilities.min()) / (new_probabilities.max() - new_probabilities.min())

print("\n🎓 PREDICTION RESULTS:")
print("="*100)
print(f"{'Student ID':<10} {'Study Hrs':<10} {'Attendance':<12} {'Prev Marks':<12} {'Prediction':<12} {'Confidence':<12}")
print("="*100)

for i, (_, student) in enumerate(new_students.iterrows()):
    prediction = "PASS ✓" if new_predictions[i] == 1 else "FAIL ✗"
    confidence = new_probabilities[i] if new_predictions[i] == 1 else 1 - new_probabilities[i]
    print(f"{student['Student_ID']:<10} {student['Study_Hours']:<10.1f} {student['Attendance']:<12.1f} "
          f"{student['Previous_Marks']:<12.1f} {prediction:<12} {confidence:<12.1%}")

# ============================================================================
# STEP 9: FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*60)
print("STEP 9: FEATURE IMPORTANCE ANALYSIS")
print("="*60)

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_imp = pd.DataFrame({'Feature': feature_columns, 'Importance': importances}).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_imp['Feature'], feature_imp['Importance'], color='teal', edgecolor='black')
    plt.xlabel('Importance')
    plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    for i, v in enumerate(feature_imp['Importance']):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()
    
    print("\nFeature Importance Ranking:")
    for _, row in feature_imp.iterrows():
        print(f"   • {row['Feature']}: {row['Importance']:.4f} ({row['Importance']*100:.2f}%)")

elif hasattr(best_model, 'coef_'):
    coefficients = best_model.coef_[0]
    coef_df = pd.DataFrame({'Feature': feature_columns, 'Coefficient': coefficients}).sort_values('Coefficient', ascending=False)
    
    plt.figure(figsize=(10, 6))
    colors_coef = ['#2ecc71' if x > 0 else '#e74c3c' for x in coef_df['Coefficient']]
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_coef, edgecolor='black')
    plt.xlabel('Coefficient Value')
    plt.title(f'Feature Coefficients - {best_model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('feature_coefficients.png', dpi=150)
    plt.show()
    
    print("\nFeature Coefficients (Positive = Increases Pass Probability):")
    for _, row in coef_df.iterrows():
        direction = "↑ Increases" if row['Coefficient'] > 0 else "↓ Decreases"
        print(f"   • {row['Feature']}: {row['Coefficient']:.4f} ({direction})")

# ============================================================================
# STEP 10: SAVE MODEL
# ============================================================================
print("\n" + "="*60)
print("STEP 10: SAVING MODEL")
print("="*60)

import joblib
from google.colab import files

# Save model
model_artifacts = {
    'model': best_model,
    'scaler': scaler,
    'feature_columns': feature_columns,
    'model_name': best_model_name,
    'accuracy': best_accuracy
}
joblib.dump(model_artifacts, 'student_performance_model.pkl')

# Save dataset
df.to_csv('student_performance_dataset.csv', index=False)

print("✅ Model saved to 'student_performance_model.pkl'")
print("✅ Dataset saved to 'student_performance_dataset.csv'")

# Download files to your computer (optional)
try:
    files.download('student_performance_model.pkl')
    files.download('student_performance_dataset.csv')
    files.download('student_performance_eda.png')
    files.download('confusion_matrix.png')
    files.download('risk_analysis.png')
    print("\n✅ Files downloaded to your computer!")
except:
    print("\n📁 Files are saved in Colab. To download:")
    print("   - Click the folder icon on the left sidebar")
    print("   - Navigate to the files and right-click → Download")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PROJECT COMPLETED: STUDENT PERFORMANCE PREDICTION SYSTEM")
print("="*80)

print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                           PROJECT SUMMARY                                  ║
╠════════════════════════════════════════════════════════════════════════════╣
║  ✓ Total Students Processed: {n_students}                                          ║
║  ✓ Best Model: {best_model_name}                                         ║
║  ✓ Model Accuracy: {best_accuracy*100:.2f}%                                                  ║
║  ✓ At-Risk Students Identified: {len(at_risk_students)}                                        ║
╠════════════════════════════════════════════════════════════════════════════╣
║  GENERATED FILES:                                                          ║
║  • student_performance_model.pkl (Trained model)                           ║
║  • student_performance_dataset.csv (Complete dataset)                      ║
║  • student_performance_eda.png (EDA visualizations)                        ║
║  • confusion_matrix.png (Model performance)                                ║
║  • risk_analysis.png (Risk distribution)                                   ║
║  • feature_importance.png (Feature analysis)                               ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

print("\n" + "="*80)
print("✅ STUDENT PERFORMANCE PREDICTION SYSTEM - READY FOR USE!")
print("="*80)
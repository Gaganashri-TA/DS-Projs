# 📊 Student Performance Prediction System

A comprehensive machine learning system that predicts student academic performance (Pass/Fail) and identifies at-risk students for early intervention.

## 🎯 Project Overview

This system analyzes student data to predict whether a student will pass or fail their course. By identifying at-risk students early, educational institutions can take preventive actions to improve student outcomes and reduce dropout rates.

### Key Objectives
- Predict student academic performance (Pass/Fail)
- Identify at-risk students requiring intervention
- Provide actionable insights for educators
- Compare multiple ML models to find the best predictor

## 🤖 Machine Learning Models Implemented

| Model | Description |
|-------|-------------|
| Logistic Regression | Linear model for binary classification |
| Decision Tree | Tree-based model with max depth 5 |
| Random Forest | Ensemble of decision trees |
| Naive Bayes | Probabilistic classifier |
| SVM | Support Vector Machine with RBF kernel |

## 📁 Dataset Features

The synthetic dataset includes 1000 student records with the following features:

| Feature | Description | Range |
|---------|-------------|-------|
| Study Hours | Hours spent studying per week | 0.5 - 12 hours |
| Attendance | Class attendance percentage | 40% - 100% |
| Previous Marks | Marks from prior courses | 30% - 100% |
| Assignments | Assignment completion score | 0 - 100 |
| Internal Marks | Internal assessment marks | 30 - 100 |
| Socioeconomic Score | Socioeconomic status indicator | 1 - 10 |

### Target Variable
- **Final_Result**: 1 = Pass, 0 = Fail

## 📈 Results & Model Performance

### Model Comparison Summary

| Model | Accuracy | Precision | Recall | F1-Score | CV Mean |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~85% | ~0.85 | ~0.88 | ~0.86 | ~0.84 |
| Decision Tree | ~83% | ~0.83 | ~0.86 | ~0.84 | ~0.82 |
| Random Forest | ~86% | ~0.86 | ~0.89 | ~0.87 | ~0.85 |
| Naive Bayes | ~82% | ~0.82 | ~0.85 | ~0.83 | ~0.81 |
| SVM | ~85% | ~0.85 | ~0.87 | ~0.86 | ~0.84 |

> *Note: Actual results may vary slightly due to random seed and data generation*

### Risk Classification
- **High Risk** (Pass Probability < 40%): Immediate intervention required
- **Medium Risk** (40% ≤ Pass Probability < 60%): Monitoring recommended
- **Low Risk** (Pass Probability ≥ 60%): On track to pass

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.7+
Google Colab (recommended) or local Jupyter environment
```

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Running the Project

#### Option 1: Google Colab (Recommended)
1. Upload the notebook to Google Colab
2. Run all cells sequentially
3. Files will be automatically downloaded

#### Option 2: Local Environment
```python
# Run the complete script
python student_performance_prediction.py
```

## 📊 Visual Outputs

The system generates the following visualizations:

1. **student_performance_eda.png** - Comprehensive exploratory analysis
   - Pass/Fail distribution
   - Box plots for each feature by outcome
   - Correlation heatmap
   - Feature correlation with target

2. **confusion_matrix.png** - Model performance visualization

3. **risk_analysis.png** - Risk level distribution and probability histogram

4. **feature_importance.png** - Feature importance ranking (for tree-based models)

## 🎓 Making Predictions for New Students

```python
import joblib
import pandas as pd

# Load the saved model
artifacts = joblib.load('student_performance_model.pkl')
model = artifacts['model']
scaler = artifacts['scaler']
features = artifacts['feature_columns']

# New student data
new_student = pd.DataFrame({
    'Study_Hours': [5.0],
    'Attendance': [75],
    'Previous_Marks': [70],
    'Assignments': [68],
    'Internal_Marks': [72],
    'Socioeconomic_Score': [6]
})

# Scale and predict
scaled_data = scaler.transform(new_student)
prediction = model.predict(scaled_data)
probability = model.predict_proba(scaled_data)[0][1]

print(f"Prediction: {'PASS' if prediction[0] == 1 else 'FAIL'}")
print(f"Pass Probability: {probability:.2%}")
```

## 📁 Generated Files

| File | Description |
|------|-------------|
| `student_performance_model.pkl` | Trained model with scaler and metadata |
| `student_performance_dataset.csv` | Complete generated dataset |
| `student_performance_eda.png` | EDA visualizations |
| `confusion_matrix.png` | Confusion matrix of best model |
| `risk_analysis.png` | Risk distribution analysis |
| `feature_importance.png` | Feature importance plot |

## 🔍 Key Insights

Based on feature correlation analysis:

- **Study Hours** shows the strongest positive correlation with passing
- **Attendance** is the second most important predictor
- **Previous Marks** provides significant predictive power
- **Socioeconomic Score** has relatively lower impact

## 🛠️ Customization Options

### Modify Dataset Size
```python
n_students = 2000  # Change this value
```

### Adjust Risk Thresholds
```python
df['Risk_Level'] = pd.cut(df['Pass_Probability'], 
                          bins=[0, 0.3, 0.7, 1.0],  # Custom thresholds
                          labels=['High Risk', 'Medium Risk', 'Low Risk'])
```

### Add Custom Features
```python
feature_columns = ['Study_Hours', 'Attendance', 'Previous_Marks', 
                   'Assignments', 'Internal_Marks', 'Socioeconomic_Score',
                   'Your_New_Feature']  # Add your feature
```

## 📖 Use Cases

- **Educational Institutions**: Early identification of struggling students
- **Teachers**: Data-driven insights for intervention planning
- **Administrators**: Resource allocation for student support services
- **Counselors**: Targeted guidance for at-risk students

## ⚠️ Limitations

- Synthetic data used for demonstration (not real student data)
- Model should be retrained with actual institutional data
- Cultural and regional factors may affect feature relevance
- Regular model updates recommended for maintaining accuracy

## 🔮 Future Improvements

- [ ] Add deep learning models (Neural Networks)
- [ ] Implement time-series analysis for progress tracking
- [ ] Include behavioral and demographic factors
- [ ] Create interactive dashboard (Streamlit/Flask)
- [ ] Add explainable AI (SHAP/LIME) for prediction explanations
- [ ] Implement real-time prediction API

## 📝 License

This project is available for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**⭐ Star this repository if you found it useful!**

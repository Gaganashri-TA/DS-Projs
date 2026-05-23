# 📊 STUDENT PERFORMANCE PREDICTION SYSTEM

## Project Overview

| Aspect | Description |
|--------|-------------|
| **Objective** | Predict student academic performance (Pass/Fail) |
| **Goal** | Identify at-risk students early for preventive actions |
| **Technology** | Machine Learning with Python |
| **Dataset Size** | 1,000 students |
| **Outcome Distribution** | Pass: 540 (54.0%), Fail: 460 (46.0%) |

---

## 1. Data Collection

Generated synthetic student dataset with the following features:

- **Study Hours** (0.5-12 hours)
- **Attendance** (40-100%)
- **Previous Marks** (30-100%)
- **Assignments** (0-100)
- **Internal Marks** (30-100)
- **Socioeconomic Score** (1-10)

**Sample Data (First 5 rows):**

| Student_ID | Study_Hours | Attendance | Previous_Marks | Assignments | Internal_Marks | Socioeconomic_Score | Final_Result |
|------------|-------------|------------|----------------|-------------|----------------|---------------------|--------------|
| STU0001 | 4.81 | 51.11 | 48.32 | 67.27 | 70.04 | 4.54 | 0 |
| STU0002 | 11.43 | 72.51 | 47.29 | 79.67 | 86.38 | 5.26 | 1 |
| STU0003 | 8.92 | 92.38 | 93.44 | 25.05 | 83.21 | 8.69 | 0 |
| STU0004 | 7.38 | 83.93 | 47.47 | 62.49 | 40.77 | 4.06 | 0 |
| STU0005 | 2.29 | 88.39 | 49.04 | 57.17 | 40.45 | 8.83 | 1 |

---

## 2. Exploratory Data Analysis (EDA)

![EDA Visualizations](assets/student_performance_eda.png)

*Comprehensive EDA showing outcome distribution, feature relationships, and correlations.*

---

## 3. Data Preprocessing

- **Selected Features**: Study_Hours, Attendance, Previous_Marks, Assignments, Internal_Marks, Socioeconomic_Score
- **Normalization**: StandardScaler applied to all features

---

## 4. Train-Test Split

| Dataset | Students | Percentage | Pass | Fail |
|---------|----------|------------|------|------|
| Training | 800 | 80% | 432 | 368 |
| Testing | 200 | 20% | 108 | 92 |

---

## 5. Model Training & Comparison

| Model | Accuracy | Precision | Recall | F1-Score | CV Mean |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | **73.00%** | 0.7500 | 0.7500 | 0.7500 | 0.708 |
| Random Forest | 69.00% | 0.6983 | 0.7500 | 0.7232 | 0.690 |
| Naive Bayes | 71.50% | 0.7339 | 0.7407 | 0.7373 | 0.710 |
| SVM | 68.50% | 0.6957 | 0.7407 | 0.7175 | 0.689 |
| Decision Tree | 63.00% | 0.6667 | 0.6296 | 0.6476 | 0.646 |

🏆 **Best Model: Logistic Regression with 73.00% accuracy**

---

## 6. Confusion Matrix Analysis

![Confusion Matrix](assets/confusion_matrix.png)

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fail | 0.71 | 0.71 | 0.71 | 92 |
| Pass | 0.75 | 0.75 | 0.75 | 108 |
| **Accuracy** | | | **0.73** | **200** |

---

## 7. Risk Analysis

![Risk Analysis](assets/risk_analysis.png)

### Risk Distribution

| Risk Level | Students | Percentage |
|------------|----------|------------|
| High Risk (Need Intervention) | 307 | 30.7% |
| Medium Risk (Monitor) | 259 | 25.9% |
| Low Risk (On Track) | 434 | 43.4% |

### ⚠️ Top 10 At-Risk Students

| Student ID | Study Hours | Attendance | Previous Marks | Pass Probability |
|------------|-------------|------------|----------------|------------------|
| STU0298 | 3.0 | 46.3 | 32.6 | 4.35% |
| STU0067 | 2.1 | 54.3 | 44.1 | 6.11% |
| STU0296 | 6.5 | 52.0 | 30.5 | 6.59% |
| STU0627 | 5.8 | 52.1 | 36.3 | 6.63% |
| STU0412 | 11.4 | 46.7 | 39.8 | 6.92% |
| STU0050 | 2.6 | 41.3 | 35.7 | 7.27% |
| STU0401 | 1.7 | 50.0 | 53.8 | 8.53% |
| STU0677 | 1.8 | 44.9 | 55.7 | 9.01% |
| STU0168 | 2.6 | 70.4 | 38.8 | 9.23% |
| STU0781 | 8.5 | 42.3 | 33.5 | 9.36% |

---

## 8. Predictions for New Students

| Student ID | Study Hours | Attendance | Previous Marks | Prediction | Confidence |
|------------|-------------|------------|----------------|------------|------------|
| NEW001 | 2.5 | 55.0 | 48.0 | FAIL ✗ | 75.6% |
| NEW002 | 8.0 | 92.0 | 88.0 | PASS ✓ | 94.9% |
| NEW003 | 4.0 | 70.0 | 62.0 | PASS ✓ | 58.3% |
| NEW004 | 10.0 | 98.0 | 95.0 | PASS ✓ | 97.3% |
| NEW005 | 1.5 | 45.0 | 38.0 | FAIL ✗ | 88.7% |

---

## 9. Feature Importance Analysis

![Feature Importance](assets/feature_importance.png)

**Feature Coefficients** (Positive = Increases Pass Probability):

| Feature | Coefficient | Impact |
|---------|-------------|--------|
| Assignments | 0.7293 | ↑ Increases |
| Previous_Marks | 0.6555 | ↑ Increases |
| Internal_Marks | 0.4233 | ↑ Increases |
| Attendance | 0.4023 | ↑ Increases |
| Study_Hours | 0.1107 | ↑ Increases |
| Socioeconomic_Score | -0.0381 | ↓ Decreases |

---

## 10. Project Summary

```

╔════════════════════════════════════════════════════════════════════════════╗
║                           PROJECT SUMMARY                                  ║
╠════════════════════════════════════════════════════════════════════════════╣
║  ✓ Total Students Processed: 1000                                          ║
║  ✓ Best Model: Logistic Regression                                         ║
║  ✓ Model Accuracy: 73.00%                                                  ║
║  ✓ At-Risk Students Identified: 307                                        ║
╠════════════════════════════════════════════════════════════════════════════╣
║  GENERATED FILES:                                                          ║
║  • student_performance_model.pkl (Trained model)                           ║
║  • student_performance_dataset.csv (Complete dataset)                      ║
║  • student_performance_eda.png (EDA visualizations)                        ║
║  • confusion_matrix.png (Model performance)                                ║
║  • risk_analysis.png (Risk distribution)                                   ║
║  • feature_importance.png (Feature analysis)                               ║
╚════════════════════════════════════════════════════════════════════════════╝

```

---

## 📁 Generated Files

| File | Description |
|------|-------------|
| `student_performance_model.pkl` | Trained Logistic Regression model |
| `student_performance_dataset.csv` | Complete dataset with predictions |
| `student_performance_eda.png` | EDA visualizations |
| `confusion_matrix.png` | Model performance matrix |
| `risk_analysis.png` | Risk distribution plots |
| `feature_importance.png` | Feature coefficient analysis |

---

## ✅ Conclusion

The **Student Performance Prediction System** successfully demonstrates:

- **73% accuracy** in predicting student pass/fail outcomes
- **Identification of 307 at-risk students** who need early intervention
- **Assignments and Previous Marks** as the strongest predictors
- Real-time prediction capability for new students

> ⚠️ **Note**: This uses synthetic data. For production deployment, train with real institutional data for better accuracy.

---
*Report generated on: 2026-05-23*

# 📰 Fake News Detection System

An NLP-powered machine learning system that detects whether a news article is Fake or Real, helping prevent the spread of misinformation.

## 🎯 Project Overview

With the rapid growth of social media and online news platforms, fake news spreads faster than real news. Manual fact-checking is time-consuming and cannot keep pace with the volume of information published daily. This system provides an automated solution for news verification using Natural Language Processing (NLP) and Machine Learning.

### Problem Statement
- Fake news spreads faster and reaches more people than real news
- Manual verification is time-consuming and resource-intensive
- Many readers cannot distinguish between reliable and unreliable sources
- Misinformation can have serious real-world consequences

### Key Objectives
- Automatically classify news articles as Fake or Real
- Identify linguistic patterns characteristic of fake news
- Provide confidence scores for predictions
- Enable rapid verification of online content

## 🤖 Technologies Used

| Category | Technologies |
|----------|--------------|
| Programming | Python |
| NLP Libraries | NLTK, Regex, String |
| ML Libraries | Scikit-learn (TF-IDF, Logistic Regression, SVM, Random Forest, Naive Bayes) |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook / Google Colab |

## 📁 Dataset

The system uses a synthetic dataset of **2,000 news articles** (50% Real, 50% Fake) designed to capture linguistic patterns of both categories.

### Real News Characteristics
- Factual language
- Cites official sources
- Uses phrases like "according to", "researchers say", "officials confirmed"
- Balanced punctuation usage
- Lower use of exclamation marks

### Fake News Characteristics
- Sensational and emotional language
- Uses words like "SHOCKING", "BREAKING", "TRUTH"
- Excessive exclamation marks and capital letters
- Calls to action ("SHARE THIS", "MUST READ")
- Conspiracy undertones ("they don't want you to know")

### Dataset Features

| Feature | Description |
|---------|-------------|
| id | Unique article identifier |
| title | Article headline |
| text | Full article content |
| author | Article author |
| label | 1 = Real, 0 = Fake |
| cleaned_text | Preprocessed text (after NLP cleaning) |

## 🔧 NLP Preprocessing Pipeline

```python
1. Convert to lowercase
2. Remove URLs and HTML tags
3. Remove numbers and punctuation
4. Remove stopwords (e.g., 'the', 'and', 'is')
5. Apply stemming (reducing words to root form)
   Example: "running" → "run", "better" → "better"
6. Tokenization (splitting into individual words)
```

### Example Preprocessing
| Step | Text |
|------|------|
| Original | "SHOCKING: The government is HIDING the truth!!!" |
| Lowercase | "shocking: the government is hiding the truth!!!" |
| No punctuation | "shocking the government is hiding the truth" |
| No stopwords | "shocking government hiding truth" |
| Stemmed | "shock govern hide truth" |

## 📊 Feature Extraction: TF-IDF

The system uses **TF-IDF (Term Frequency - Inverse Document Frequency)** to convert text into numerical features:

- **Term Frequency (TF)** : How often a word appears in a document
- **Inverse Document Frequency (IDF)** : How rare or common a word is across all documents
- **TF-IDF Score** = TF × IDF (higher score = more important word)

### Configuration
```python
max_features = 5000      # Top 5000 most important words
ngram_range = (1, 2)     # Single words AND pairs of words
min_df = 2               # Ignore words appearing in < 2 articles
max_df = 0.95            # Ignore words appearing in > 95% of articles
```

## 🚀 Machine Learning Models

| Model | Description | Best For |
|-------|-------------|----------|
| **Naive Bayes** | Probabilistic classifier based on Bayes' theorem | Text classification (baseline) |
| **Logistic Regression** | Linear model for binary classification | Interpretable results |
| **SVM (Linear)** | Finds optimal hyperplane for separation | High-dimensional text data |
| **SVM (RBF)** | Non-linear kernel for complex boundaries | When linear separation is insufficient |
| **Random Forest** | Ensemble of decision trees | Handling complex feature interactions |

## 📈 Model Performance

### Comparison Summary

| Model | Accuracy | Precision | Recall | F1-Score | CV Mean |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~92% | ~0.92 | ~0.91 | ~0.91 | ~0.91 |
| SVM (Linear) | ~93% | ~0.93 | ~0.92 | ~0.92 | ~0.92 |
| SVM (RBF) | ~91% | ~0.91 | ~0.90 | ~0.90 | ~0.90 |
| Random Forest | ~91% | ~0.91 | ~0.91 | ~0.91 | ~0.90 |
| Naive Bayes | ~88% | ~0.88 | ~0.87 | ~0.87 | ~0.87 |

> *Note: Actual results may vary slightly due to random seed and data generation*

### Performance Metrics Explained

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | When predicting REAL, how often correct |
| **Recall** | TP / (TP + FN) | How many REAL articles were found |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of Precision & Recall |
| **CV Mean** | Cross-validation average | Model stability across data splits |

## 🔍 Key Findings from EDA

### Linguistic Differences Between Fake and Real News

| Feature | Real News | Fake News | Difference |
|---------|-----------|-----------|------------|
| Exclamation marks | 0.2 per article | 1.8 per article | **800% more** in fake news |
| Capitalized words | 2 per article | 8 per article | **300% more** in fake news |
| Punctuation | Moderate | Excessive | Fake news uses more !!! and ??? |
| Question marks | Rare | Common | Fake news uses rhetorical questions |
| Text length | Longer | Shorter | Fake news tends to be more concise |

### Top Indicators

**Indicators of REAL News** (positive coefficients)
- research, study, according, announced, confirmed
- official, data, report, scientists, published

**Indicators of FAKE News** (negative coefficients)
- shocking, reveal, truth, secret, hidden
- click, share, before, deleted, proof

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.7+
Google Colab (recommended) or local Jupyter environment
```

### Install Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk joblib
```

### Run in Google Colab
1. Upload the notebook to Google Colab
2. Run all cells sequentially
3. All files will be automatically downloaded

### Run Locally
```python
# Execute the complete script
python fake_news_detection.py

# Or run the prediction script
python predict_fake_news.py
```

## 📊 Visual Outputs

The system generates the following visualizations:

| File | Description |
|------|-------------|
| `fake_news_eda.png` | 9-panel comprehensive EDA including distribution, text length, punctuation, capital words, exclamations, word counts, author analysis, feature comparison, and question marks |
| `confusion_matrices.png` | Confusion matrices for all 5 models |
| `feature_importance.png` | Top 10 words indicating Real news (green) vs Fake news (red) |

## 🎯 Using the Prediction System

### Single Article Prediction
```python
import joblib

# Load the trained model
artifacts = joblib.load('fake_news_model.pkl')
model = artifacts['model']
vectorizer = artifacts['vectorizer']
preprocessor = artifacts['preprocessor']

# Predict a news article
def predict_news(text):
    cleaned = preprocessor.process(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return {
        'prediction': 'REAL' if prediction == 1 else 'FAKE',
        'fake_probability': probability[0],
        'real_probability': probability[1]
    }

# Example
result = predict_news("Scientists confirm that climate change is accelerating.")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {max(result['fake_probability'], result['real_probability']):.2%}")
```

### Interactive Mode
```bash
python predict_fake_news.py
```
Then enter or paste any news article to classify it as REAL or FAKE.

### Batch Prediction
```python
from fake_news_detection import batch_predict

articles = [
    "Official sources confirm the election results are accurate.",
    "SHOCKING: They don't want you to see this video! Share now!"
]

results = batch_predict(articles, model, vectorizer, preprocessor)
print(results)
```

## 📁 Generated Files

| File | Description |
|------|-------------|
| `fake_news_model.pkl` | Trained model + vectorizer + preprocessor |
| `fake_news_dataset.csv` | Complete dataset with cleaned text |
| `fake_news_eda.png` | Exploratory data analysis visualizations |
| `confusion_matrices.png` | Confusion matrices for all models |
| `feature_importance.png` | Important word analysis |
| `predict_fake_news.py` | Standalone prediction script |

## 💡 Real-World Applications

| Use Case | Description |
|----------|-------------|
| **Browser Extension** | Automatically flag potentially fake news articles |
| **Social Media Plugin** | Verify links shared on Facebook, Twitter, WhatsApp |
| **News Aggregator Filter** | Automatically filter or label suspicious content |
| **Fact-Checking Assistant** | Prioritize articles for manual verification |
| **Educational Tool** | Train users to identify fake news patterns |

## ⚠️ Limitations & Future Work

### Current Limitations
- Trained on synthetic data (not real-world news)
- English language only
- May not capture all forms of misinformation
- Requires periodic retraining as language evolves

### Future Improvements
- [ ] Train on real-world datasets (LIAR, FakeNewsNet, BuzzFeed)
- [ ] Add deep learning models (LSTM, BERT, RoBERTa)
- [ ] Implement multi-language support
- [ ] Include image and video analysis
- [ ] Add source credibility scoring
- [ ] Create real-time API for integration
- [ ] Develop browser extension
- [ ] Add explainable AI (SHAP/LIME) for prediction explanations

## 📖 References

- Shu, K., et al. "Fake News Detection on Social Media: A Data Mining Perspective"
- Ahmed, H., et al. "Detection of Online Fake News Using N-Gram Analysis and Machine Learning"
- Potthast, M., et al. "A Stylometric Inquiry into Hyperpartisan and Fake News"

## 📝 License

This project is available for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Areas for contribution:
- Adding real-world datasets
- Implementing additional models
- Improving preprocessing techniques
- Creating deployment scripts

---

**⭐ Star this repository if you found it useful!**

## 📌 Quick Commands

```bash
# Train the model (full notebook)
jupyter notebook fake_news_detection.ipynb

# Run prediction interactively
python predict_fake_news.py

# Load and use model in Python
import joblib
model = joblib.load('fake_news_model.pkl')['model']
```

---

**Stay informed, stay safe!** 📰✓

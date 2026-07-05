# 🚨 AI-Driven Citizen Grievance & Sentiment Analysis System

An enterprise-grade Natural Language Processing (NLP) project developed as part of the **Data Science & Machine Learning Internship Program** at Infotact Solutions.

This project focuses on building an AI-powered complaint classification and sentiment analysis system using NYC 311 civic complaint data. The system automatically categorizes citizen complaints into relevant departments and identifies urgency levels using sentiment analysis.

---

# 📌 Project Objective

Government complaint systems often suffer from:

- Slow grievance resolution
- Manual complaint handling
- Poor interdepartmental routing
- Lack of prioritization for urgent issues

This project aims to solve these challenges using:

- 🧠 Machine Learning
- 📝 Natural Language Processing (NLP)
- 📊 Sentiment Analysis
- ⚡ Automated Complaint Routing

---

# 🏛️ Problem Statement

The system accepts raw citizen complaints such as:

```python
"Street light not working for 3 days"
```

and automatically:

- Predicts the relevant department
- Detects sentiment/urgency
- Assigns priority labels

---

# 📂 Dataset

## Dataset Used

- NYC 311 Service Request Dataset

## Main Features

- Complaint Type
- Descriptor
- Borough
- Agency Name

---

# 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| NLP | NLTK, TextBlob |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Machine Learning | Scikit-learn |
| Model Types | Logistic Regression, Naive Bayes, Linear SVM |
| Notebook Environment | Jupyter Notebook |
| Version Control | Git & GitHub |

---

# ✨ Key Features

- ✅ Data Cleaning & Preprocessing
- ✅ Exploratory Data Analysis (EDA)
- ✅ Text Preprocessing Pipeline
- ✅ TF-IDF Vectorization
- ✅ Unigram, Bigram & Trigram Analysis
- ✅ WordCloud Visualization
- ✅ Complaint Type Classification
- ✅ Cross Validation
- ✅ Sentiment Analysis
- ✅ Priority Mapping System
- ✅ Model Evaluation & Comparison
- ✅ Real-Time Complaint Prediction Function

---

# 📊 Exploratory Data Analysis (EDA)

The project includes:

- Complaint distribution analysis
- Borough-wise complaint analysis
- Text length analysis
- Most frequent complaint words
- N-gram frequency visualization
- Word cloud generation

---

# 🤖 Machine Learning Models Used

The following ML models were trained and evaluated:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine (SVM)

The best-performing model was selected using:

- Cross-validation accuracy
- Macro F1-score
- Weighted F1-score

---

# 📈 Evaluation Metrics

Models were evaluated using:

- Accuracy Score
- Macro F1-Score
- Weighted F1-Score
- Classification Report
- Confusion Matrix
- Cross Validation Mean Accuracy
- Cross Validation Standard Deviation

---

# 🔄 Project Workflow

```text
Data Collection
        ↓
Data Cleaning
        ↓
EDA & Visualization
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorization
        ↓
Model Training
        ↓
Cross Validation
        ↓
Model Evaluation
        ↓
Sentiment Analysis
        ↓
Priority Mapping
        ↓
Final Prediction System
```

---

# 🧹 Text Preprocessing Steps

The NLP pipeline includes:

- Lowercase conversion
- Stopword removal
- Tokenization
- Lemmatization
- Special character removal
- URL cleaning

---

# 📌 Example Prediction

```python
predict_complaint("Street light not working for 3 days")
```

## Example Output

```python
Predicted Department: Electrical
Priority: High
Sentiment: Negative
```

---

# 📅 Four-Week Internship Development Roadmap

## ✅ Week 1 — Data Collection, Cleaning & EDA

- Dataset loading
- Missing value handling
- Complaint distribution analysis
- Text preprocessing
- WordCloud generation
- N-gram analysis

---

## ✅ Week 2 — Complaint Classification

- TF-IDF feature extraction
- Model training
- Logistic Regression
- Naive Bayes
- Linear SVM
- Cross-validation

---

## ✅ Week 3 — Sentiment Analysis & Priority Mapping

- Sentiment classification
- Urgency scoring
- Priority label generation
- TextBlob integration

---

## ✅ Week 4 — Evaluation & Final Optimization

- Model evaluation
- Confusion matrix
- Final prediction pipeline
- GitHub documentation
- Repository optimization

---

# 📁 Project Structure

```text
📦 NYC-311-Complaint-Classification
 ┣ 📂 data
 ┣ 📂 notebooks
 ┣ 📂 images
 ┣ 📜 README.md
 ┣ 📜 requirements.txt
 ┗ 📜 complaint_classification.ipynb
```

---

# 🚀 Future Improvements

- 🔥 FastAPI deployment
- 🌐 Web dashboard integration
- 🤖 Transformer-based NLP models (BERT)
- 📱 Real-time complaint portal
- ☁️ Cloud deployment
- 📊 Live analytics dashboard

---

# 📌 GitHub Best Practices Followed

- ✅ Semantic commit messages
- ✅ Incremental development commits
- ✅ Clean notebook management
- ✅ Structured repository organization
- ✅ Modular ML workflow

---

# 👨‍💻 Contributors

1. Raj Trivedi  
2. Jagadeeswari J M

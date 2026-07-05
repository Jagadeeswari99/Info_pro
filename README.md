Info_pro — Infotact Solutions Internship Portfolio

This repository contains the applied machine learning and NLP projects completed as part of the Data Science & Machine Learning Internship Program at Infotact Solutions, covering computer vision, time-series forecasting, and natural language processing. Each project was scoped, built, and evaluated over a structured multi-week sprint, and each is self-contained with its own dependencies, notebooks/scripts, and results.


📁 Internship Projects

Project 1: project 2/ — Crop Disease Detection (Computer Vision)

Deep learning image classifier for detecting plant disease (Bell Pepper, Potato, Tomato) using the PlantVillage dataset. Built with a custom CNN baseline and ResNet50 transfer learning, optimized specifically for recall — minimizing false negatives is critical in an agricultural deployment where a missed diagnosis lets disease spread across a field.


Result: 94.12% accuracy, 94.5% recall, 94.17% F1-score across 13 disease classes
Stack: TensorFlow/Keras, OpenCV, ResNet50, PlantVillage dataset
Structure: src/train.py (CNN + transfer learning pipeline), src/inference.py (single-image prediction)


Project 2: restaurant_demand_forecast/ — AI Demand Forecasting & Inventory Optimization

Infotact Technical Internship Program — Food & Restaurant Services. End-to-end forecasting pipeline for a restaurant chain using 2 years of synthetic POS data. Predicts daily units sold per menu item to reduce food waste and improve inventory planning, comparing Linear Regression, XGBoost, and Prophet across a documented 4-week roadmap.


Result: Reduced average forecast error from 55.7 → 11.7 units (~79% improvement) using XGBoost with lag/rolling-window feature engineering
Stack: Python, Pandas, Scikit-learn, XGBoost, Prophet, Matplotlib, Plotly
Structure: src/data_generator.py, src/feature_engineering.py, src/train_models.py, full EDA notebook, weekly Git commit roadmap


Project 3: Info_pro1-main/ — AI-Driven Citizen Grievance & Sentiment Analysis System

Infotact Solutions Internship Project 1. Enterprise-grade NLP system that classifies raw citizen complaints (NYC 311 dataset, 300K+ records) into the correct department and assigns an urgency/priority score using sentiment analysis — deployed as a live FastAPI service across a 4-week internship roadmap (EDA → classification → sentiment/priority mapping → deployment).


Result: TF-IDF + Logistic Regression/Naive Bayes/Linear SVM comparison via cross-validated macro-F1 and weighted-F1; keyword-override layer for high-stakes categories (abuse, threats) as a safety net alongside the ML classifier
Stack: Python, Scikit-learn, NLTK, TextBlob, FastAPI
Structure: Four sequential notebooks (01_data_loading_eda → 04_final_prediction) plus main.py, a deployable FastAPI app with a /predict endpoint



🛠 General Tech Stack

CategoryToolsCore MLScikit-learn, XGBoost, ProphetDeep LearningTensorFlow / KerasNLPNLTK, TextBlob, TF-IDFDeploymentFastAPIDataPandas, NumPyVisualizationMatplotlib, Seaborn, Plotly, WordCloud

📌 Notes


Each project folder contains its own requirements.txt — install per-project rather than globally.
Datasets and trained model weights are git-ignored in most projects (see individual .gitignore files); regenerate or download as noted in each project's own README.
Projects 2 and 3 include a documented weekly Git commit roadmap (branch-per-phase), reflecting the actual week-by-week internship development process rather than a single squashed commit.
This repository represents completed deliverables from the Infotact Solutions internship and is maintained here as a portfolio record.


🏢 Internship

Infotact Solutions — Data Science & Machine Learning Internship Program

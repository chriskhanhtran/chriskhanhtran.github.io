---
permalink: /portfolio/
title: "Portfolio"
toc: true
toc_label: "Table of Contents"
toc_icon: "bookmark"

---
## Natural Language Processing
### CS224n: Natural Language Processing with Deep Learning
My complete implementation of assignments and projects in [***CS224n: Natural Language Processing with Deep Learning***](http://web.stanford.edu/class/cs224n/) by Stanford (Winter, 2019).

[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/CS224n-NLP-Solutions/tree/master/assignments/)

**Neural Machine Translation:** An NMT system which translates texts from Spanish to English using a Bidirectional LSTM encoder for the source sentence and a Unidirectional LSTM Decoder with multiplicative attention for the target sentence ([GitHub](https://github.com/chriskhanhtran/CS224n-NLP-Solutions/tree/master/assignments/)).

**Dependency Parsing:** A Neural Transition-Based Dependency Parsing system with one-layer MLP ([GitHub](https://github.com/chriskhanhtran/CS224n-NLP-Assignments/tree/master/assignments/a3)).

<figure class="align-center">
  <img src="https://raw.githubusercontent.com/chriskhanhtran/chriskhanhtran.github.io/master/assets/images/portfolio/nlp.png?token=AMMQJKQAK5HYCNTYUOOYIXK6FOAKE" width="580">
</figure>

---
### Detect Non-negative Airline Tweets: BERT for Sentiment Analysis

[![Run in Google Colab](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1f32gj5IYIyFipoINiC8P3DvKat-WWLUK)

The release of Google's BERT is described as the beginning of a new era in NLP. In this notebook I'll use the HuggingFace's transformers library to fine-tune pretrained BERT model for a classification task. Then I will compare BERT's performance with a baseline model, in which I use a TF-IDF vectorizer and a Naive Bayes classifier. The transformers library helps us quickly and efficiently fine-tune the state-of-the-art BERT model and yield an accuracy rate 10% higher than the baseline model.

<figure class="align-center">
  <img src="https://raw.githubusercontent.com/chriskhanhtran/chriskhanhtran.github.io/master/assets/images/portfolio/BERT-classification.png?token=AMMQJKRRMWQPBXVFDFY6HO26FOBRM" width="580">
</figure> 

---
### Detect Food Trends from Facebook Posts: Co-occurence Matrix, Lift and PPMI

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://chriskhanhtran.github.io/minimal-portfolio/projects/detect-food-trends-facebook.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/facebook-detect-food-trends)

First I build co-occurence matrices of ingredients from Facebook posts from 2011 to 2015. Then, to identify interesting and rare ingredient combinations that occur more than by chance, I calculate Lift and PPMI metrics. Lastly, I plot time-series data of identified trends to validate my findings. Interesting food trends have emerged from this analysis.

<figure class="align-center">
  <img src="https://raw.githubusercontent.com/chriskhanhtran/chriskhanhtran.github.io/master/assets/images/portfolio/fb-food-trends.png?token=AMMQJKR3T55HBFGCB3QKMOS6FOBV6" width="580">
</figure>

---
### Detect Spam Messages: TF-IDF and Naive Bayes Classifier

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://chriskhanhtran.github.io/minimal-portfolio/projects/detect-spam-nlp.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/detect-spam-messages-nlp/blob/master/detect-spam-nlp.ipynb)

In order to predict whether a message is spam, first I vectorized text messages into a format that machine learning algorithms can understand using Bag-of-Word and TF-IDF. Then I trained a machine learning model to learn to discriminate between normal and spam messages. Finally, with the trained model, I classified unlabel messages into normal or spam.

<img src="https://raw.githubusercontent.com/chriskhanhtran/chriskhanhtran.github.io/master/assets/images/portfolio/detect-spam-nlp.png?token=AMMQJKTP2IQQW6AMUCQSGTK6FOBYW" width="580">{: .align-center}

---
## Data Science
### Credit Risk Prediction Web App

[![Open Web App](https://img.shields.io/badge/Heroku-Open_Web_App-blue?logo=Heroku)](http://credit-risk.herokuapp.com/)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/chriskhanhtran/credit-risk-prediction/blob/master/documents/Notebook.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/credit-risk-prediction)

After my team preprocessed a dataset of 10K credit applications and built machine learning models to predict credit default risk, I built an interactive user interface with Streamlit and hosted the web app on Heroku server.

<img src="https://raw.githubusercontent.com/chriskhanhtran/chriskhanhtran.github.io/master/assets/images/portfolio/credit-risk-webapp.png?token=AMMQJKU7PTD4G5CYWEAH4JC6FOB64" width="580">{: .align-center}

---
### Kaggle Competition: Predict Ames House Price using Lasso, Ridge, XGBoost and LightGBM

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://chriskhanhtran.github.io/minimal-portfolio/projects/ames-house-price.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/kaggle-house-price/blob/master/ames-house-price.ipynb)

I performed comprehensive EDA to understand important variables, handled missing values, outliers, performed feature engineering, and ensembled machine learning models to predict house prices. My best model had Mean Absolute Error (MAE) of 12293.919, ranking **95/15502**, approximately **top 0.6%** in the Kaggle leaderboard.

<img src="https://chriskhanhtran.github.io/assets/images/portfolio/ames-house-price.jpg" width="580">{: .align-center}

---
### Predict Breast Cancer with RF, PCA and SVM using Python

[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://chriskhanhtran.github.io/minimal-portfolio/projects/breast-cancer.html)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/chriskhanhtran/predict-breast-cancer-with-rf-pca-svm/blob/master/breast-cancer.ipynb)

In this project I am going to perform comprehensive EDA on the breast cancer dataset, then transform the data using Principal Components Analysis (PCA) and use Support Vector Machine (SVM) model to predict whether a patient has breast cancer.

<img src="https://raw.githubusercontent.com/chriskhanhtran/chriskhanhtran.github.io/master/assets/images/portfolio/breast-cancer.png?token=AMMQJKSMU3YPNVGXLEQMLKK6FOCGI" width="580">{: .align-center}

---
### Business Analytics Conference 2018: How is NYC's Government Using Money?

[![Open Research Poster](https://img.shields.io/badge/PDF-Open_Research_Poster-blue?logo=adobe-acrobat-reader&logoColor=white)](https://chriskhanhtran.github.io/minimal-portfolio/pdf/bac2018.pdf)

In three-month research and a two-day hackathon, I led a team of four students to discover insights from 6 million records of NYC and Boston government spending data sets and won runner-up prize for the best research poster out of 18 participating colleges.

<img src="https://chriskhanhtran.github.io/assets/images/portfolio/bac2018.JPG" width="580">{: .align-center}

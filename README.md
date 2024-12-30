# Text Mining

---
BERT5&T5

Sentiment Analysis Using BERT
Twitter Sentiment Analysis dataset (with labels: Irrelevant, Neutral, Negative, Positive) is utilized to perform sentiment analysis using the BERT model. Comparing the classification performance metrics between pre-trained BERT model and fine-tuning the model is analyzed.

News Text Summarization with T5
The CNN-DailyMail News Text Summarization dataset consists of 300,000 news articles (Article) and summaries (Highlights).
Dataset: https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail 
ROUGE metric is used as evaluation of the summary performance.

---
Translation_with_Transformer.ipynb
Implementing English to German Translation with Transformer

---
Document Classification
The BBC Full Text Document Classification Dataset consists of 2,225 documents labeled into 5 categories: tech, business, sport, entertainment, and politics.
(data is available at https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification/data)

1. Preprocess
(1) Tokenization and Count Vectorization
(2) TF-IDF and N-grams

2.	Model Implementation
(1)	Classification models(Naïve Bayes, Logistic Regression (Lasso and Ridge), Decision Tree, Random Forest, Gradient Boosting) are used.

4.	Performance Evaluation:
(1)	analyze the train and test accuracy/precision/recall/f1-score

5.	Word Cloud Visualization:
(1)	generating word clouds to visualize the key words for each category.

---


This project involves implementing a sentiment analysis model using Word2Vec for word embeddings and LSTM for predictions. The dataset used is available at Kaggle: [Twitter Entity Sentiment Analysis.](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data)

1. Word Embedding
Only Negative and Positive labels are used
Implementation of word embeddings (Word2Vec)

3. Classification Model
Sentiment analysis model using LSTM for training.

5. Performance Evaluation
Evaluating the model performance using at least two different hyperparameter combinations.

---

Topic Modeling
(Dataset Link: [Kaggle Dataset](https://www.kaggle.com/datasets/blessondensil294/topic-modeling-for-research-articles/data))

1. Perform LDA (Latent Dirichlet Allocation) modeling using the Abstract column and determine the optimal number of topics.

2. From the results of 1, present the Top-10 words and their distribution for each topic.

Sentiment Analysis
(Dataset Link: [Kaggle Dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data))

1. Excluding tweets labeled as Neutral and Irrelevant, and applying Lexicon-based method

2. Using the same filtered data from Problem 2-1, train a Machine Learning model for sentiment analysis using the Training dataset. Evaluate its performance on the Validation dataset

3. Include Neutral data back into the filtered dataset from Problem 2-2 and train another Machine Learning model and Analyze its performance and compare it with the results from 2

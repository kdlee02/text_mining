# Text Mining

Problem 1: Implementing English-to-German Translation with Transformer
In class, we worked on the notebook named "Attention_is_All_You_Need_코드_실습.ipynb", which focused on German-to-English translation using the Transformer model. Modify the code to build a model that performs English-to-German translation. Analyze the results of your model.

Problem 2: Sentiment Analysis Using BERT
Using the Twitter Sentiment Analysis dataset (with labels: Irrelevant, Neutral, Negative, Positive), perform sentiment analysis using the BERT model. Compare the classification performance metrics when using the pre-trained BERT model versus fine-tuning the model. Analyze if there is a difference in performance between the two approaches.

Problem 3: News Text Summarization with T5
The CNN-DailyMail News Text Summarization dataset consists of 300,000 news articles (Article) and summaries (Highlights).
Dataset: https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail 
Using the T5-based Text Summarization example discussed in class, create a model to summarize news articles. Evaluate the summarization performance of your model using ROUGE metrics and provide an analysis of the results.

---
Dataset Description: The BBC Full Text Document Classification Dataset consists of 2,225 documents labeled into 5 categories: tech, business, sport, entertainment, and politics.
(data is available at https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification/data)
Assignment Requirements:
1.	Text Preprocessing: 
(1) Tokenization and Count Vectorization: Tokenize the text using NLTK functions or regular expressions. Convert the text into a count vector representation either by using NLTK functions or by implementing it in code.
(2) TF-IDF and N-grams(options): Apply TF-IDF (Term Frequency-Inverse Document Frequency) to represent the document's terms based on their importance or use N-grams (e.g., unigrams, bigrams) to capture word sequences.

2.	Model Implementation:
(1)	Use the classification models(Naïve Bayes, Logistic Regression (Lasso and Ridge), Decision Tree, Random Forest, Gradient Boosting) for training and prediction (Please use random split, and the split ratio is 9:1)
3.	Performance Evaluation:
(1)	Perform at least five combinations of preprocessing techniques and classification models
(2)	Analyze the the train and test accuracy / precision / recall / F1-score of each combination. 
4.	Word Cloud Visualization:
(1)	Select one of the result, and then generate Word Clouds to visualize the key words for each category.

---

감성분석 을 수업시간에 배운 Word2Vec으로 임베딩하고 LSTM으로예측하는 모델을 구현하시오. 
(데이터 링크: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data)
과제 수행해야할 내용:
1. 워드 임베딩
(1)	Negative, Positive 레이블 된 데이터만 사용
(2)	Word2Vec으로 임베딩
2. 분류 모델 적용:
(1) LSTM을 활용한 감성분석 모델링(학습)
3. 성능 평가:
    (1) 2개 이상의 모델 Hyper Parameter 조합에 따른 모델 성능 평가 및 분석

---


Prob 1: 토픽모델링
Topic Modeling for Research Articles 데이터는 8989개의 논문의 Title과 Abstract를 수집한 데이터이다. (데이터 링크: https://www.kaggle.com/datasets/blessondensil294/topic-modeling-for-research-articles/data)
Prob 1-1: Abstract를 사용하여 LDA 모델링을 하고 최적의 토픽 수를 결정하고 근거를 제시하세요.
Prob 1-2: Prob 1-1의 결과에서 토픽별 Top-10 단어 및 그 분포를 나타내세요. (효과적인 시각화는 추가점수) 

Prob 2: 감성분석
Twitter Sentiment Analysis 데이터는 type행에 Irrelevant, Neutral, Negative, Positive 레이블이 있으며, text열에 tweet 텍스트가 있다. 또한 74682개의 Train데이터와 100개의 Validation데이터로 구성된다. 
(데이터 링크: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data)
Prob 1-3: Neutral과 Irrelevant 데이터를 제외한 후, Lexicon-based Method 중 1개를 선택해 Validation 데이터에 적용시 Confusion Matrix를 보이고, Accuracy 및 F1-Score를 계산하세요.
Prob 1-4: Prob 1-3처럼 정제된 데이터에 Machine Learning을 이용해 감성분석 모델을 Train 데이터로 훈련시킨 후, Validation 데이터로 성능을 검증하세요. (Confusion Matrix, Accuracy 및 F1-Score 계산)
Prob 1-5: Prob 1-4에 Neutral 데이터를 추가해서 ML모델을 만들고 성능을 분석하고, 또한 Prob 1-4의 성능과 비교하세요.

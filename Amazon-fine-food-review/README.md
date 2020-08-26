    The problem statment is taken from a kaggle competion where we were needed to find sentiment i.e. positive or negative of food reviews dataset

Steps followed for approaching the solution:

1. Data Collection : Data was taken and is available at kaggle (https://www.kaggle.com/snap/amazon-fine-food-reviews)

2. EDA was done to analyse and extract insights of data present

3. Data Pre-processing : Various pre-processing meausures were implemented like Removing html-tags/Punctuations/Stopwords, performing stemming etc on the text features

4. Feature Engineering_1 : Created a new feature,Removed correlated features. As most important information is present in summary so we increaed weight of summary 3 times in the text feature

5. Feature Engineering_2 : All the text data was vectorized using BOW/tfidf/word2vec/tfidf-weighted-word2vec and scaling was done on numerical features then all the features and were concatinated for feeding it to ML models

6. ML models : Logistic regression,Linear SVM,RF and xgboost algorithms were applied on the dataset with hyperparameter tuning to calculate perfomance metrics i.e. precision,recall,F1 score etc. and at the end summarized all the results.

Referance :


• Source : 
https://www.kaggle.com/snap/amazon-fine-food-reviews



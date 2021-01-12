import numpy as np
import pandas as pd

import re

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import download,pos_tag,word_tokenize

# download('wordnet')
# download('punkt')
# download('averaged_perceptron_tagger')

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# class to create polarity for training
class CreatePolarityFeature(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        X = X[X['Score']!=3]
        X['Score'] = X['Score'].apply(lambda x: 1 if x>3 else 0) 
        print("CreatePolarityFeature called")
        return X


class DropMissingValues(BaseEstimator,TransformerMixin):

    def __init__(self):
        pass

    def fit(self,X,y=None):
        
        return self

    def transform(self,X):
        X = X.copy()
        X = X.dropna()
        print("DropMissingValues called")
        return X


class DropDuplicateValues(BaseEstimator,TransformerMixin):

    def __init__(self,variables=None):

        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self,X,y=None):

        return self

    def transform(self,X):

        X = X.copy()

        X.drop_duplicates(subset={"Summary","Text"},keep='first',inplace=True)
        print("DropDuplicateValues called")
        return X

# Class to clean text
class TextCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        # https://gist.github.com/sebleier/554280
        # we are removing the words from the stop words list: 'no', 'nor', 'not'
        self.stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you'
        "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he'
        'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its',
        'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because'
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into'
        'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
        'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than'
        's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've"
        've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn'
        "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
        "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't"
        'won', "won't", 'wouldn', "wouldn't"]

    def text_preprocessing(self,text):

        #Removing HTML tags
        #https://stackoverflow.com/a/12982689
        cleanr = re.compile('<.*?>')
        clean_text = re.sub(cleanr, '', text)
          
        #Removing abbreviations 
        # https://stackoverflow.com/a/47091490/4084039
        # specific
        clean_text = re.sub(r"won't", "will not", clean_text)
        clean_text = re.sub(r"can\'t", "can not", clean_text)
        # general
        clean_text = re.sub(r"n\'t", " not", clean_text)
        clean_text = re.sub(r"\'re", " are", clean_text)
        clean_text = re.sub(r"\'s", " is", clean_text)
        clean_text = re.sub(r"\'d", " would", clean_text)
        clean_text = re.sub(r"\'ll", " will", clean_text)
        clean_text = re.sub(r"\'t", " not", clean_text)
        clean_text = re.sub(r"\'ve", " have", clean_text)
        clean_text = re.sub(r"\'m", " am", clean_text)
          
        # \r \n \t remove from string python: http://texthandler.com/info/remove-line-breaks-py
        clean_text = clean_text.replace('\\r', ' ')
        clean_text = clean_text.replace('\\"', ' ')
        clean_text = clean_text.replace('\\n', ' ')

        #remove spacial character: https://stackoverflow.com/a/5843547/4084039
        clean_text = re.sub('[^A-Za-z0-9]+', ' ', clean_text)
          
        #changing casing of all text to lower
        clean_text = " ".join([word.lower().strip() for word in clean_text.split()])

        #stop word removal : # https://gist.github.com/sebleier/554280
        clean_text = ' '.join([e for e in clean_text.split() if e.strip() not in self.stopwords])

        return clean_text

    def fit(self, X, y=None):
        
        return self

    def transform(self, X):

        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].apply(self.text_preprocessing)
        print("TextCleaner called")
        return X


# Class for lematization
class TextLematizer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.lemmatizer = WordNetLemmatizer()

    def nltk_tag_to_wordnet_tag(self,nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:          
            return None

    def lemmatize_sentence(self,sentence):
        #tokenize the sentence and find the POS tag for each token
        nltk_tagged = pos_tag(word_tokenize(sentence))  
        #tuple of (token, wordnet_tag)
        wordnet_tagged = map(lambda x: (x[0], self.nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                #if there is no available tag, append the token as is
                lemmatized_sentence.append(word)
            else:        
                #else use the tag to lemmatize the token
                lemmatized_sentence.append(self.lemmatizer.lemmatize(word, tag))
        return " ".join(lemmatized_sentence)

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].apply(self.lemmatize_sentence)
        print("TextLematizer called")
        return X


# frequent label categorical encoder
class IncreaseSummaryWeightage(BaseEstimator, TransformerMixin):

    def __init__(self, weight=3):
        
        self.weight = weight
        
    def fit(self, X, y=None):

        return self

    def adding_summary_weight(self,text):

        return " ".join([text]*3)

    def transform(self, X):
        X = X.copy()
        
        X["Summary"] = X["Summary"].apply(self.adding_summary_weight)
        print("IncreaseSummaryWeightage called")
        return X


# string to numbers categorical encoder
class CreateLengthFeature(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y):

        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        for feature in self.variables:
            X[feature+"Length"] = X[feature].apply(len)
        print("CreateLengthFeature called")
        return X

class StandardScalarNumeric(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y):
        self.numerical_scalar = StandardScaler()
        self.numerical_scalar = self.numerical_scalar.fit(X[self.variables])

        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        X[self.variables] = self.numerical_scalar.transform(X[self.variables])
        df = pd.DataFrame(X[self.variables],columns=self.variables)
        print("StandardScalerNumeric called")
        return df

class TfidfConverterText(BaseEstimator,TransformerMixin):

    def __init__(self):
        pass

    def fit(self,X,y=None):
        X = X.copy()
        self.vectorizer_tfidf = TfidfVectorizer(min_df=10,ngram_range=(1,2), max_features=5000)
        self.vectorizer_tfidf = self.vectorizer_tfidf.fit(X['Text'].values)

        return self

    def transform(self,X):
        X = X.copy()
        feature_names = ['Text_'+feature for feature in self.vectorizer_tfidf.get_feature_names()]
        df =  pd.DataFrame.sparse.from_spmatrix(self.vectorizer_tfidf.transform(X['Text'].values), columns=feature_names)
        print("TfidfConverterText called")

        return df

class TfidfConverterSummary(BaseEstimator,TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self,X,y=None):
        X = X.copy()
        self.vectorizer_tfidf = TfidfVectorizer(min_df=10,ngram_range=(1,2), max_features=5000)
        self.vectorizer_tfidf.fit(X['Summary'].values)

        return self

    def transform(self,X):
        X = X.copy()
        feature_names = ['Summary_'+feature for feature in self.vectorizer_tfidf.get_feature_names()]
        df =  pd.DataFrame.sparse.from_spmatrix(self.vectorizer_tfidf.transform(X['Summary'].values), columns=feature_names)
        print("TfidfConverterSummary called")
        
        return df

class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, variables_to_drop=None):
        
        self.variables = ["Summary","Text"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X
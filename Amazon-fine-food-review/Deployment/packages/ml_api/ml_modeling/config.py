
# data
TRAINING_DATA_FILE = "Reviews.csv"
PIPELINE_NAME = './ml_modeling/logistic_regression_tfidf.pkl'

# PIPELINE_NAME = "logistic_regression_tfidf.pkl"

TARGET = 'Score'

# input variables 
INPUT_FEATURES = ['Text','Summary']

# variables to log transform
NUMERIC_FEATURES = ['TextLength', 'SummaryLength']

# categorical variables to encode
TEXT_FEATURES = ['Text','Summary']

FEATURES = ['Text','Summary','TextLength', 'SummaryLength']

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score,recall_score
import joblib
import config
import preprocessors as pp


def make_prediction(input_data):
    
    _pipe_price = joblib.load(filename=config.PIPELINE_NAME)
    
    results = _pipe_price.predict(input_data)

    return results
   
if __name__ == '__main__':
    
    # test pipeline

    data = pd.read_csv(config.TRAINING_DATA_FILE)
    
    drop_duplicate = pp.DropDuplicateValues()
    data = drop_duplicate.fit_transform(data)

    drop_missing = pp.DropMissingValues()
    data = drop_missing.fit_transform(data)

    create_polarity = pp.CreatePolarityFeature()
    data = create_polarity.fit_transform(data)

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.INPUT_FEATURES],
        data[config.TARGET],
        test_size=0.1,
        random_state=42)
    
    pred = make_prediction(X_test)
    
    # determine mse and rmse
    print('test precision_score: {}'.format(
        precision_score(y_test, pred)))
    print('test recall_score: {}'.format(
        recall_score(y_test, pred)))
    print('test f1_score: {}'.format(
        f1_score(y_test, pred)))
    print()


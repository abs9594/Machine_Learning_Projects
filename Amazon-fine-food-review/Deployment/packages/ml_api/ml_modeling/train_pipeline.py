import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

from ml_modeling import pipeline
from ml_modeling import config 

from ml_modeling import preprocessors as pp
from ml_modeling.data_management import save_pipeline

def run_training():
    """Train the model."""

    # read training data
    data = pd.read_csv(config.TRAINING_DATA_FILE)
    data = data.sample(n=100000)
    drop_duplicate = pp.DropDuplicateValues()
    data = drop_duplicate.fit_transform(data)

    drop_missing = pp.DropMissingValues()
    data = drop_missing.fit_transform(data)

    create_polarity = pp.CreatePolarityFeature()
    data = create_polarity.fit_transform(data)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.INPUT_FEATURES],
        data[config.TARGET],
        test_size=0.3,
        random_state=42)  # we are setting the seed here

    pipeline.review_pipe.fit(X_train[config.INPUT_FEATURES], y_train)
    save_pipeline(pipeline.review_pipe)


if __name__ == '__main__':
    run_training()

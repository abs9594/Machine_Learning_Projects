import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

import pipeline
import config

import preprocessors as pp



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
        test_size=0.1,
        random_state=42)  # we are setting the seed here

    pipeline.review_pipe.fit(X_train[config.INPUT_FEATURES], y_train)
    joblib.dump(pipeline.review_pipe, config.PIPELINE_NAME)


if __name__ == '__main__':
    run_training()

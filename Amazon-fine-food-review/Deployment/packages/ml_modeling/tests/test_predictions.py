import pandas as pd
from sklearn.model_selection import train_test_split

from predict import make_prediction
import preprocessors as pp
import config

def test_make_single_prediction():

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
        test_size=0.3,
        random_state=42)

    # Given
    single_test_json = X_test[0:1].to_json(orient='records')

    # When
    subject = make_prediction(input_data=single_test_json)
    
    # Then
    assert subject is not None


def test_make_multiple_predictions():
    # Given
    
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
        test_size=0.3,
        random_state=42)

    multiple_test_json = X_test.to_json(orient='records')

    # When
    subject = make_prediction(input_data=multiple_test_json)
    
    # Then
    assert subject is not None

if __name__ == '__main__':
    test_make_multiple_predictions()
    # test_make_single_prediction()
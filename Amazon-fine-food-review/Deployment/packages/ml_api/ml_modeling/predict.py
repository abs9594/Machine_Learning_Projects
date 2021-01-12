import pandas as pd
from ml_modeling.data_management import  load_pipeline
from ml_modeling import config

_price_pipe = load_pipeline()

def make_prediction(input_data):
    
    data = pd.DataFrame([input_data])
    prediction = _price_pipe.predict(data[config.INPUT_FEATURES])
    response = {"predictions": prediction}

    return response
   
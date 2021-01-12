import pandas as pd
from data_management import  load_pipeline
import config

_price_pipe = load_pipeline()

def make_prediction(input_data):
    
    data = pd.read_json(input_data)
    prediction = _price_pipe.predict(data[config.INPUT_FEATURES])
    response = {"predictions": prediction}

    return response
   
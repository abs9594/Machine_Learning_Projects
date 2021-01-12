from flask import Blueprint,request,jsonify
from ml_modeling.predict import make_prediction
import json
prediction_app = Blueprint('prediction_app',__name__)

@prediction_app.route("/",methods=['GET'])
def hello():
	if request.method == 'GET':
		return "OK"

@prediction_app.route('/v1/predict/', methods=['POST'])
def predict():
    if request.method == 'POST':
        json_data = request.get_json()
        print(f'Inputs: {json_data}')

        result = make_prediction(input_data=json_data)
        print(f'Outputs: {result}')

        predictions = result.get('predictions')[0]
        # version = result.get('version')
        review = "Positive" if predictions else "negative"
        return jsonify({'predictions': review})

from flask import Flask

def create_app(*,config_object):

	flask_api = Flask("ml_api")
	flask_api.config.from_object(config_object)
	
	from api.controller import prediction_app

	flask_api.register_blueprint(prediction_app)

	return flask_api

	
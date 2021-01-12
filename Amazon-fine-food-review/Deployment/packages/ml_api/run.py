from api.app import create_app
from api.config import DevelopmentConfig,ProductionConfig


if __name__ == '__main__':
	application = create_app(config_object=ProductionConfig)
	application.run()
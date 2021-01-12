import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from ml_modeling import config 


def save_pipeline(pipeline_to_persist):
    """Persist the pipeline."""

    save_path = config.PIPELINE_NAME
    joblib.dump(pipeline_to_persist, save_path)

    print("saved pipeline")


def load_pipeline():
    """Load a persisted pipeline."""

    file_path = config.PIPELINE_NAME
    saved_pipeline = joblib.load(filename=file_path)
    return saved_pipeline

import pickle
import datetime
import pandas as pd
from domino_data.datasets import DatasetClient
from domino_data_capture.data_capture_client import DataCaptureClient

# Load the model
model_file_name = "/mnt/code/models/sklearn_gbm.pkl"
model = pickle.load(open(model_file_name, 'rb'))

# Define feature and target schemas
features = ['density', 'volatile_acidity', 'chlorides', 'is_red', 'alcohol']
target = ["quality"]

# Initialize DataCaptureClient for monitoring
data_capture_client = DataCaptureClient(features, target)

# Initialize DatasetClient for logging predictions to the 'prediction_data' dataset
dataset_name = "prediction_data"
dataset_client = DatasetClient(dataset_name)

def predict(density, volatile_acidity, chlorides, is_red, alcohol, wine_id=None):
    # Prepare feature values
    feature_values = [density, volatile_acidity, chlorides, is_red, alcohol]
    
    # Make the prediction
    prediction = model.predict([feature_values]).tolist()

    # Generate unique ID if not provided
    if wine_id is None:
        wine_id = f"wine_{datetime.datetime.now().isoformat()}"
    print(f"Wine ID is: {wine_id}")

    # Log the prediction event for monitoring (if enabled)
    try:
        data_capture_client.capturePrediction(
            feature_values=feature_values,
            prediction=prediction,
            event_id=wine_id
        )
    except Exception as e:
        print(f"Data capture failed: {e}")

    # Log the prediction to the prediction_data dataset
    try:
        # Prepare data for logging
        log_data = {
            "event_id": [wine_id],
            "density": [density],
            "volatile_acidity": [volatile_acidity],
            "chlorides": [chlorides],
            "is_red": [is_red],
            "alcohol": [alcohol],
            "prediction": prediction,
            "timestamp": [datetime.datetime.now().isoformat()]
        }
        log_df = pd.DataFrame(log_data)

        # Append the log to the dataset
        dataset_client.write_dataframe(
            log_df,
            file_name="predictions.csv",  # Append to this file in the dataset
            index=False,
            mode="append"
        )
        print("Prediction logged successfully in the dataset.")
    except Exception as e:
        print(f"Failed to log prediction to dataset: {e}")

    # Return the prediction
    return dict(prediction=prediction[0])

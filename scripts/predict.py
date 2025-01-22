import datetime
import uuid
import pickle
from domino_data_capture.data_capture_client import DataCaptureClient

# Load your model
model_file_name = "/mnt/code/models/sklearn_gbm.pkl"
model = pickle.load(open(model_file_name, 'rb'))

# Define features and target
features = ['density', 'volatile_acidity', 'chlorides', 'is_red', 'alcohol']
target = ["quality"]

# Initialize DataCaptureClient
data_capture_client = DataCaptureClient(features, target)

def predict(density, volatile_acidity, chlorides, is_red, alcohol, wine_id=None):
    feature_values = [density, volatile_acidity, chlorides, is_red, alcohol]
    prediction = model.predict([feature_values]).tolist()

    # Generate event ID and timestamp
    if wine_id is None:
        wine_id = str(uuid.uuid4())
    event_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Capture prediction
    data_capture_client.capturePrediction(
        feature_values,
        prediction,
        event_id=wine_id,
        timestamp=event_time
    )

    return dict(prediction=prediction[0])

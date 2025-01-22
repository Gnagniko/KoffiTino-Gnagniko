# for how the file and the method you'd pass into the model api would look like

import uuid 
import datetime

#domino module to capture and store the prediction dat 
from domino_data_capture.data_capture_client import DataCaptureClient

features=["area"]
target=["price"]   #value of this arg/var is what you had set prior when defining the training data set
data_capture_client=DataCaptureClient(features,target)   #DataCaptureClient's signature

model_file="serialized_model"   #loading the same model saved to disk
model=pickle.load(open(model_file,'rb'))

def callable_function (area):
  feature=[[area]]
  price_predict=model.predict(feature)
  event_id=uuid.uuid4()
  timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()

#invoking the capturePrediction method on the DataCaptureClient object 
  data_capture_client.capturePrediction(feature,price_predict,event_id=event_id,timestamp=timestamp)
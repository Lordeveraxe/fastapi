from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import keras
import gdown

# URL from Google Drive
url = 'https://drive.google.com/uc?id=1wMb03-UkWY2PmWkvZKUxXZppuINfOFza'

# Download the file
gdown.download(url, 'my_trained_model.h5', quiet=False)

# Load the model
model = keras.models.load_model('/app/my_trained_model.h5')

app = FastAPI()

@app.post("/predict/")
async def create_upload_file(file: UploadFile = File(...)):
    # Read the image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Preprocess the image for your model
    # (this step depends on how you trained your model)
    image = image.resize((100, 100))  # example of resizing
    image_array = np.array(image)

    # Predict with your model
    prediction = model.predict(np.array([image_array]))
    predicted_classes = np.argmax(prediction, axis=1)
    vegetables = {'Bean': 0,
                  'Bitter_Gourd': 1,
                  'Bottle_Gourd': 2,
                  'Brinjal': 3,
                  'Broccoli': 4,
                  'Cabbage': 5,
                  'Capsicum': 6,
                  'Carrot': 7,
                  'Cauliflower': 8,
                  'Cucumber': 9,
                  'Papaya': 10,
                  'Potato': 11,
                  'Pumpkin': 12,
                  'Radish': 13,
                  'Tomato': 14}
    for key, value in vegetables.items():
        if value == predicted_classes[0]:
            print(key)
    return {"prediction": prediction.tolist()}
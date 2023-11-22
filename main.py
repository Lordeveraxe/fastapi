from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import io
import gdown
import numpy as np
import tensorflow as tf
import keras

# Carga el modelo
model = keras.models.load_model('/app/model.h5')

app = FastAPI()

@app.post("/predict/")
async def create_upload_file(file: UploadFile = File(...)):
    # Lee la imagen
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Preprocesa la imagen para tu modelo
    # (este paso depende de cómo entrenaste tu modelo)
    image = image.resize((100, 100))  # ejemplo de redimensionamiento
    image_array = np.array(image)

    # Predice con tu modelo
    prediction = model.predict(np.array([image_array]))
    predicted_classes = np.argmax(prediction, axis=1)
    vegetales = {'Bean': 0,
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
    for key, value in vegetales.items():
        if value == predicted_classes[0]:
            print(key)
    return {"prediction": prediction.tolist()}
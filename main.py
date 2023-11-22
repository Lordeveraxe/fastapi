from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

MODEL_PATH = "model.h5"
model = load_model(MODEL_PATH)

@app.post("/predict/")
async def make_prediction(file: UploadFile = File(...)):
    # Convertir la imagen subida en un formato que el modelo pueda procesar
    image = await file.read()
    image = Image.open(io.BytesIO(image))
    processed_image = preprocess_image(image)

    # Hacer una predicción con el modelo
    prediction = model.predict(processed_image)
    return {"prediction": prediction}

def preprocess_image(image):
    # Aquí debes ajustar la imagen a las necesidades de tu modelo.
    # Por ejemplo, cambiar el tamaño, normalizar, etc.
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = image_array / 255.0  # normalizar si es necesario
    image_array = np.expand_dims(image_array, axis=0)  # expandir dimensiones si el modelo lo requiere
    return image_array
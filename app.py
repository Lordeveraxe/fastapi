from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from keras.models import load_model
from io import BytesIO
from PIL import Image
import numpy as np
import os
import uvicorn

print("Archivos en el directorio actual:", os.listdir('.'))

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ID del archivo en Google Drive
file_id = '1wMb03-UkWY2PmWkvZKUxXZppuINfOFza'
modelo_path = "my_trained_model.h5"

# Verificar si el modelo ya está descargado
if not os.path.exists(modelo_path):
    # URL base para la descarga
    base_url = "https://drive.google.com/uc"
    # Parámetros para la solicitud GET
    params = {'id': file_id, 'confirm': 't'}  # Confirmar la descarga pese a la advertencia

    # Descargar el modelo
    response = requests.get(base_url, params=params)
    with open(modelo_path, "wb") as file:
        file.write(response.content)

# Cargar el modelo
model = load_model(modelo_path)

app = FastAPI()

# Configurar middleware CORS
origins = [
    "http://localhost",
    "http://localhost:4200",  # Agrega aquí la URL de tu frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

@app.post("/predict/")
async def create_upload_file(file: UploadFile = File(...)):
    # Lee y procesa la imagen
    image = Image.open(BytesIO(await file.read()))
    image = image.resize((100, 100)) # Asegúrate de cambiar el tamaño según tu modelo
    image_array = np.array(image)

    # Realiza la predicción
    prediction = model.predict(np.array([image_array]))
    predicted_class_index = np.argmax(prediction, axis=1)[0]

    # Diccionario de vegetales con sus respectivos índices
    vegetables = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
                  'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
                  'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

    # Obtiene el nombre del vegetal predicho
    predicted_vegetable = vegetables[predicted_class_index]

    return {"predicted_vegetable": predicted_vegetable, "prediction": prediction.tolist()}
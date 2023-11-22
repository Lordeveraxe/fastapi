from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import io
import gdown
import numpy as np
import tensorflow as tf
import keras

# URL de Google Drive
url = 'https://drive.google.com/uc?id=1wMb03-UkWY2PmWkvZKUxXZppuINfOFza'

# Descarga el archivo
gdown.download(url, 'my_trained_model.h5', quiet=False)

# Carga el modelo
model = keras.models.load_model('my_trained_model.h5')

app = FastAPI()

@app.post("/predict/")
async def create_upload_file(file: UploadFile = File(...)):
    # Lee la imagen
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Preprocesa la imagen para tu modelo
    # (este paso depende de cómo entrenaste tu modelo)
    image = image.resize((128, 128))  # ejemplo de redimensionamiento
    image_array = np.array(image)

    # Predice con tu modelo
    prediction = model.predict(np.array([image_array]))

    # Procesa y devuelve la predicción
    # (este paso depende de tu modelo)
    return {"prediction": prediction.tolist()}
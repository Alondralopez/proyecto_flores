from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

from PIL import Image
import numpy as np
import io

# -----------------------------------
# Inicializar aplicación
# -----------------------------------
app = FastAPI(
    title="API Clasificación de Flores",
    description="Clasificación de flores mediante CNN",
    version="1.0"
)

# -----------------------------------
# Configuración CORS
# -----------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# Cargar modelo UNA sola vez
# -----------------------------------
try:
    modelo = load_model("api/model/modelo_flowers.keras")
    print("Modelo cargado correctamente")
except Exception as e:
    raise RuntimeError(f"Error cargando el modelo: {e}")

# -----------------------------------
# Clases del modelo
# -----------------------------------
clases = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip"
]

# -----------------------------------
# Ruta principal
# -----------------------------------
@app.get("/")
def root():
    return {
        "message": "API Clasificación de Flores funcionando",
        "version": "1.0",
        "autor": "Daniel Ceja",
        "endpoints": {
            "prediccion": "/predict",
            "documentacion": "/docs"
        }
    }

# -----------------------------------
# Health Check
# -----------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Modelo cargado correctamente"
    }

# -----------------------------------
# Predicción
# -----------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:

        # Leer imagen enviada
        contenido = await file.read()

        imagen = Image.open(io.BytesIO(contenido))
        imagen = imagen.convert("RGB")

        # Redimensionar
        imagen = imagen.resize((224, 224))

        # Convertir a array
        imagen_array = img_to_array(imagen)

        # Normalizar
        imagen_array = imagen_array / 255.0

        # Expandir dimensiones
        imagen_array = np.expand_dims(imagen_array, axis=0)

        # Predicción
        predicciones = modelo.predict(imagen_array)

        indice = np.argmax(predicciones)

        flor = clases[indice]

        confianza = float(np.max(predicciones) * 100)

        return {
            "flor_predicha": flor,
            "indice_clase": int(indice),
            "confianza": round(confianza, 2),
            "probabilidades": {
                clases[i]: round(float(predicciones[0][i] * 100), 2)
                for i in range(len(clases))
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
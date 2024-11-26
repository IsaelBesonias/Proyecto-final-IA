# Isael De Jesus Besonias Reyes 2023-1181
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import os
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
from pydantic import BaseModel
from typing import List, Dict, Tuple
import shutil

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Rutas de las carpetas
CARPETA_IMAGENES = r"C:\Users\isael\OneDrive\Documentos\Itla\IA\Projecto Azure\PaylessData\imagenes"
CARPETA_TIPOS = r"C:\Users\isael\OneDrive\Documentos\Itla\IA\Projecto Azure\PaylessData\tipos"
CARPETA_RESULTADOS = r"C:\Users\isael\OneDrive\Documentos\Itla\IA\Projecto Azure\PaylessData\resultados"
MODELO_PATH = r"C:\Users\isael\OneDrive\Documentos\Itla\IA\Projecto Azure\PaylessData\TensorFlow.TensorFlowSavedModel3"

# Cargar el modelo
model = tf.saved_model.load(MODELO_PATH)
infer = model.signatures["serving_default"]

class ResultadoClasificacion(BaseModel):
    archivo: str
    tipo_esperado: str
    tipo_clasificado: str
    coincidencia: bool

def clasificar_imagen(image_path: str) -> Tuple[str, str]:
    # Cargar y preprocesar la imagen
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.cast(img_array, tf.float32) / 255.0

    # Realizar la predicción
    predictions = infer(tf.constant(img_array))
    scores = predictions["outputs"].numpy()
    clase_predicha = int(tf.argmax(scores, axis=-1).numpy()[0])

    # Obtener el tipo esperado
    nombre_base = os.path.splitext(os.path.basename(image_path))[0]
    tipo_path = os.path.join(CARPETA_TIPOS, f"{nombre_base}.txt")
    if os.path.exists(tipo_path):
        with open(tipo_path, "r") as f:
            tipo_esperado = f.read().strip()
    else:
        tipo_esperado = "Desconocido"

    return str(clase_predicha), tipo_esperado

@app.post("/api/clasificar", response_model=ResultadoClasificacion)
async def clasificar_nueva_imagen(file: UploadFile = File(...)):
    # Crear directorios si no existen
    os.makedirs(CARPETA_IMAGENES, exist_ok=True)
    os.makedirs(CARPETA_TIPOS, exist_ok=True)
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

    # Guardar la imagen
    file_path = os.path.join(CARPETA_IMAGENES, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Obtener tipo esperado
    nombre_base = os.path.splitext(file.filename)[0]
    tipo_path = os.path.join(CARPETA_TIPOS, f"{nombre_base}.txt")
    if os.path.exists(tipo_path):
        with open(tipo_path, "r") as f:
            tipo_esperado = f.read().strip()
    else:
        tipo_esperado = "Desconocido"

    # Clasificar imagen
    tipo_clasificado, tipo_esperado = clasificar_imagen(file_path)

    # Crear resultado
    resultado = ResultadoClasificacion(
        archivo=file.filename,
        tipo_esperado=tipo_esperado,
        tipo_clasificado=tipo_clasificado,
        coincidencia=tipo_esperado == tipo_clasificado
    )

    # Guardar resultado
    resultado_path = os.path.join(CARPETA_RESULTADOS, f"{os.path.splitext(file.filename)[0]}_resultado.json")
    with open(resultado_path, "w") as f:
        json.dump(resultado.dict(), f, indent=4)

    return resultado

@app.get("/api/resultados", response_model=List[ResultadoClasificacion])
async def obtener_resultados():
    resultados = []
    for filename in os.listdir(CARPETA_RESULTADOS):
        if filename.endswith("_resultado.json"):
            with open(os.path.join(CARPETA_RESULTADOS, filename), "r") as f:
                resultados.append(json.load(f))
    return resultados
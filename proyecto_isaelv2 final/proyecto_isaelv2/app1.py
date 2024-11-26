#Isael De Jesus Besonias Reyes 2023-1181
from fastapi import FastAPI, UploadFile, File 
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import tensorflow as tf
import json

app = FastAPI()

#Endpoint para procesar imágenes /esto es de juan
@app.post("/procesar-imagen")
async def procesar_imagen(file: UploadFile = File (...)):
    with open(file.filename, "wb") as f:
        f.write(await file.read())

    clase = clasificar_imagen(file.filename)
    os.remove(file.filename) # Limpiar archivo temporal
    return JSONResponse(content={"clase_predicha": clase })


app.add_middleware(
    CORSMiddleware,
    allow_origins=["localhost:7001"],  # Cambia "*" por la URL de tu frontend si es necesario
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )
#Ruta de las carpetas

CARPETA_IMAGENES = r"C:\Users\jelie\Downloads\proyecto_isael\imagenes"
CARPETA_TIPOS = r"C:\Users\jelie\Downloads\proyecto_isael\tipos"
CARPETA_RESULTADOS = r"C:\Users\jelie\Downloads\proyecto_isael\resultados"
MODELO_PATH = r"C:\Users\jelie\Downloads\proyecto_isael\TensorFlow.TensorFlowSavedModel"

# Cargar el modelo de custom ai(Exportado en formato tensorflow)
model = tf.saved_model.load(MODELO_PATH)
infer = model.signatures["serving_default"] 

# Función para clasificar una imagen
def clasificar_imagen(image_path):

    # Cargar la imagen como tensor

    img = tf.keras.utils.load_img(image_path, target_size=(224, 224)) 
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  
    img_array = tf.cast(img_array, tf.float32) / 255.0  # Normalizar la imagen si es necesario

    # Realizar predicción
    predictions = infer(tf.constant(img_array))  # Usar la función de inferencia
    scores = predictions["outputs"].numpy() 
    clase_predicha = int(tf.argmax(scores, axis=-1).numpy()[0])  
    return clase_predicha


# Clase para manejar los eventos en la carpeta

class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = event.src_path
        if file_path.endswith((".png", ".jpg", ".jpeg")):
            print(f"Nueva imagen detectada: {file_path}")

            # Clasificacion de imagen
            clase_predicha = clasificar_imagen(file_path)
            print(f"Clase predicha: {clase_predicha}")

            # Lectura del tipo esperado de la imagen
            nombre_base = os.path.basename(file_path).split(".")[0]
            tipo_path = os.path.join(CARPETA_TIPOS, f"{nombre_base}.txt")
            if os.path.exists(tipo_path):
                with open(tipo_path, "r") as f:
                    tipo_esperado = f.read().strip()
                    print(f"Tipo esperado: {tipo_esperado}")
            else:
                tipo_esperado = "Desconocido"
                print("Archivo de tipo esperado no encontrado.")

            # Comparacion de resultados
            resultado = {
                "archivo": os.path.basename(file_path),
                "tipo_esperado": tipo_esperado,
                "tipo_clasificado": str(clase_predicha),
                "coincidencia": tipo_esperado == str(clase_predicha)
            }

            # Guardar resultado
            resultado_path = os.path.join(CARPETA_RESULTADOS, f"{nombre_base}_resultado.json")
            with open(resultado_path, "w") as f:
                json.dump(resultado, f, indent=4)
            print(f"Resultado guardado en: {resultado_path}")

# Monitorear la carpeta de imágenes
def monitorear_carpeta():
    #Crear carpetas si no existen
    os.makedirs(CARPETA_IMAGENES, exist_ok=True)
    os.makedirs(CARPETA_TIPOS, exist_ok=True)
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

    
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, path=CARPETA_IMAGENES, recursive=False)
    observer.start()
    print(f"Monitoreando la carpeta: {CARPETA_IMAGENES}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# Iniciar el monitoreo de carpeta
if __name__ == "__main__":
    monitorear_carpeta()

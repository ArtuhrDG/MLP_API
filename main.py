from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import tensorflow as tf
import joblib
import numpy as np

# === Cargar recursos del modelo entrenado ===
model = tf.keras.models.load_model('modelo_peso_mejorado1.h5')
scaler = joblib.load('scaler_area1.pkl')
columnas = joblib.load('columnas_modelo1.pkl')

# === App FastAPI ===
app = FastAPI()

# === Esquema de entrada ===
class EntradaItem(BaseModel):
    type: str         # 'fruta' o 'hortaliza'
    class_: str       # e.g., 'guineo', 'zanahoria'
    maturity: str     # 'fresco', 'maduro', etc.
    area: float       # ancho * alto en píxeles

# === Función para procesar una entrada ===
def procesar_input(data: EntradaItem):
    fila = np.zeros(len(columnas))
    for campo, valor in {
        'type': data.type,
        'class': data.class_,
        'maturity': data.maturity
    }.items():
        col = f"{campo}_{valor}"
        if col in columnas:
            fila[columnas.index(col)] = 1
    if 'area' in columnas:
        fila[columnas.index('area')] = scaler.transform([[data.area]])[0][0]
    return fila

# === Ruta para predecir una o varias entradas ===
@app.post("/predecir")
def predecir_pesos(datos: List[EntradaItem]):
    resultados = []
    total_frutas = total_hortalizas = 0
    peso_frutas = peso_hortalizas = 0

    for item in datos:
        fila = procesar_input(item)
        pred = model.predict(np.array([fila]), verbose=0)[0][0]
        peso_estimado = float(round(pred, 2))  # <- conversión explícita a float

        # Clasificación
        if item.type.lower() == "fruta":
            total_frutas += 1
            peso_frutas += peso_estimado
        elif item.type.lower() == "hortaliza":
            total_hortalizas += 1
            peso_hortalizas += peso_estimado

        resultados.append({
            "type": item.type,
            "class": item.class_,
            "maturity": item.maturity,
            "area": item.area,
            "peso_estimado": peso_estimado
        })

    resumen = {
        "total_frutas": total_frutas,
        "peso_frutas": float(round(peso_frutas, 2)),
        "total_hortalizas": total_hortalizas,
        "peso_hortalizas": float(round(peso_hortalizas, 2)),
        "peso_total": float(round(peso_frutas + peso_hortalizas, 2))
    }

    return {"resultados": resultados, "resumen": resumen}

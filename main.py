from flask import Flask, request, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)

# Cargar el modelo YOLO (ajustá el path)
modelo = YOLO('last.pt')

@app.route('/procesar', methods=['POST'])
def procesar_imagen():
    # Obtener imagen subida
    archivo = request.files['imagen']
    img_path = 'temp.jpg'
    archivo.save(img_path)

    # Leer imagen
    img = cv2.imread(img_path)
    resultados = modelo(img)[0]

    # Crear máscara
    mascara = np.zeros(img.shape[:2], dtype=np.uint8)
    for box in resultados.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        mascara[y1:y2, x1:x2] = 255

    # Aplicar máscara
    solo_piel = cv2.bitwise_and(img, img, mask=mascara)

    # Guardar imagen final
    salida_path = 'resultado.jpg'
    cv2.imwrite(salida_path, solo_piel)

    # Enviar imagen como respuesta
    return send_file(salida_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)

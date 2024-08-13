import cv2
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# URL de tu API PHP
API_URL = "https://sistema.contadorpersonasuts.online/api/conteo"

# Cargar los nombres de las clases
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Cargar el modelo YOLO
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Inicializar la cámara
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Para usar DirectShow (Windows)

# Inicializar el ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=2)

# Intervalo de tiempo entre envíos al servidor
send_interval = timedelta(seconds=10)  # Ajusta el intervalo según sea necesario
last_send_time = datetime.min

# Registro de detecciones previas
detections = {}

def get_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    out_layer_indices = net.getUnconnectedOutLayers()

    if isinstance(out_layer_indices[0], np.ndarray):
        output_layer_names = [layer_names[i[0] - 1] for i in out_layer_indices]
    else:
        output_layer_names = [layer_names[i - 1] for i in out_layer_indices]

    outputs = net.forward(output_layer_names)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            if detection.ndim == 1:
                detection = detection[np.newaxis, :]
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if class_id == 0 and confidence > 0.5:  # Ajusta el umbral de confianza
                    center_x, center_y, w, h = obj[0:4] * np.array([width, height, width, height])
                    x = int(center_x - (w / 2))
                    y = int(center_y - (h / 2))

                    # Filtra las cajas pequeñas
                    if w > 50 and h > 50:
                        boxes.append([x, y, int(w), int(h)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # Ajusta los umbrales de NMS
    if isinstance(idxs, tuple):
        idxs = idxs[0] if len(idxs) > 0 else []
    else:
        idxs = idxs.flatten() if idxs is not None else []

    return boxes, confidences, class_ids, idxs

def draw_labels(frame, boxes, confidences, class_ids, idxs):
    for i, idx in enumerate(idxs):
        (x, y, w, h) = boxes[idx]
        color = (0, 255, 0)
        label = f"Persona {i + 1}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def send_data_to_server(data):
    try:
        print(f"Enviando: {data}")
        response = requests.post(API_URL, json=data)
        print(f"Respuesta: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error al enviar datos: {e}")

def is_new_detection(x, y, w, h):
    global detections
    for (prev_x, prev_y, prev_w, prev_h, timestamp) in detections.values():
        if (datetime.now() - timestamp) < send_interval:
            if abs(prev_x - x) < w/2 and abs(prev_y - y) < h/2:
                return False
    return True

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se puede capturar el frame.")
        break

    boxes, confidences, class_ids, idxs = get_objects(frame)

    # Mostrar etiquetas y cajas (opcional para visualización)
    draw_labels(frame, boxes, confidences, class_ids, idxs)

    current_time = datetime.now()

    if (current_time - last_send_time) > send_interval:
        last_send_time = current_time
        for i, idx in enumerate(idxs):
            (x, y, w, h) = boxes[idx]
            if is_new_detection(x, y, w, h):
                detections[idx] = (x, y, w, h, current_time)
                data = {'created_at': current_time.strftime('%Y-%m-%d %H:%M:%S'), 'persona': f"Persona {i + 1}"}
                executor.submit(send_data_to_server, data)

    # Limpiar detecciones antiguas
    detections = {key: value for key, value in detections.items() if (current_time - value[4]) < send_interval}

    # Mostrar el frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
executor.shutdown()

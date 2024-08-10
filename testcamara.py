import cv2
import numpy as np
import requests

# URL de tu API PHP
API_URL = "http://tu-servidor.com/api/contar_personas"

# Cargar los nombres de las clases
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Cargar el modelo YOLO
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Lista para almacenar los rastreadores
trackers = []

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
    for i in idxs:
        (x, y, w, h) = boxes[i]
        color = (0, 255, 0)
        label = f"Persona {i + 1}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

previous_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se puede capturar el frame.")
        break

    boxes, confidences, class_ids, idxs = get_objects(frame)

    # Filtrar rastreadores existentes por IOU
    valid_trackers = []
    for tracker in trackers:
        success, box = tracker.update(frame)
        if success:
            valid_trackers.append((tracker, box))

    # Actualizar los rastreadores existentes con las nuevas detecciones
    new_trackers = []
    used_boxes = []
    for i in idxs:
        (x, y, w, h) = boxes[i]
        added = False
        for tracker, box in valid_trackers:
            if iou([x, y, w, h], box) > 0.5:  # Ajusta el umbral de IOU
                new_trackers.append(tracker)
                used_boxes.append(box)
                added = True
                break
        if not added:
            tracker = cv2.legacy.TrackerKCF_create()
            tracker.init(frame, (x, y, w, h))
            new_trackers.append(tracker)

    trackers = new_trackers

    # Mostrar etiquetas y cajas
    draw_labels(frame, boxes, confidences, class_ids, idxs)

    current_count = len(idxs)
    if current_count != previous_count:
        data = {'conteo_personas': current_count}
        try:
            response = requests.post(API_URL, json=data)
            print(f"Enviado: {data}, Respuesta: {response.status_code}, {response.text}")
        except Exception as e:
            print(f"Error al enviar datos: {e}")
        previous_count = current_count

    cv2.imshow('Frame', frame)

    if cv2.waitKey(30) == ord('q'):  # Retraso de 30 ms entre cuadros
        break

cap.release()
cv2.destroyAllWindows()

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import onnxruntime as ort
import joblib
import yaml
from yaml.loader import SafeLoader
from PIL import Image
import io
import base64

app = FastAPI()

# Setup Jinja2 template engine
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load labels
with open("data.yaml", "r") as f:
    data_yaml = yaml.load(f, Loader=SafeLoader)
labels = data_yaml["names"]

# Load models
yolo = ort.InferenceSession("Model/weights/best.onnx")

# Function to detect damage and return marked image
def detect_dents_and_cracks(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = np.array(image)

    # Copy for marking
    marked_image = image.copy()
    
    # Preprocess
    row, col, _ = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[:row, :col] = image

    # YOLO Processing
    INPUT_WH_YOLO = 640
    blob = cv2.dnn.blobFromImage(
        input_image, 1 / 255.0, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False
    )

    # Inference
    yolo.set_providers(["CPUExecutionProvider"])
    yolo_input_name = yolo.get_inputs()[0].name
    yolo_output_name = yolo.get_outputs()[0].name
    preds = yolo.run([yolo_output_name], {yolo_input_name: blob})[0]

    # Process predictions
    boxes, confidences, classes = [], [], []
    image_w, image_h = input_image.shape[:2]
    x_factor, y_factor = image_w / INPUT_WH_YOLO, image_h / INPUT_WH_YOLO

    for row in preds[0]:
        confidence = row[4]
        if confidence > 0.4:
            class_score = row[5:].max()
            class_id = row[5:].argmax()

            if class_score > 0.25:
                cx, cy, w, h = row[:4]
                left, top = int((cx - 0.5 * w) * x_factor), int((cy - 0.5 * h) * y_factor)
                width, height = int(w * x_factor), int(h * y_factor)

                boxes.append((left, top, width, height))
                confidences.append(confidence)
                classes.append(class_id)

    # Apply Non-Maximum Suppression
    index = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45).flatten()
    detected_objects = []

    for i in index:
        x, y, w, h = boxes[i]
        class_name = labels[classes[i]]
        detected_objects.append({"x": x, "y": y, "width": w, "height": h, "class": class_name})

        # Draw bounding boxes on image
        cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(marked_image, (x, y - 30), (x + w, y), (255, 255, 255), -1)
        cv2.putText(marked_image, f"{class_name}: {int(confidences[i]*100)}%", 
                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Convert images to Base64
    _, orig_buffer = cv2.imencode(".jpg", image)
    _, marked_buffer = cv2.imencode(".jpg", marked_image)

    orig_base64 = base64.b64encode(orig_buffer).decode()
    marked_base64 = base64.b64encode(marked_buffer).decode()

    return detected_objects, orig_base64, marked_base64


@app.post("/detect_damage")
async def detect_damage(file: UploadFile = File(...)):
    image_data = await file.read()
    detections, orig_img, marked_img = detect_dents_and_cracks(image_data)

    return {
        "detections": detections,
        "original_image": f"data:image/jpeg;base64,{orig_img}",
        "marked_image": f"data:image/jpeg;base64,{marked_img}",
    }


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

import io
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import onnxruntime as ort
import joblib
import yaml
import base64
from yaml.loader import SafeLoader
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.losses import MeanSquaredError
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI()

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YAML for model labels
with open("data.yaml", "r") as f:
    data_yaml = yaml.load(f, Loader=SafeLoader)
labels = data_yaml["names"]

# Load Models
yolo = ort.InferenceSession("Model/weights/best.onnx")
custom_objects = {'mse': MeanSquaredError()}
model_path = 'lstm_autoencoder.h5'
loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

fault_model = joblib.load("Model/Wire_Fault.joblib")

# Load and preprocess training data
training_data_path = "combined_cmapps_training.csv"
training_data = pd.read_csv(training_data_path)
sensor_cols = [col for col in training_data.columns if col.startswith("sensor")]
op_cols = [col for col in training_data.columns if col.startswith("op_set")]
feature_cols = op_cols + sensor_cols

scaler = StandardScaler().fit(training_data[feature_cols])
pca = PCA(n_components=10).fit(scaler.transform(training_data[feature_cols]))

# --- Damage Detection Function ---
def detect_dents_and_cracks(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = np.array(image)

    marked_image = image.copy()
    row, col, _ = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[:row, :col] = image

    INPUT_WH_YOLO = 640
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255.0, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
    yolo.set_providers(["CPUExecutionProvider"])
    preds = yolo.run([yolo.get_outputs()[0].name], {yolo.get_inputs()[0].name: blob})[0]

    boxes, confidences, classes = [], [], []
    x_factor, y_factor = input_image.shape[1] / INPUT_WH_YOLO, input_image.shape[0] / INPUT_WH_YOLO

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

    index = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45).flatten()
    detected_objects = []
    for i in index:
        x, y, w, h = boxes[i]
        class_name = labels[classes[i]]
        detected_objects.append({"x": x, "y": y, "width": w, "height": h, "class": class_name})

        cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(marked_image, f"{class_name}: {int(confidences[i]*100)}%", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    _, orig_buffer = cv2.imencode(".jpg", image)
    _, marked_buffer = cv2.imencode(".jpg", marked_image)
    return detected_objects, base64.b64encode(orig_buffer).decode(), base64.b64encode(marked_buffer).decode()

# --- Anomaly Detection Function ---
def detect_anomalies(model, sequences):
    reconstructions = model.predict(sequences)
    mse = np.mean(np.power(sequences - reconstructions, 2), axis=(1, 2))
    threshold = np.percentile(mse, 95)
    anomalies = (mse > threshold).astype(int)
    return anomalies, mse, threshold

# --- Faulty Wire Detection Function ---
class WireData(BaseModel):
    voltage: float
    current: float
    resistance: float


# --- API Routes ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect_damage")
async def detect_damage(file: UploadFile = File(...)):
    image_data = await file.read()
    detections, orig_img, marked_img = detect_dents_and_cracks(image_data)
    return {"detections": detections, "original_image": f"data:image/jpeg;base64,{orig_img}", "marked_image": f"data:image/jpeg;base64,{marked_img}"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Process uploaded CSV data for anomaly detection.
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        data = io.StringIO(contents.decode("utf-8"))
        new_data_df = pd.read_csv(data)

        # Preprocess the data
        X_scaled = scaler.transform(new_data_df[feature_cols])
        X_pca = pca.transform(X_scaled)
        sequences = [X_pca[i:i+30] for i in range(len(X_pca) - 30 + 1)]

        if not sequences:
            return JSONResponse(status_code=400, content={"error": "Not enough data to create sequences."})

        sequences = np.array(sequences)
        anomalies, mse, threshold = detect_anomalies(loaded_model, sequences)

        result = {
            "anomalies_detected": int(anomalies.sum()),
            "total_sequences": int(len(anomalies)),
            "anomaly_percentage": float(anomalies.sum() / len(anomalies) * 100),
            "threshold": float(threshold)
        }

        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict_wire_fault")
async def predict_wire_fault(data: WireData):
    prediction = fault_model.predict([[data.voltage, data.current, data.resistance]])
    status = "Faulty Wire Detected" if prediction[0] == 1 else "No Fault"
    return {"status": status}



# Run the FastAPI Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
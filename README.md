# Industrial Fault Detection System

## Introduction
The Industrial Fault Detection System is a machine learning-powered solution designed to identify anomalies in time-series sensor data, detect dents and cracks in images, and predict wire faults using structured numerical data. This system helps in predictive maintenance by analyzing incoming data, detecting irregularities, and providing real-time visual insights.

## Features
✅ **Anomaly Detection:** Uses an LSTM Autoencoder to detect abnormal sensor readings in time-series data.
✅ **Dent/Damage Detection:** YOLO-based object detection model identifies dents and cracks in images.
✅ **Wire Fault Detection:** Predicts wire faults using voltage, current, and resistance values.
✅ **FastAPI Backend:** API endpoints for data processing, inference, and visualization.

## System Architecture
1. **Data Ingestion:** Accepts CSV files (sensor data) and images.
2. **Preprocessing:** Normalizes sensor values, applies PCA, and resizes images.
3. **Model Inference:** Runs LSTM Autoencoder, YOLO object detection, and wire fault classification.
4. **Visualization:** Displays interactive anomaly timelines, error distributions, and feature importance.
5. **API Interface:** FastAPI endpoints enable easy data upload and processing.

## Installation & Setup
### Prerequisites
- Python 3.8+
- FastAPI
- TensorFlow
- ONNXRuntime
- Scikit-learn
- OpenCV
- Pandas & NumPy
- Uvicorn

### Install Dependencies
Run the following command:
```bash
pip install fastapi tensorflow plotly scikit-learn pandas numpy uvicorn onnxruntime opencv-python
```

### Start the Server
```bash
uvicorn main:app --reload
```
Access the **dashboard** at: [http://localhost:8000](http://localhost:8000)

## API Endpoints
| Endpoint            | Method | Description |
|---------------------|--------|-------------|
| `/predict`         | POST   | Detect anomalies in sensor data (CSV) |
| `/detect_damage`   | POST   | Detect dents/cracks in images |
| `/detect_wire_fault` | POST   | Predict wire faults using numerical data |
| `/`               | GET    | Returns the dashboard UI |

## Usage
### 1. **Anomaly Detection** (Time-Series Data)
- Upload a CSV file containing sensor readings.
- The system preprocesses the data, applies PCA, sequences it, and detects anomalies.
- View anomaly timelines and feature importance on the dashboard.

### 2. **Dent/Damage Detection** (Images)
- Upload an image of an industrial component.
- The YOLO model detects dents or cracks and marks them.
- View detected faults on the dashboard.

### 3. **Wire Fault Detection** (Numerical Data)
- Submit voltage, current, and resistance values.
- The model predicts whether the wire is faulty or not.
- Get real-time fault classification results.

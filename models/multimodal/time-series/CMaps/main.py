import io
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
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

# Create necessary directories
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Initialize templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved LSTM autoencoder model
custom_objects = {'mse': MeanSquaredError()}
model_path = 'lstm_autoencoder.h5'
loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
print("[INFO] LSTM Autoencoder loaded.")

# --- Recreate Preprocessors ---
training_data_path = "combined_cmapps_training.csv"
training_data = pd.read_csv(training_data_path)

# Identify feature columns
sensor_cols = [col for col in training_data.columns if col.startswith("sensor")]
op_cols = [col for col in training_data.columns if col.startswith("op_set")]
feature_cols = op_cols + sensor_cols

# Create and fit preprocessors
scaler = StandardScaler()
scaler.fit(training_data[feature_cols])

pca = PCA(n_components=10)
X_scaled = scaler.transform(training_data[feature_cols])
pca.fit(X_scaled)
print("[INFO] Preprocessors ready.")

# --- Data Preparation Functions ---
def prepare_new_data_from_df(new_data_df, scaler, pca, sequence_length=30):
    """
    Load and prepare new data from a DataFrame:
      - Converts feature columns to numeric,
      - Applies scaling and PCA,
      - Creates sequences for LSTM processing.
    """
    # Identify features
    sensor_cols = [col for col in new_data_df.columns if col.startswith('sensor')]
    op_cols = [col for col in new_data_df.columns if col.startswith('op_set')]
    feature_cols = op_cols + sensor_cols

    # Convert feature columns to numeric
    for col in feature_cols:
        new_data_df[col] = pd.to_numeric(new_data_df[col], errors='coerce')
    
    # Scale the features
    X_scaled = scaler.transform(new_data_df[feature_cols])
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # Add back metadata columns if they exist
    for col in ['unit', 'cycle', 'source_file']:
        if col in new_data_df.columns:
            X_scaled_df[col] = new_data_df[col].values

    # Apply PCA reduction
    X_pca = pca.transform(X_scaled_df[feature_cols])
    X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    for col in ['unit', 'cycle', 'source_file']:
        if col in new_data_df.columns:
            X_pca_df[col] = new_data_df[col].values

    # Create sequences per engine unit
    sequences = []
    metadata = []
    for unit in X_pca_df['unit'].unique():
        unit_data = X_pca_df[X_pca_df['unit'] == unit].sort_values('cycle')
        if len(unit_data) >= sequence_length:
            features = unit_data[[f'PC{i+1}' for i in range(pca.n_components_)]].values
            unit_metadata = unit_data[['unit', 'cycle', 'source_file']].values
            for i in range(0, len(features) - sequence_length + 1):
                sequences.append(features[i:i+sequence_length])
                metadata.append(unit_metadata[i+sequence_length-1])
    return np.array(sequences), np.array(metadata), new_data_df

def detect_anomalies(model, sequences, default_threshold=None):
    """
    Detect anomalies using the LSTM autoencoder
    """
    reconstructions = model.predict(sequences)
    mse = np.mean(np.power(sequences - reconstructions, 2), axis=(1, 2))
    if default_threshold is None:
        default_threshold = np.percentile(mse, 95)
    anomalies = (mse > default_threshold).astype(int)
    return anomalies, mse, default_threshold

# --- Visualization Functions ---
def create_anomaly_timeline(metadata, mse, threshold, anomalies):
    """
    Create a timeline visualization of anomalies
    """
    # Create DataFrame for visualization
    timeline_df = pd.DataFrame({
        'Unit': metadata[:, 0].astype(int),
        'Cycle': metadata[:, 1].astype(int),
        'Error': mse,
        'Is Anomaly': ['Anomaly' if a == 1 else 'Normal' for a in anomalies]
    })
    
    # Sort by unit and cycle
    timeline_df = timeline_df.sort_values(['Unit', 'Cycle'])
    
    # Create the figure
    fig = px.scatter(
        timeline_df, 
        x='Cycle', 
        y='Error', 
        color='Is Anomaly',
        hover_data=['Unit', 'Cycle', 'Error'],
        title='Anomaly Detection Timeline',
        color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
        labels={'Cycle': 'Operating Cycle', 'Error': 'Reconstruction Error'}
    )
    
    # Add threshold line
    fig.add_shape(
        type="line",
        x0=timeline_df['Cycle'].min(),
        y0=threshold,
        x1=timeline_df['Cycle'].max(),
        y1=threshold,
        line=dict(color="green", width=2, dash="dash"),
    )
    
    fig.update_layout(
        legend_title="Status",
        xaxis_title="Operating Cycle",
        yaxis_title="Reconstruction Error (MSE)",
        template="plotly_white"
    )
    
    return fig.to_json()

def create_error_distribution(mse, threshold):
    """
    Create a histogram of error distribution with a threshold line.
    
    This function uses Plotly Express to create a histogram of the reconstruction errors (MSE)
    and adds a vertical line for the threshold. If the histogram's y-values are missing, it
    computes the histogram manually using numpy.
    """
    import plotly.express as px

    # Create the histogram figure with a specified number of bins
    fig = px.histogram(
        x=mse,
        nbins=30,
        title='Reconstruction Error Distribution',
        labels={'x': 'Reconstruction Error'},
        color_discrete_sequence=['lightblue'],
        opacity=0.8
    )
    
    # Try to extract y-values from the Plotly figure's first trace
    y_vals = fig.data[0].y
    if y_vals is None or len(y_vals) == 0:
        # If y-values are not available, compute the histogram manually
        counts, bins = np.histogram(mse, bins=30)
        if counts.size > 0:
            y_max = int(counts.max())
        else:
            y_max = 1  # Fallback to 1 if counts are empty
    else:
        y_max = max(y_vals)
    
    # Ensure y_max is not zero (to allow drawing the line)
    if y_max == 0:
        y_max = 1

    # Add a vertical line at the threshold value
    fig.add_shape(
        type="line",
        x0=threshold,
        y0=0,
        x1=threshold,
        y1=y_max,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    # Add an annotation for the threshold
    fig.add_annotation(
        x=threshold,
        y=y_max / 2,
        text=f"Threshold: {threshold:.4f}",
        showarrow=True,
        arrowhead=1,
        ax=50,
        ay=0
    )
    
    fig.update_layout(
        xaxis_title="Reconstruction Error (MSE)",
        yaxis_title="Count",
        template="plotly_white"
    )
    
    return fig.to_json()



def create_unit_health_heatmap(metadata, mse, threshold):
    """
    Create a heatmap showing health status of each unit
    """
    # Create DataFrame for unit health
    health_df = pd.DataFrame({
        'Unit': metadata[:, 0].astype(int),
        'Cycle': metadata[:, 1].astype(int),
        'Error': mse,
        'Health Score': 1 - (mse / (threshold * 2))  # Normalize: 1 is healthy, lower is worse
    })
    
    # Clip health score between 0 and 1
    health_df['Health Score'] = health_df['Health Score'].clip(0, 1)
    
    # Aggregate by unit
    unit_health = health_df.groupby('Unit')['Health Score'].mean().reset_index()
    unit_health['Health Status'] = unit_health['Health Score'].apply(
        lambda x: 'Critical' if x < 0.4 else ('Warning' if x < 0.7 else 'Healthy')
    )
    
    # Sort by health score
    unit_health = unit_health.sort_values('Health Score')
    
    # Create horizontal bar chart
    fig = px.bar(
        unit_health,
        y='Unit',
        x='Health Score',
        color='Health Status',
        title='Unit Health Overview',
        color_discrete_map={
            'Healthy': 'green',
            'Warning': 'orange',
            'Critical': 'red'
        },
        labels={'Unit': 'Engine Unit', 'Health Score': 'Health Score (0-1)'},
        orientation='h'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Health Score (higher is better)",
        yaxis_title="Engine Unit",
        template="plotly_white"
    )
    
    return fig.to_json()

def extract_feature_importance(original_df, mse, top_n=5):
    """
    Create feature importance chart based on correlation with anomaly scores
    """
    # Create DataFrame with MSE and all features
    sensor_cols = [col for col in original_df.columns if col.startswith('sensor')]
    
    # We need to match the sequence data with the original data
    feature_df = original_df.copy()
    
    # If we have more sequences than original data points, we'll just use the latest ones
    if len(mse) <= len(feature_df):
        feature_df = feature_df.iloc[-len(mse):].reset_index(drop=True)
        feature_df['MSE'] = mse
    else:
        # We have fewer MSE values than data points, so we'll just use the ones we have
        feature_df = feature_df.iloc[:len(mse)].reset_index(drop=True)
        feature_df['MSE'] = mse
    
    # Calculate correlation between features and MSE
    correlations = {}
    for col in sensor_cols:
        correlations[col] = abs(feature_df[col].corr(feature_df['MSE']))
    
    # Get top correlated features
    top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Create bar chart
    feature_names = [f[0] for f in top_features]
    correlation_values = [f[1] for f in top_features]
    
    fig = px.bar(
        x=correlation_values,
        y=feature_names,
        orientation='h',
        title=f'Top {top_n} Features Correlated with Anomalies',
        labels={'x': 'Absolute Correlation', 'y': 'Feature'},
        color=correlation_values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Absolute Correlation with Anomaly Score",
        yaxis_title="Feature",
        template="plotly_white"
    )
    
    return fig.to_json()

# --- FastAPI Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serve the main HTML frontend
    """
    # Create the index.html file if it doesn't exist
    index_path = os.path.join("templates", "index.html")
    if not os.path.exists(index_path):
        with open(index_path, "w") as f:
            f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Engine Anomaly Detection Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/plotly.js@2.12.1/dist/plotly.min.js"></script>
    <style>
        .card {
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .chart-container {
            height: 400px;
            width: 100%;
        }
        .alert-panel {
            max-height: 300px;
            overflow-y: auto;
        }
        .severity-high {
            color: #dc3545;
            font-weight: bold;
        }
        .severity-medium {
            color: #fd7e14;
            font-weight: bold;
        }
        .dashboard-header {
            background-color: #343a40;
            color: white;
            padding: 15px 0;
            margin-bottom: 20px;
        }
        .loader {
            border: 16px solid #f3f3f3;
            border-top: 16px solid #3498db;
            border-radius: 50%;
            width: 120px;
            height: 120px;
            animation: spin 2s linear infinite;
            margin: 0 auto;
            display: none;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="dashboard-header">
        <div class="container">
            <h1>Engine Anomaly Detection Dashboard</h1>
            <p class="lead">Upload engine sensor data to detect anomalies and predict failures</p>
        </div>
    </div>

    <div class="container">
        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Upload Engine Sensor Data</h5>
                        <p>Upload a CSV file containing sensor readings from the engines.</p>
                        <form id="upload-form" enctype="multipart/form-data">
                            <div class="mb-3">
                                <input class="form-control" type="file" id="formFile" accept=".csv">
                            </div>
                            <button type="submit" class="btn btn-primary">Analyze Data</button>
                        </form>
                        <div class="loader mt-3" id="loader"></div>
                    </div>
                </div>
            </div>
        </div>

        <div id="results-container" style="display: none;">
            <div class="row mb-4">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Analysis Summary</h5>
                            <div class="row">
                                <div class="col-md-3">
                                    <div class="card bg-light">
                                        <div class="card-body text-center">
                                            <h3 id="anomaly-count">0</h3>
                                            <p>Anomalies Detected</p>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="card bg-light">
                                        <div class="card-body text-center">
                                            <h3 id="total-sequences">0</h3>
                                            <p>Total Sequences</p>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="card bg-light">
                                        <div class="card-body text-center">
                                            <h3 id="anomaly-percentage">0%</h3>
                                            <p>Anomaly Percentage</p>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="card bg-light">
                                        <div class="card-body text-center">
                                            <h3 id="threshold-value">0</h3>
                                            <p>Threshold Value</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Anomaly Timeline</h5>
                            <div id="timeline-chart" class="chart-container"></div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Error Distribution</h5>
                            <div id="error-distribution" class="chart-container"></div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Unit Health Overview</h5>
                            <div id="unit-health" class="chart-container"></div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Feature Importance</h5>
                            <div id="feature-importance" class="chart-container"></div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Detailed Anomaly Alerts</h5>
                            <div class="alert-panel" id="anomaly-details">
                                <!-- Anomaly details will be inserted here -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('upload-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const fileInput = document.getElementById('formFile');
            if (!fileInput.files[0]) {
                alert('Please select a file');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            // Show loader
            document.getElementById('loader').style.display = 'block';
            document.getElementById('results-container').style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData,
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Unknown error occurred');
                }
                
                const result = await response.json();
                displayResults(result);
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('loader').style.display = 'none';
            }
        });
        
        function displayResults(result) {
            // Show results container
            document.getElementById('results-container').style.display = 'block';
            
            // Update summary statistics
            document.getElementById('anomaly-count').textContent = result.summary.anomalies_detected;
            document.getElementById('total-sequences').textContent = result.summary.total_sequences;
            document.getElementById('anomaly-percentage').textContent = result.summary.anomaly_percentage.toFixed(1) + '%';
            document.getElementById('threshold-value').textContent = result.summary.threshold.toFixed(4);
            
            // Display visualizations
            Plotly.newPlot('timeline-chart', JSON.parse(result.visualizations.timeline));
            Plotly.newPlot('error-distribution', JSON.parse(result.visualizations.error_distribution));
            Plotly.newPlot('unit-health', JSON.parse(result.visualizations.unit_health));
            Plotly.newPlot('feature-importance', JSON.parse(result.visualizations.feature_importance));
            
            // Update detail panels
            const anomalyDetailsElem = document.getElementById('anomaly-details');
            anomalyDetailsElem.innerHTML = '';
            
            if (Object.keys(result.unit_details).length === 0) {
                anomalyDetailsElem.innerHTML = '<div class="alert alert-success">No anomalies detected.</div>';
            } else {
                for (const [unit, anomalies] of Object.entries(result.unit_details)) {
                    const unitElement = document.createElement('div');
                    unitElement.className = 'alert alert-warning';
                    
                    const unitHeader = document.createElement('h6');
                    unitHeader.textContent = `Engine Unit ${unit} - ${anomalies.length} anomalies detected`;
                    unitElement.appendChild(unitHeader);
                    
                    const anomalyList = document.createElement('ul');
                    anomalies.forEach(anomaly => {
                        const listItem = document.createElement('li');
                        listItem.innerHTML = `Cycle ${anomaly.cycle}: <span class="severity-${anomaly.severity.toLowerCase()}">${anomaly.severity} severity</span> (Error: ${anomaly.error.toFixed(4)})`;
                        anomalyList.appendChild(listItem);
                    });
                    
                    unitElement.appendChild(anomalyList);
                    anomalyDetailsElem.appendChild(unitElement);
                }
            }
            
            // Resize charts to fit containers
            window.dispatchEvent(new Event('resize'));
        }
    </script>
</body>
</html>
            """)
            print(f"[INFO] Created {index_path}")
            
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Process uploaded data and return anomaly detection results with visualizations
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        data = io.StringIO(contents.decode("utf-8"))
        new_data_df = pd.read_csv(data)
        print(f"[DEBUG] New data shape: {new_data_df.shape}")
        
        # Prepare the data
        sequences, metadata, original_df = prepare_new_data_from_df(new_data_df, scaler, pca, sequence_length=30)
        if len(sequences) == 0:
            return JSONResponse(status_code=400, content={"error": "Not enough data to create sequences."})
        
        # Detect anomalies
        anomalies, mse, threshold = detect_anomalies(loaded_model, sequences)
        
        # Create visualizations
        timeline_chart = create_anomaly_timeline(metadata, mse, threshold, anomalies)
        error_distribution = create_error_distribution(mse, threshold)
        unit_health = create_unit_health_heatmap(metadata, mse, threshold)
        feature_importance = extract_feature_importance(original_df, mse)
        
        # Prepare detailed result with unit-specific information
        units_with_anomalies = {}
        for i in range(len(anomalies)):
            if anomalies[i] == 1:
                unit = int(metadata[i][0])
                cycle = int(metadata[i][1])
                error = float(mse[i])
                if unit not in units_with_anomalies:
                    units_with_anomalies[unit] = []
                units_with_anomalies[unit].append({
                    "cycle": cycle,
                    "error": error,
                    "severity": "High" if error > threshold * 1.5 else "Medium"
                })
        
        # Organize results
        result = {
            "summary": {
                "anomalies_detected": int(anomalies.sum()),
                "total_sequences": int(len(anomalies)),
                "anomaly_percentage": float(anomalies.sum() / len(anomalies) * 100),
                "threshold": float(threshold)
            },
            "unit_details": units_with_anomalies,
            "visualizations": {
                "timeline": timeline_chart,
                "error_distribution": error_distribution,
                "unit_health": unit_health,
                "feature_importance": feature_importance
            }
        }
        
        return result
    except Exception as e:
        print(f"[ERROR] {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
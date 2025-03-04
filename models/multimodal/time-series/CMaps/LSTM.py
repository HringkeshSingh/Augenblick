import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

custom_objects = {
    'mse': tf.keras.losses.MeanSquaredError()
}

# Load the saved model with custom objects
model_path = 'lstm_autoencoder.h5'
loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Function to load and prepare new data
def prepare_new_data(data_path, scaler, pca, sequence_length=30):
    """
    Load and prepare new data for anomaly detection
    """
    # Load data
    new_data = pd.read_csv(data_path)
    
    # Prepare features (similar to preprocessing in the original code)
    sensor_cols = [col for col in new_data.columns if col.startswith('sensor')]
    op_cols = [col for col in new_data.columns if col.startswith('op_set')]
    feature_cols = op_cols + sensor_cols
    
    # Apply the same scaling
    for col in feature_cols:
        new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
    
    X_scaled = scaler.transform(new_data[feature_cols])
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # Add metadata columns back
    for col in ['unit', 'cycle', 'source_file']:
        if col in new_data.columns:
            X_scaled_df[col] = new_data[col].values
    
    # Apply PCA reduction
    X_pca = pca.transform(X_scaled_df[feature_cols])
    X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    
    # Add metadata columns back
    for col in ['unit', 'cycle', 'source_file']:
        if col in new_data.columns:
            X_pca_df[col] = new_data[col].values
    
    # Create sequences
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
    
    return np.array(sequences), np.array(metadata)
    
# Function to detect anomalies
def detect_anomalies(model, sequences, metadata=None, thresholds=None, default_threshold=None):
    """
    Detect anomalies in the sequences using reconstruction error
    """
    # Generate reconstructions
    reconstructions = model.predict(sequences)
    
    # Calculate MSE
    mse = np.mean(np.power(sequences - reconstructions, 2), axis=(1, 2))
    
    # If thresholds provided, use source-specific thresholds
    if thresholds is not None and metadata is not None and len(thresholds) > 0:
        source_file_col_idx = 2  # assuming source_file is the 3rd column in metadata
        anomalies = np.zeros(len(sequences), dtype=int)
        
        # Apply source-specific thresholds
        for metadata_idx, sequence_metadata in enumerate(metadata):
            source = sequence_metadata[source_file_col_idx]
            if source in thresholds:
                anomalies[metadata_idx] = 1 if mse[metadata_idx] > thresholds[source] else 0
            elif default_threshold is not None:
                anomalies[metadata_idx] = 1 if mse[metadata_idx] > default_threshold else 0
    else:
        # Use a global threshold if source-specific ones aren't available
        if default_threshold is None:
            default_threshold = np.percentile(mse, 95)
        anomalies = (mse > default_threshold).astype(int)
    
    return anomalies, mse, default_threshold

# Function to visualize results
def visualize_anomalies(unit_ids, cycles, mse, anomalies, threshold=None):
    """
    Visualize the anomaly detection results
    """
    plt.figure(figsize=(12, 6))
    
    # Plot reconstruction error
    plt.plot(cycles, mse, 'b-', label='Reconstruction Error')
    
    # Mark anomalies
    anomaly_indices = np.where(anomalies == 1)[0]
    plt.scatter(cycles[anomaly_indices], mse[anomaly_indices], color='red', label='Anomalies')
    
    # Draw threshold line if provided
    if threshold:
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
    
    plt.title(f'Anomaly Detection for Unit {unit_ids[0]}')
    plt.xlabel('Cycle')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Alternative approach if you don't have the saved scaler and PCA
    # This recreates them using the original code's approach
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # 1. Load your training data to recreate preprocessors
    training_data_path = "combined_cmapps_training.csv"
    training_data = pd.read_csv(training_data_path)
    
    # 2. Extract features
    sensor_cols = [col for col in training_data.columns if col.startswith('sensor')]
    op_cols = [col for col in training_data.columns if col.startswith('op_set')]
    feature_cols = op_cols + sensor_cols
    
    # 3. Recreate scaler
    scaler = StandardScaler()
    scaler.fit(training_data[feature_cols])
    
    # 4. Recreate PCA
    pca = PCA(n_components=10)
    X_scaled = scaler.transform(training_data[feature_cols])
    pca.fit(X_scaled)
    
    # Now you can use these recreated preprocessors with your new data
    
    # 5. Load and prepare new data for testing
    new_data_path = "combined_cmapps_dataset.csv"
    sequences, metadata = prepare_new_data(new_data_path, scaler, pca)
    
    # 6. Detect anomalies
    # If you don't have source thresholds, you can use a global threshold
    anomalies, mse, threshold = detect_anomalies(loaded_model, sequences)
    
    print(f"Detected {anomalies.sum()} anomalies out of {len(anomalies)} sequences")
    print(f"Global threshold: {threshold:.6f}")
    
    # 7. Plot anomaly detection results for a specific unit
    import matplotlib.pyplot as plt
    
    # Example for one unit
    unit_id = metadata[0, 0]  # First unit ID in the data
    unit_indices = metadata[:, 0] == unit_id
    
    unit_cycles = metadata[unit_indices, 1]
    unit_mse = mse[unit_indices]
    unit_anomalies = anomalies[unit_indices]
    
    plt.figure(figsize=(12, 6))
    plt.plot(unit_cycles, unit_mse, 'b-', label='Reconstruction Error')
    
    # Mark anomalies
    anomaly_indices = np.where(unit_anomalies == 1)[0]
    if len(anomaly_indices) > 0:
        plt.scatter(unit_cycles[anomaly_indices], unit_mse[anomaly_indices], 
                   color='red', label='Anomalies')
    
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.title(f'Anomaly Detection for Unit {unit_id}')
    plt.xlabel('Cycle')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()
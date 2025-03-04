# Step 1: Model Training
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv("./dataset.csv")

# Split features and target variable
X = data[['voltage', 'current', 'resistance']]
y = data['label']


# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model,'Wire_Fault.joblib')

# Step 2: Prediction Function
def predict_wire_status(voltage, current, resistance):

    # Make prediction for the input data
    prediction = model.predict([[voltage, current, resistance]])

    if prediction[0] == 1:
        return "Faulty wire"
    else:
        return "Not a faulty wire"


# Suppress warning about missing feature names
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# Step 3: Interactive Input and Prediction
while True:
    try:
        voltage = float(input("Enter the voltage of the wire: "))
        current = float(input("Enter the current flowing through the wire: "))
        resistance = float(input("Enter the resistance of the wire: "))
        result = predict_wire_status(voltage, current, resistance)
        print("Prediction:", result)      
        break
    except ValueError:
        print("Please enter valid numerical values for voltage, current, and resistance.")
    except KeyboardInterrupt:

        print("\nExiting...")

    break
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.preprocessing import LabelEncoder
import os

# File paths
CRIME_DATA_FILE = "crime_data/reported_crimes.csv"
MODEL_FILE = "models/crime_validation_model.pkl"

def train_and_update_model():
    # Read the crime data
    try:
        df = pd.read_csv(CRIME_DATA_FILE)
    except Exception as e:
        print("Error reading CSV:", e)
        return

    # Check if data is empty
    if df.empty:
        print("No data available for training. Skipping model training.")
        return

    # Recalculate encoded features for the entire dataset
    # Use the index as a simple encoding for location
    df["location_encoded"] = range(len(df))
    
    # Encode incident_type using LabelEncoder
    label_encoder = LabelEncoder()
    df["incident_encoded"] = label_encoder.fit_transform(df["incident_type"].astype(str))

    # Prepare features and labels
    X = df[["location_encoded", "incident_encoded"]]
    y = df["verified"].astype(int)

    # Train the RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    try:
        model.fit(X, y)
    except ValueError as e:
        print("Error during model training:", e)
        return

    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    # Update the 'verified' column using model predictions
    df["verified"] = model.predict(X)
    
    # Save the updated DataFrame back to the CSV file
    df.to_csv(CRIME_DATA_FILE, index=False)

    print("Crime validation model trained & updated successfully.")

if __name__ == "__main__":
    train_and_update_model()

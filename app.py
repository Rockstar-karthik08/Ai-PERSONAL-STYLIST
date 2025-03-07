import os
import random
import pickle
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load dataset
data = pd.read_csv("stylist_data.csv")

# Encode categorical variables
encoder_season = LabelEncoder()
encoder_occasion = LabelEncoder()
encoder_body_type = LabelEncoder()
encoder_clothing_type = LabelEncoder()

data["Season"] = encoder_season.fit_transform(data["Season"])
data["Occasion"] = encoder_occasion.fit_transform(data["Occasion"])
data["Body_Type"] = encoder_body_type.fit_transform(data["Body_Type"])
data["Clothing_Type"] = encoder_clothing_type.fit_transform(data["Clothing_Type"])

# Define features and target
X = data[["Season", "Occasion", "Body_Type"]]
y = data["Clothing_Type"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model and encoders
with open("stylist_model.pkl", "wb") as f:
    pickle.dump((model, encoder_season, encoder_occasion, encoder_body_type, encoder_clothing_type), f)

# Load the trained model and encoders
with open("stylist_model.pkl", "rb") as f:
    model, encoder_season, encoder_occasion, encoder_body_type, encoder_clothing_type = pickle.load(f)

# Pexels API Key (Replace with your own key)
PEXELS_API_KEY = "cbdMZ4fXmxRRBmVUR8MNMVLahFZ30RCapVqI3uz3oYMOL5U9EaJDl6Ry"

def get_random_image(query):
    """Fetch a random clothing image from Pexels."""
    url = f"https://api.pexels.com/v1/search?query={query}&per_page=50"
    headers = {"Authorization": PEXELS_API_KEY}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        results = response.json().get("photos", [])
        if results:
            return random.choice(results)["src"]["original"]
    return "static/images/default.jpg"  # Fallback image if API fails

def log_prediction(season, occasion, body_type, prediction):
    """Log user predictions with timestamp."""
    log_file = "predictions_log.csv"
    log_data = pd.DataFrame([[datetime.now(), season, occasion, body_type, prediction]],
                            columns=["Timestamp", "Season", "Occasion", "Body_Type", "Prediction"])
    
    if os.path.exists(log_file):
        log_data.to_csv(log_file, mode='a', header=False, index=False)  # Append without header
    else:
        log_data.to_csv(log_file, index=False)  # Create file with header

def get_recent_predictions():
    """Retrieve last 5 predictions for display."""
    log_file = "predictions_log.csv"
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        return df.tail(5).to_dict(orient="records")  # Convert to list of dictionaries
    return []

@app.route("/", methods=["GET", "POST"])
def index():
    prediction, image_url = None, None

    if request.method == "POST":
        season = request.form.get("season")
        occasion = request.form.get("occasion")
        body_type = request.form.get("body_type")

        if not all([season, occasion, body_type]):
            return render_template("index.html", error="⚠️ Please fill out all fields.", past_predictions=get_recent_predictions())

        try:
            # Ensure input values exist in encoder classes
            for encoder, value in zip(
                [encoder_season, encoder_occasion, encoder_body_type],
                [season, occasion, body_type]
            ):
                if value not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, value)  # Convert to NumPy array

            # Encode input
            input_data = [
                encoder_season.transform([season])[0],
                encoder_occasion.transform([occasion])[0],
                encoder_body_type.transform([body_type])[0],
            ]

            # Predict clothing type
            prediction_index = model.predict([input_data])[0]
            prediction = encoder_clothing_type.inverse_transform([prediction_index])[0]

            # Log the prediction
            log_prediction(season, occasion, body_type, prediction)

            # Fetch image
            image_url = get_random_image(prediction)

        except ValueError as e:
            return render_template("index.html", error=f"⚠️ Invalid input: {str(e)}", past_predictions=get_recent_predictions())

    return render_template("index.html", prediction=prediction, image_url=image_url, past_predictions=get_recent_predictions())
    
@app.route("/history")
def history():
    return render_template("history.html", past_predictions=get_recent_predictions())
from flask import Response

@app.route("/download_csv")
def download_csv():
    df = pd.read_csv("predictions_log.csv")  # Load the stored history

    csv_data = df.to_csv(index=False)  # Convert to CSV format
    response = Response(csv_data, mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=History.csv"
    return response

if __name__ == "__main__":
    app.run(debug=True)

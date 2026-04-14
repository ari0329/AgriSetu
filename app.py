from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from twilio.rest import Client

app = Flask(__name__)

# ================== LOAD MODELS ==================
crop_model = joblib.load("crop_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

FEATURE_COLUMNS = [
    "Soil_Moisture_%",
    "Soil_Temperature_C",
    "Rainfall_ml",
    "Air_Temperature_C",
    "Humidity_%"
]

# ================== TWILIO CONFIG ==================
account_sid = os.getenv("TWILIO_SID")
auth_token = os.getenv("TWILIO_AUTH")
twilio_number = os.getenv("TWILIO_NUMBER")
user_number = os.getenv("USER_NUMBER")

client = Client(account_sid, auth_token)

# ================== ROUTE ==================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    soil = float(data.get("soil", 0))
    temp = float(data.get("temp", 0))

    # Dummy values (you can improve later)
    rainfall = 100
    air_temp = 30
    humidity = 60

    sample = pd.DataFrame([{
        "Soil_Moisture_%": soil,
        "Soil_Temperature_C": temp,
        "Rainfall_ml": rainfall,
        "Air_Temperature_C": air_temp,
        "Humidity_%": humidity
    }])

    pred = crop_model.predict(sample)
    crop = label_encoder.inverse_transform(pred)[0]

    # ================== TWILIO ALERT ==================
    message_body = f"🌱 AgriSetu Alert\nRecommended Crop: {crop}\nSoil: {soil}"

    client.messages.create(
        body=message_body,
        from_=twilio_number,
        to=user_number
    )

    return jsonify({
        "status": "success",
        "crop": crop,
        "soil": soil,
        "temperature": temp
    })


@app.route("/")
def home():
    return "AgriSetu API Running 🚀"


if __name__ == "__main__":
    app.run(debug=True)
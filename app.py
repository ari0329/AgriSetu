from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# ================== INIT ==================
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

# ================== TWILIO CONFIG (SAFE) ==================
account_sid = os.getenv("TWILIO_SID")
auth_token = os.getenv("TWILIO_AUTH")
twilio_number = os.getenv("TWILIO_NUMBER")
user_number = os.getenv("USER_NUMBER")

twilio_enabled = False

if all([account_sid, auth_token, twilio_number, user_number]):
    try:
        from twilio.rest import Client
        client = Client(account_sid, auth_token)
        twilio_enabled = True
        print("✅ Twilio initialized")
    except Exception as e:
        print("⚠️ Twilio init failed:", e)
else:
    print("⚠️ Twilio credentials missing. SMS disabled.")

# ================== ROUTES ==================
@app.route("/")
def home():
    return "🌱 AgriSetu API Running Successfully!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        soil = float(data.get("soil", 0))
        temp = float(data.get("temp", 0))

        # Dummy environmental values (you can replace later)
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

        # ================== PREDICTION ==================
        pred = crop_model.predict(sample)
        crop = label_encoder.inverse_transform(pred)[0]

        # ================== TWILIO ALERT ==================
        if twilio_enabled:
            message_body = f"🌱 AgriSetu Alert\nRecommended Crop: {crop}\nSoil: {soil}"

            try:
                client.messages.create(
                    body=message_body,
                    from_=twilio_number,
                    to=user_number
                )
                print("📩 SMS sent")
            except Exception as e:
                print("⚠️ SMS failed:", e)

        # ================== RESPONSE ==================
        return jsonify({
            "status": "success",
            "crop": crop,
            "soil": soil,
            "temperature": temp
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# ================== RUN SERVER ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
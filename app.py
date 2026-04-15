from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import os

from thingesp_client import get_sensor_data
from pdf_generator import generate_pdf

app = Flask(__name__)

# Twilio credentials
ACCOUNT_SID = os.getenv("TWILIO_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH")
FROM_WHATSAPP = "whatsapp:+14155238886"  # sandbox

client = Client(ACCOUNT_SID, AUTH_TOKEN)

# ================== WEBHOOK ==================
@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    incoming_msg = request.values.get("Body", "").lower()
    sender = request.values.get("From")

    resp = MessagingResponse()

    if "prediction" in incoming_msg:
        try:
            # 1. Fetch real-time data
            sensor_data = get_sensor_data()

            # 2. Generate PDF
            pdf_file, crop = generate_pdf(sensor_data)

            # 3. Send PDF via Twilio
            message = client.messages.create(
                from_=FROM_WHATSAPP,
                to=sender,
                body=f"🌱 Crop Recommendation: {crop}",
                media_url=[f"{os.getenv('BASE_URL')}/{pdf_file}"]
            )

            return "OK"

        except Exception as e:
            resp.message("❌ Error generating report")
            print(e)
            return str(resp)

    else:
        resp.message("Send 'prediction' to get crop report 🌱")
        return str(resp)

# ================== SERVE FILE ==================
@app.route("/<filename>")
def serve_file(filename):
    return open(filename, "rb").read()

# ================== RUN ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
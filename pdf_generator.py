import joblib
import pandas as pd
from datetime import datetime
import random
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

FEATURE_COLUMNS = [
    "Soil_Moisture_%",
    "Soil_Temperature_C",
    "Rainfall_ml",
    "Air_Temperature_C",
    "Humidity_%"
]

crop_model = joblib.load("crop_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def generate_pdf(sensor_data):
    soil = sensor_data["soil"]
    temp = sensor_data["temp"]

    sample = pd.DataFrame([{
        "Soil_Moisture_%": soil,
        "Soil_Temperature_C": temp,
        "Rainfall_ml": random.uniform(80,150),
        "Air_Temperature_C": random.uniform(20,40),
        "Humidity_%": random.uniform(40,80)
    }])

    pred = crop_model.predict(sample)
    crop = label_encoder.inverse_transform(pred)[0]

    file_name = f"report_{datetime.now().timestamp()}.pdf"

    doc = SimpleDocTemplate(file_name)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("AgriSetu Smart Report", styles["Title"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph(f"Soil: {soil}", styles["Normal"]))
    content.append(Paragraph(f"Temp: {temp}", styles["Normal"]))
    content.append(Paragraph(f"Recommended Crop: {crop}", styles["Normal"]))

    doc.build(content)

    return file_name, crop
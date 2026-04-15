import requests

THINGESP_API = "https://thingesp.com/api/users/Noctum/projects/Agrisetu/webhooks/twilio?token=bBV87AWa"  # replace with your actual endpoint

def get_sensor_data():
    try:
        res = requests.get(THINGESP_API, timeout=5)
        data = res.json()

        return {
            "soil": float(data.get("soil", 50)),
            "temp": float(data.get("temp", 25))
        }
    except Exception as e:
        print("ThingESP Error:", e)
        return {
            "soil": 50,
            "temp": 25
        }
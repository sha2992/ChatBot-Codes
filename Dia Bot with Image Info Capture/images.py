import openai
import base64
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import re
import psycopg
import rag

openai.api_key=os.environ.get("OPENAI_API_KEY")

client = openai.OpenAI()

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to parse data 
def parse_result(result):
    if "Glucose" in result or "mg/dL" in result or "mmol/L" in result:
        match = re.search(r"(\d+\.?\d*)\s*(mg/dL|mmol/L)?", result)
        if match:
            return {
                "device": "glucose",
                "glucose_value": float(match.group(1)),
                "unit": match.group(2) if match.group(2) else None
            }

    elif "SYS" in result and "DIA" in result:
        sys_match = re.search(r"SYS[:\-]?\s*(\d+)\s*(\w+)?", result)
        dia_match = re.search(r"DIA[:\-]?\s*(\d+)\s*(\w+)?", result)
        pul_match = re.search(r"PUL[:\-]?\s*(\d+)\s*(\w+)?", result)

        return {
            "device": "blood_pressure",
            "sys": int(sys_match.group(1)) if sys_match else None,
            "sys_unit": sys_match.group(2) if sys_match and sys_match.group(2) else None,

            "dia": int(dia_match.group(1)) if dia_match else None,
            "dia_unit": dia_match.group(2) if dia_match and dia_match.group(2) else None,

            "pul": int(pul_match.group(1)) if pul_match else None,
            "pul_unit": pul_match.group(2) if pul_match and pul_match.group(2) else None
        }

    return None 

# Storing Data into database
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

def insert_to_db(data,user_id=1):
    try:
        conn = psycopg.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT)
        cur = conn.cursor()

        if data["device"] == "glucose":
            cur.execute(
                "INSERT INTO image_glucose_readings (user_id, glucose_value, unit) VALUES (%s,%s, %s);",
                (user_id, data["glucose_value"], data["unit"])
            )
        elif data["device"] == "blood_pressure":
            cur.execute(
                "INSERT INTO image_blood_pressure_readings (user_id, sys, sys_unit, dia, dia_unit, pul, pul_unit) VALUES (%s, %s, %s,%s, %s, %s,%s);",
                (user_id,data["sys"],data["sys_unit"], data["dia"],data["dia_unit"], data["pul"], data["pul_unit"])
            )

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[DB ERROR] {e}")

def extract_reading(image_path):
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze the image and identify the type of medical device shown.\n\n"
                            "If it's a **Blood Glucose Meter**, extract the numeric reading and its unit (if visible). "
                            "If it's a **Blood Pressure Meter**, extract:\n"
                            "- SYS (systolic unit)\n"
                            "- DIA (diastolic unit)\n"
                            "- PUL (pulse rate unit)\n"
                            "Return result in a compact readable format based on detected device."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high" ############# this can be changed to 'low', would save the cost, by reducing image quality (can cause issue to detect at times) ########## 
                        }
                    }
                ]
            }
        ],
        max_tokens=50
    )

    result = response.choices[0].message.content.strip()

    ################ providing image info to the rag chatbot by sending to that session #########
    session_id = "abc123" ############# for the time being, static, same as the rag chat model session id
                        ### please evaluate whether fetch the session id here or earlier outside the function

    history = rag.get_session_history(session_id)
    ##############################################################################

    history.add_user_message("Uploaded an image")
    history.add_ai_message(f"Image analysis result: {result}")

    data = parse_result(result)
    if data:
        insert_to_db(data)
    return result
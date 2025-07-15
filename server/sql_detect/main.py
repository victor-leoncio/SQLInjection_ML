from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import subprocess
import requests

app = FastAPI()

model = joblib.load('model_rf_word2vec.pkl')
vectorizer = joblib.load('model_word2vec.pkl')

def detectar_sql_injection(sentence):
    words = sentence.split()
    vector = [vectorizer.wv[word] for word in words if word in vectorizer.wv]
    
    if vector:
        input_vector = np.mean(vector, axis=0).reshape(1, -1)
        prediction = model.predict(input_vector)
        return prediction
    else:
        return None

class InputData(BaseModel):
    sentence: str
    user_ip: str

@app.post("/detectar")
async def detectar(data: InputData):
    texto = data.sentence
    ip = data.user_ip
    
    print(texto)
    print(ip)

    if not texto:
        raise HTTPException(status_code=400, detail="Texto não fornecido")
    
    resultado = detectar_sql_injection(texto)
    print(f"DEBUG ##################################### -> IP = {ip}")

    if resultado == 1:
        # SQL Injection detected - notify firewall to block IP
        try:
            firewall_response = requests.post(
                "http://firewall:8080/block-ip",
                json={"ip": ip, "reason": "SQL Injection detected"},
                timeout=5
            )
            if firewall_response.status_code == 200:
                print(f"IP {ip} successfully blocked by firewall")
            else:
                print(f"Failed to block IP {ip} in firewall: {firewall_response.text}")
        except Exception as e:
            print(f"Error communicating with firewall: {str(e)}")
        
        return {"detected": True, "ip_blocked": True}
    else:
        return {"detected": False, "ip_blocked": False}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "sql-injection-detector",
        "model_loaded": model is not None and vectorizer is not None
    }


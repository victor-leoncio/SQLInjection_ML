from flask import Flask, render_template, request
import json
import requests
import time
import subprocess

app = Flask(__name__)
api_sql_injection = "http://sql_detect:7000/detectar"

def block_ip(ip):
    try:
        result = subprocess.run(
            ["iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        return {"message": f"IP {ip} bloqueado com sucesso!"}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Erro ao bloquear o IP: {e.stderr.decode()}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        start_time = time.time()
        user_ip = request.remote_addr
        sql_input = request.form['sql_input']
        
        response = requests.post(api_sql_injection, json={"sentence": sql_input.lower(), "user_ip": user_ip})
        
        dados = response.json()

        if dados["detected"] == 1:
            block_ip(user_ip)
            end_time = time.time()
            result = f"SQL Injection!\n\nTEMPO DA REQUISIÇÃO: {end_time - start_time}\n\nIP: {user_ip}"
        else:
            end_time = time.time()
            result = f"SQL Normal!\n\nTEMPO DA REQUISIÇÃO: {end_time - start_time}\n\nIP: {user_ip}"
        
        return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)

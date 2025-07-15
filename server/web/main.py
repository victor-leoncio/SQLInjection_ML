from flask import Flask, render_template, request, session, redirect, url_for
import logging
import json
import requests
import time
import subprocess
from werkzeug.exceptions import InternalServerError

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'sua-chave-secreta'  # Alterar para produção
api_sql_injection = "http://sql_detect:7000/detectar"

# IP blocking is now handled by the firewall container
# This function is kept for compatibility but doesn't do anything
def block_ip(ip):
    logger.info(f"IP blocking request for {ip} - handled by firewall container")
    return {"message": f"IP {ip} blocking handled by firewall"}

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user_ip = request.remote_addr
        sql_input = f"{username} {password}"
        
        try:
            response = requests.post(
                api_sql_injection,
                json={"sentence": sql_input.lower(), "user_ip": user_ip}
            )
            response.raise_for_status()
            
            dados = response.json()
            
            if dados["detected"] == True:
                # SQL Injection detected - IP already blocked by firewall via API
                result = f"SQL Injection detectado!\nIP: {user_ip}\nEntrada: {sql_input}"
                ip_blocked = dados.get("ip_blocked", False)
                if ip_blocked:
                    result += f"\nIP bloqueado automaticamente pelo firewall"
                return render_template('login.html', attack_detected=True, attack_result=result, is_injection=True)
            
            return render_template('login.html', result="Consulta segura", is_injection=False)
            
        except Exception as e:
            logger.error(f"Erro ao verificar SQL Injection: {str(e)}")
            return render_template('login.html', error="Erro ao processar a solicitação")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    try:
        user_ip = request.remote_addr
        sql_input = request.form['sql_input']
        
        response = requests.post(
            api_sql_injection,
            json={"sentence": sql_input.lower(), "user_ip": user_ip}
        )
        response.raise_for_status()
        
        dados = response.json()
        
        if dados["detected"] == True:
            # SQL Injection detected - IP already blocked by firewall via API
            result = "SQL Injection detectado!"
            ip_blocked = dados.get("ip_blocked", False)
            if ip_blocked:
                result += " - IP bloqueado pelo firewall"
        else:
            result = "Consulta segura"
            
        return render_template('index.html', result=result)
    
    except Exception as e:
        logger.error(f"Erro: {str(e)}")
        return render_template('index.html', error="Ocorreu um erro interno")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

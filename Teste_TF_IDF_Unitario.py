import joblib
import pandas as pd
import time
from monitor import monitorar_recursos

model = joblib.load('modelo_sql_injection_reduced.pkl')
vectorizer = joblib.load('vectorizer_tfidf.pkl')
important_indices = joblib.load('important_indices.pkl')

print("\n################ Teste de Métricas do TF-IDF Unitário (Sem Paralelismo) ################\n")

@monitorar_recursos
def predict():
    cnt_normal = 0;
    cnt_injection = 0;

    sentence = "OR 3409=3409 AND ('pytW' LIKE 'pytW"

    start_time = time.time()
  
    sql_input_vectorized = vectorizer.transform([sentence])
    
    sql_input_reduced = sql_input_vectorized[:, important_indices]
    
    prediction = model.predict(sql_input_reduced)
    
    end_time = time.time()

    print(f"Tempo Demorado: {end_time - start_time}\n")

    if prediction == 1 :
        cnt_injection += 1;
    else:
        cnt_normal += 1;
   
    duration = end_time - start_time

    print(f"SQL Nomal: {cnt_normal}\nSQL Injection: {cnt_injection}\nTotal: {cnt_normal+cnt_injection}")

predict()

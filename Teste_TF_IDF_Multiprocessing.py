import joblib
import pandas as pd
import time
from monitor_multiprocessing import monitorar_recursos
import multiprocessing

model = joblib.load('modelo_sql_injection_reduced.pkl')
vectorizer = joblib.load('vectorizer_tfidf.pkl')
important_indices = joblib.load('important_indices.pkl')

df = pd.read_csv("SQLiV3_Clean.csv")

print("\n################ Teste de Métricas do TF-IDF (Com Multiprocessing) ################\n")

def process_sentence(sentence):
    sql_input_vectorized = vectorizer.transform([sentence])
    sql_input_reduced = sql_input_vectorized[:, important_indices]
    prediction = model.predict(sql_input_reduced)
    return prediction[0]

@monitorar_recursos
def predict():
    num_cores = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_cores) as pool:
        start_time = time.time()
        predictions = pool.map(process_sentence, df['Sentence'])

    cnt_normal = sum(1 for pred in predictions if pred == 0)
    cnt_injection = sum(1 for pred in predictions if pred == 1)

    end_time = time.time()
    duration = end_time - start_time
    
    print(f"SQL Normal: {cnt_normal}\nSQL Injection: {cnt_injection}\nTotal: {cnt_normal + cnt_injection}")

predict()

import joblib
import pandas as pd
import time
import numpy as np
from monitor_multiprocessing import monitorar_recursos
import multiprocessing

model = joblib.load('model_rf_word2vec.pkl')
vectorizer = joblib.load('model_word2vec.pkl')

df = pd.read_csv("SQLiV3_Clean.csv")

print("\n################ Teste de Métricas do Word2Vec (Com Paralelismo) ################\n")


def process_sentence(sentence):
    words = sentence.split()
    vector = [vectorizer.wv[word] for word in words]

    if vector:
        input_vector = np.mean(vector, axis=0).reshape(1, -1)
        prediction = model.predict(input_vector)

        return prediction[0]
    else:
        print(sentence,words)
        print(vectorizer)
        return None

@monitorar_recursos
def predict():
    cnt_normal = 0
    cnt_injection = 0

    start_time = time.time()
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_sentence, df['Sentence'])

    for result in results:
        if result == 1:
            cnt_injection += 1
        elif result == 0:
            cnt_normal += 1

    end_time = time.time()
    duration = end_time - start_time
    print(f"SQL Normal: {cnt_normal}\nSQL Injection: {cnt_injection}\nTotal: {cnt_normal + cnt_injection}")

predict()

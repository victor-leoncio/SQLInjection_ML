import joblib
import pandas as pd
import time
import numpy as np
from monitor import monitorar_recursos

model = joblib.load('model_rf_word2vec.pkl')
vectorizer = joblib.load('model_word2vec.pkl')

print("\n################ Teste de Métricas do Word2Vec (Sem Paralelismo) ################\n")

@monitorar_recursos
def predict():
    cnt_normal = 0;
    cnt_injection = 0;
    sentence = "OR 3409=3409 AND ('pytW' LIKE 'pytW".upper()
    #sentence = "OR 3409=3409 AND ('pytW' LIKE 'pytW".lower()

    start_time = time.time()

    words = sentence.split()
    vector = [vectorizer.wv[word] for word in words if word in vectorizer.wv]
    
    if vector:
        input_vector = np.mean(vector, axis=0).reshape(1, -1)
        prediction = model.predict(input_vector)
        end_time = time.time()

        if prediction == 1 :
            cnt_injection += 1;
        else:
            cnt_normal += 1;
    else:
        print(sentence)

    duration = end_time - start_time
    print(f"Total Demorado: {duration}")
    print(f"SQL Nomal: {cnt_normal}\nSQL Injection: {cnt_injection}\nTotal: {cnt_normal+cnt_injection}")

predict()

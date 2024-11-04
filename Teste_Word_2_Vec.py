import joblib
import pandas as pd
import time
import numpy as np

model = joblib.load('model_rf_word2vec.pkl')
vectorizer = joblib.load('model_word2vec.pkl')

df = pd.read_csv("SQLiV3_Clean.csv")

def predict():
    cnt_normal = 0;
    cnt_injection = 0;

    start_time = time.time()
    for sentence in df['Sentence']:
        words = sentence.split()
        vector = [vectorizer.wv[word] for word in words if word in vectorizer.wv]
        
        if vector:
            input_vector = np.mean(vector, axis=0).reshape(1, -1)
            prediction = model.predict(input_vector)

            if prediction == 1 :
                cnt_injection += 1;
            else:
                cnt_normal += 1;
        else:
            print(sentence)

    end_time = time.time()
    duration = end_time - start_time
    print(f"SQL Nomal: {cnt_normal}\nSQL Injection: {cnt_injection}\nTotal: {cnt_normal+cnt_injection}\nDuração: {duration:.2f}")

predict()

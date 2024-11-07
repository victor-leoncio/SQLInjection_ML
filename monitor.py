import psutil
import os
import time

def monitorar_recursos(func):
    def wrapper(*args, **kwargs):
        pid = os.getpid()
        processo = psutil.Process(pid)
        
        memoria_inicial = processo.memory_info().rss / 1024 / 1024
        cpu_inicial = processo.cpu_percent(interval=1)

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        memoria_final = processo.memory_info().rss / 1024 / 1024
        cpu_final = processo.cpu_percent(interval=0)

        print(f"Tempo de execução: {end_time - start_time:.4f} segundos")
        print(f"Uso de memória inicial: {memoria_inicial:.2f} MB")
        print(f"Uso de memória final: {memoria_final:.2f} MB")
        print(f"Uso de CPU inicial: {cpu_inicial}%")
        print(f"Uso de CPU final: {cpu_final}%\n")

        return result
    
    return wrapper

import psutil
import os
import time
import statistics

def monitorar_recursos(func):
    def wrapper(*args, **kwargs):
        pid = os.getpid()
        processo = psutil.Process(pid)
        
        cpu_inicial = psutil.cpu_percent(interval=1, percpu=True)
        memoria_inicial = processo.memory_info().rss / 1024 / 1024

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        cpu_final = psutil.cpu_percent(interval=0, percpu=True)
        memoria_final = processo.memory_info().rss / 1024 / 1024

        print(f"Tempo de execução: {end_time - start_time:.4f} segundos")
        print(f"Uso de memória inicial: {memoria_inicial:.2f} MB")
        print(f"Uso de memória final: {memoria_final:.2f} MB")
        
        print(f"Uso de CPU inicial por núcleo: {cpu_inicial}")
        print(f"Uso de CPU final por núcleo: {cpu_final}")
        
        cpu_inicial_total = sum(cpu_inicial)
        cpu_final_total = statistics.mean(cpu_final)
        print(f"Uso total de CPU inicial: {cpu_inicial_total}%")
        print(f"Uso total de CPU final: {cpu_final_total}%\n")

        return result
    
    return wrapper

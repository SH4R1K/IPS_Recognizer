import json

import numpy as np
#############################################################################

from sklearn.metrics import f1_score

# predicted_labels - предсказанные метки (классы)
# true_labels - истинные метки
def calculate_f1_score(predicted_labels, true_labels):
    # Взвешенная F1-мера
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    return f1

def calculate_mq(wer, f1_score):
    # Формула для расчета M_q
    mq = 0.25 * (1 - wer) + 0.75 * f1_score
    return mq

#############################################################################
from jiwer import wer
def calculate_wer(predicted_text, reference_text):
    return wer(reference_text, predicted_text)

#############################################################################

# Начало отслеживания использования памяти

import tracemalloc

tracemalloc.start()

# Вызов вашей модели (например, model.predict())
# predicted_output = model.predict(input_data)

# Оценка пикового потребления памяти
current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory usage: {peak / 10**6} MB")

# Остановка отслеживания
tracemalloc.stop()

##############################################################################

import time

# Пример оценки времени выполнения
start_time = time.time()

# Вызов вашей модели (например, model.predict())
# predicted_output = model.predict(input_data)

end_time = time.time()
latency = end_time - start_time
print(f"Latency: {latency} seconds")

def convert_numpy_types(data):
    if isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, np.int64):  # Преобразуем int64 в int
        return int(data)
    return data

def load_ground_truth(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
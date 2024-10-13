import json
import tracemalloc  # Для отслеживания использования памяти
import time  # Для замера времени выполнения

import librosa  # Для работы с аудиофайлами
import numpy as np
from jiwer import wer  # Для расчета метрики WER (Word Error Rate)
from sklearn.metrics import f1_score  # Для расчета F1-score
from tensorflow.keras.models import load_model  # Для загрузки обученной модели
import src.metrics  # Пользовательские метрики
from src import metrics


class MLModel:
    # Инициализируем модель и создаем словарь для маппинга меток
    model = None
    label_to_index = {
        "отказ": 0,
        "отмена": 1,
        "подтверждение": 2,
        "начать осаживание": 3,
        "осадить на (количество) вагон": 4,
        "продолжаем осаживание": 5,
        "зарядка тормозной магистрали": 6,
        "вышел из межвагонного пространства": 7,
        "продолжаем роспуск": 8,
        "растянуть автосцепки": 9,
        "протянуть на (количество) вагон": 10,
        "отцепка": 11,
        "назад на башмак": 12,
        "захожу в межвагонное пространство": 13,
        "остановка": 14,
        "вперед на башмак": 15,
        "сжать автосцепки": 16,
        "назад с башмака": 17,
        "тише": 18,
        "вперед с башмака": 19,
        "прекратить зарядку тормозной магистрали": 20,
        "тормозить": 21,
        "отпустить": 22,
    }

    def __init__(self, model_file):
        # Загружаем модель при инициализации
        self.model = self.load_model(model_file)

    def load_model(self, model_file):
        # Метод загрузки модели с обработкой исключений
        try:
            return load_model(model_file)
        except Exception as e:
            print(e)

    def predict(self, file_path):
        # Выполняем предсказание для аудиофайла
        predicted_label = self.predict_audio_file(file_path, self.model, self.label_to_index)
        index_to_label = {v: k for k, v in self.label_to_index.items()}  # Обратный словарь
        predicted_text = index_to_label[predicted_label] if predicted_label in index_to_label else "Unknown"
        return {"predicted_label": str(predicted_label), "predicted_text": predicted_text}

    def predict_audio_file(self, file_path, model, label_to_index):
        # Загружаем и обрабатываем аудиофайл
        spectrogram = self.load_audio_file(file_path)
        spectrogram = np.expand_dims(spectrogram, axis=-1)  # Добавляем канал для модели
        spectrogram = np.expand_dims(spectrogram, axis=0)  # Добавляем размерность для батча

        # Выполняем предсказание и определяем класс
        prediction = model.predict(spectrogram)
        predicted_class = np.argmax(prediction[0])

        return predicted_class

    def load_audio_file(self, file_path, n_mels=128, fixed_length=128):
        # Загружаем аудио и конвертируем его в мел-спектрограмму
        audio, sr = librosa.load(file_path, sr=None)
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

        # Добавляем паддинг или обрезаем до фиксированной длины
        if log_spectrogram.shape[1] < fixed_length:
            padding = np.zeros((n_mels, fixed_length - log_spectrogram.shape[1]))
            log_spectrogram = np.concatenate((log_spectrogram, padding), axis=1)
        else:
            log_spectrogram = log_spectrogram[:, :fixed_length]

        return log_spectrogram

    def evaluate_model(self, ground_truth_data, basePath='./src/data/hr_bot_merged/'):
        # Инициализируем списки для хранения истинных и предсказанных меток, а также метрик
        y_true_labels = []
        y_pred_labels = []
        wer_scores = []

        latency_times = []  # Список для времени задержки
        memory_usages = []  # Список для использования памяти

        for entry in ground_truth_data:
            audio_filepath = entry['audio_filepath']
            true_text = entry['text']
            true_label = entry['label']
            true_attribute = entry['attribute']

            # Измеряем время начала инференса
            start_time = time.time()
            tracemalloc.start()  # Начинаем отслеживание памяти

            # Выполняем предсказание
            predicted_label = self.predict_audio_file(basePath + audio_filepath, self.model, self.label_to_index)

            # Останавливаем замеры времени и памяти
            latency = time.time() - start_time
            memory_info = tracemalloc.get_traced_memory()[1] / 1024  # Использование памяти в КБ

            # Оценка WER для текста
            pred_text = self.label_to_index.get(predicted_label, "Unknown")
            wer_score = wer(true_text, pred_text)

            # Оценка F1-score
            y_true_labels.append(true_label)
            y_pred_labels.append(predicted_label)

            latency_times.append(latency)
            memory_usages.append(memory_info)

            tracemalloc.stop()  # Останавливаем отслеживание памяти

            wer_scores.append(wer_score)

        # Подсчет средних значений WER, F1-score, времени задержки и использования памяти
        average_wer = np.mean(wer_scores)
        f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

        avg_latency = np.mean(latency_times)
        avg_memory = np.max(memory_usages)

        # Возвращаем метрики и качество модели
        return {
            'average_wer': average_wer,
            'f1_score': f1,
            'average_latency': avg_latency,
            'peak_memory_usage': avg_memory,
            'mq': metrics.calculate_mq(average_wer, f1)  # Расчет пользовательской метрики MQ
        }

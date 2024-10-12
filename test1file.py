import json
import librosa
import numpy as np
from tensorflow.keras.models import load_model  # Импортируем нужный метод из Keras

# Загружаем модель Keras
model = load_model('voice_command_model.h5')  # Используем load_model для загружки модели
print("Модель загружена!")

# Функция для извлечения признаков
def extract_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Функция для классификации аудиофайла
def classify_audio(file_path):
    # Загружаем аудиофайл
    audio, sr = librosa.load(file_path)

    # Извлекаем признаки
    features = extract_features(audio, sr).reshape(1, -1)  # Добавляем размерность

    # Предсказание класса
    prediction = model.predict(features)

    # Создаем JSON-объект с результатами
    result = {
        "audio_filepath": file_path,
        "predicted_label": int(np.argmax(prediction))  # Получаем индекс класса с максимальной вероятностью
    }

    return result

# Пример использования
audio_file = 'data/audio/hr_bot_clear/891bcb06-76ff-11ee-9082-c09bf4619c03.mp3'  # Замените на ваш путь к аудиофайлу
result = classify_audio(audio_file)

# Сохраним результат в JSON файл
with open('classification_result.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print("Результат классификации сохранен в 'classification_result.json'")

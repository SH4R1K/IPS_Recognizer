import json
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import os
from model_speechRecognation import process_audio

# Словарь для маппинга числовых индексов на текстовые описания
label_mapping = {
    0: "отказ",
    1: "отмена",
    2: "подтверждение",
    3: "начать осаживание",
    4: "осадить на (количество) вагон",
    5: "продолжаем осаживание",
    6: "зарядка тормозной магистрали",
    7: "вышел из межвагонного пространства",
    8: "продолжаем роспуск",
    9: "растянуть автосцепки",
    10: "протянуть на (количество) вагон",
    11: "отцепка",
    12: "назад на башмак",
    13: "захожу в межвагонное,пространство",
    14: "остановка",
    15: "вперед на башмак",
    16: "сжать автосцепки",
    17: "назад с башмака",
    18: "тише",
    19: "вперед с башмака",
    20: "прекратить зарядку тормозной магистрали",
    21: "тормозить",
    22: "отпустить"
}

# Функция для предсказания метки нового аудиофайла
def predict_audio_file(file_path, model):
    audio, sr = librosa.load(file_path, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    if log_spectrogram.shape[1] < 128:
        padding = np.zeros((128, 128 - log_spectrogram.shape[1]))
        log_spectrogram = np.concatenate((log_spectrogram, padding), axis=1)
    else:
        log_spectrogram = log_spectrogram[:, :128]

    log_spectrogram = np.expand_dims(log_spectrogram, axis=-1)
    log_spectrogram = np.expand_dims(log_spectrogram, axis=0)  # Добавляем размерность для батча

    prediction = model.predict(log_spectrogram)
    predicted_label = np.argmax(prediction[0])

    return predicted_label

# Загрузка модели
model = load_model('my_module.h5')

# Пример использования
new_audio_file = '4.mp3'  # Укажите путь к вашему аудиофайлу
predicted_label = predict_audio_file(new_audio_file, model)

if (predicted_label == 4 or predicted_label == 10):
    input_file_path = new_audio_file
    recognized_text, recognized_attribute = process_audio(input_file_path)

    result = {
        "audio_filepath": new_audio_file,
        "id": os.path.splitext(os.path.basename(new_audio_file))[0],
        "label_text": str(recognized_text),
        "label_index": str(predicted_label),
        "attribute": str(recognized_attribute),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
else:
    # Формирование результата в JSON
    result = {
        "audio_filepath": new_audio_file,
        "id": os.path.splitext(os.path.basename(new_audio_file))[0],
        "label_text": label_mapping.get(predicted_label, "Неизвестный класс"),
        "label_index": str(predicted_label),
        "attribute": -1
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

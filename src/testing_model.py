import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.io import wavfile
import noisereduce as nr
import matplotlib.pyplot as plt
def record_audio(file_name, duration, fs=44100):
    print("Recording...")
    record_voice = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until the recording is finished
    write(file_name, fs, record_voice)  # Save as WAV file
    print("Recording completed. Saved to", file_name)

def predict_from_recording(model, label_to_index, recording_file):
    predicted_label = predict_audio_file(recording_file, model, label_to_index)
    print(f"Predicted label from recording: {predicted_label}")
# Функция для загрузки и преобразования аудио в спектрограмму
def load_audio_file(file_path, n_mels=128, fixed_length=128):
    audio, sr = librosa.load(file_path, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Паддинг или обрезка
    if log_spectrogram.shape[1] < fixed_length:
        padding = np.zeros((n_mels, fixed_length - log_spectrogram.shape[1]))
        log_spectrogram = np.concatenate((log_spectrogram, padding), axis=1)
    else:
        log_spectrogram = log_spectrogram[:, :fixed_length]

    return log_spectrogram


# Функция для предсказания метки нового аудиофайла
def predict_audio_file(file_path, model, label_to_index):
    spectrogram = load_audio_file(file_path)
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Добавляем размерность для батча

    prediction = model.predict(spectrogram)
    predicted_class = np.argmax(prediction[0])

    return label_to_index[predicted_class]


# Основная функция для тестирования модели
def test_model(model_path, new_audio_files, label_to_index):
    # Загружаем модель
    try:
        model = load_model(model_path)
        print("Модель успешно загружена")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {str(e)}")
        return

    # Тестирование на новых файлах
    for file_path in new_audio_files:
        predicted_label = predict_audio_file(file_path, model, label_to_index)
        print(f"\nФайл: {file_path}")
        print(f"Прогнозированная метка: {predicted_label}")


# Основной блок выполнения
if __name__ == "__main__":
    # Путь к обученной модели
    model_path = 'my_module.h5'

    # Список путей к новым аудиофайлам для тестирования
    new_audio_files = [
        'data/audio/test_files/1osaditna.wav',
        'data/audio/test_files/1potanytna.mp3',
        'data/audio/test_files/1tishe.wav',
        'data/audio/test_files/2potanytna.mp3',
        'data/audio/test_files/2osaditna.wav',
        'data/audio/test_files/2tishe.wav',
        'data/audio/test_files/gleb_mezhvagon.mp3',
        'data/audio/test_files/otkaz.wav',
        'data/audio/test_files/3osaditna.mp3',
        'data/audio/test_files/3tishe.mp3',
        'data/audio/test_files/3zahojyvmezhvagon.mp3',
        'data/audio/test_files/4prodoljaenosajivanie.mp3',
        'data/audio/test_files/5sjatavtoscepki.mp3',
        'data/audio/test_files/6prodoljaenosajivanie.mp3',
        'data/audio/test_files/7osaditna.mp3',
    ]

    # Загружаем словарь меток (должен быть тот же, что был использован при обучении)
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
        "захожу в межвагонное,пространство": 13,
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

    # Инвертированный словарь для получения имени метки по индексу
    index_to_label = {index: label for label, index in label_to_index.items()}

    # Запуск тестирования
    test_model(model_path, new_audio_files, index_to_label)

while True:
    record_time = int(input("Enter the recording time in seconds: "))
    recording_file = 'recorded_audio.wav'
    record_audio(recording_file, record_time)
    rate, data = wavfile.read(recording_file)
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(recording_file, rate, reduced_noise)

    # Load the model and predict label from the recorded audio
    model = load_model(model_path)
    predict_from_recording(model, index_to_label, recording_file)
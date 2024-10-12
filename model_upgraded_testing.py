import json
import numpy as np
import librosa
from tensorflow.keras.models import load_model

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
model = load_model('speech_command_model.h5')

# Пример использования
new_audio_file = ''  # Укажите путь к вашему аудиофайлу
predicted_label = predict_audio_file(new_audio_file, model)

# Формирование результата в JSON
result = {
    "predicted_label": str(predicted_label),
}
print(json.dumps(result, ensure_ascii=False, indent=2))

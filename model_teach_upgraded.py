import json
import os
import numpy as np
import librosa
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Папка с аудиофайлами
audio_folder = 'data/audio/hr_bot_clear/'

# Загрузка данных
def load_data(json_path, n_mels=128, fixed_length=128):
    X = []
    y = []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for entry in data:
            audio_filepath = os.path.join(audio_folder, entry['audio_filepath'])
            audio, sr = librosa.load(audio_filepath, sr=None)

            spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
            log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

            if log_spectrogram.shape[1] < fixed_length:
                padding = np.zeros((n_mels, fixed_length - log_spectrogram.shape[1]))
                log_spectrogram = np.concatenate((log_spectrogram, padding), axis=1)
            else:
                log_spectrogram = log_spectrogram[:, :fixed_length]

            X.append(log_spectrogram)
            y.append(entry['label'])

    X = np.array(X)
    y = np.array(y)

    y = to_categorical(y)

    return X, y

# Определение входной формы и количества классов
json_path = 'data/annotations/hr_bot_clear.json'
X, y = load_data(json_path)

input_shape = (X.shape[1], X.shape[2], 1)  # Высота, ширина, каналы
num_classes = y.shape[1]

# Создание и обучение модели
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=32, epochs=10, validation_split=0.2)

# Сохранение модели
model.save('speech_command_model.h5')
print("Модель сохранена в 'speech_command_model.h5'")

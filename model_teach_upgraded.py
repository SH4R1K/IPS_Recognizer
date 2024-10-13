# import json
# import os
# import numpy as np
# import librosa
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# import matplotlib.pyplot as plt  # Импорт библиотеки для визуализации
#
# # Папка с аудиофайлами
# audio_folder = 'data/audio/hr_bot_clear/'
#
# # Загрузка данных
# def load_data(json_path, n_mels=128, fixed_length=128):
#     X = []
#     y = []
#
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         for entry in data:
#             audio_filepath = os.path.join(audio_folder, entry['audio_filepath'])
#             audio, sr = librosa.load(audio_filepath, sr=None)
#
#             spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
#             log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
#
#             if log_spectrogram.shape[1] < fixed_length:
#                 padding = np.zeros((n_mels, fixed_length - log_spectrogram.shape[1]))
#                 log_spectrogram = np.concatenate((log_spectrogram, padding), axis=1)
#             else:
#                 log_spectrogram = log_spectrogram[:, :fixed_length]
#
#             X.append(log_spectrogram)
#             y.append(entry['label'])
#
#     X = np.array(X)
#     y = np.array(y)
#
#     y = to_categorical(y)
#
#     return X, y
#
# # Определение входной формы и количества классов
# json_path = 'data/annotations/hr_bot_clear.json'
# X, y = load_data(json_path)
#
# input_shape = (X.shape[1], X.shape[2], 1)  # Высота, ширина, каналы
# num_classes = y.shape[1]
#
# # Создание и обучение модели
# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# history = model.fit(X, y, batch_size=32, epochs=10, validation_split=0.2)
#
# # Сохранение модели
# model.save('my_module.h5')
# print("Модель сохранена")
#
# # Визуализация результатов
# plt.figure(figsize=(12, 5))
#
# # График потерь
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Loss (train)')
# plt.plot(history.history['val_loss'], label='Loss (val)')
# plt.title('Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# # График точности
# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], label='Accuracy (train)')
# plt.plot(history.history['val_accuracy'], label='Accuracy (val)')
# plt.title('Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
#
# # Показать графики
# plt.tight_layout()
# plt.show()


import json
import os
import numpy as np
import librosa
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

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
input_shape = (X.shape[1], X.shape[2], 1)
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
history = model.fit(X, y, batch_size=32, epochs=10, validation_split=0.2)

# Сохранение модели
model.save('my_module.h5')
print("Модель сохранена")

# Загрузка тестовых данных
json_test_path = 'data/annotations/test_data.json'
X_test, y_test = load_data(json_test_path)

# Оценка модели на тестовых данных
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Предсказание на тестовых данных
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Построение графика с предсказаниями
plt.figure(figsize=(10, 5))
plt.scatter(true_classes, predicted_classes, alpha=0.5)
plt.title('True Classes vs Predicted Classes')
plt.xlabel('True Classes')
plt.ylabel('Predicted Classes')
plt.xticks(range(num_classes))  # Отображение всех классов
plt.yticks(range(num_classes))
plt.plot([0, num_classes - 1], [0, num_classes - 1], color='red', linestyle='--')  # Линия идеального соответствия
plt.grid()
plt.show()

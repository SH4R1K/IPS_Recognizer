# import json
# import os
# import librosa
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report
# import joblib
# from tensorflow.python.ops.gen_batch_ops import batch
#
# # Папка с аудиофайлами
# audio_folder = 'data/audio/hr_bot_clear/'
#
# # Списки для аудиофайлов и меток
# audio_files = []
# labels = []
#
# # Папка с аннотациями
# annotation_files = [
#     'data/annotations/hr_bot_clear.json',
# ]
#
# # Загрузка данных из всех файлов аннотаций
# for annotation_file in annotation_files:
#     with open(annotation_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)  # Читаем JSON файл
#
#         for entry in data:  # Проходим по записям
#             audio_filepath = os.path.join(audio_folder, entry['audio_filepath'])  # Полный путь к аудиофайлу
#             label = entry['label']  # Класс команды
#
#             # Загрузить аудиофайл
#             audio, sr = librosa.load(audio_filepath)
#             audio_files.append(audio)
#             labels.append(label)
#
# # Убедимся, что данные загружены
# print(f"Загружено {len(audio_files)} аудиофайлов с {len(set(labels))} уникальными метками.")
#
# X_train, X_test, y_train, y_test = train_test_split(audio_files, labels, test_size=0.2, random_state=42)
# print(f"Обучающая выборка: {len(X_train)} аудиофайлов")
# print(f"Тестовая выборка: {len(X_test)} аудиофайлов")
#
# # Функция для извлечения признаков
# def extract_features(audio):
#     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
#     return np.mean(mfccs.T, axis=0)
#
# # Подготовка признаков
# X_train_features = [extract_features(audio) for audio in X_train]
# X_test_features = [extract_features(audio) for audio in X_test]
#
# # Обучение модели
# model = SVC(kernel='linear')
# model.fit(X_train_features, y_train, epochs=10)
# print("Модель обучена!")
#
# # Оценка модели
# y_pred = model.predict(X_test_features)
# print(classification_report(y_test, y_pred))
#
# # Сохранение модели
# joblib.dump(model, 'voice_command_model.pkl')
# print("Модель сохранена!")

#ПОМОГИТЕ2
# import json
# import os
# import librosa
# import numpy as np
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras import layers, models
# import joblib
#
# # Папка с аудиофайлами
# audio_folder = 'data/audio/hr_bot_clear/'
#
# # Списки для аудиофайлов и меток
# audio_files = []
# labels = []
#
# # Загрузка данных
# annotation_files = [
#     'data/annotations/hr_bot_clear.json',
# ]
#
# for annotation_file in annotation_files:
#     with open(annotation_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         for entry in data:
#             audio_filepath = os.path.join(audio_folder, entry['audio_filepath'])
#             label = entry['label']
#
#             audio, sr = librosa.load(audio_filepath)
#             audio_files.append(audio)
#             labels.append(label)
#
# print(f"Загружено {len(audio_files)} аудиофайлов с {len(set(labels))} уникальными метками.")
#
# X_train, X_test, y_train, y_test = train_test_split(audio_files, labels, test_size=0.2, random_state=42)
#
# # Функция для извлечения признаков
# def extract_features(audio):
#     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
#     return np.mean(mfccs.T, axis=0)
#
# X_train_features = np.array([extract_features(audio) for audio in X_train])
# X_test_features = np.array([extract_features(audio) for audio in X_test])
#
# # Преобразование меток в категории
# y_train = tf.keras.utils.to_categorical(y_train)
# y_test = tf.keras.utils.to_categorical(y_test)
#
# # Создание модели
# model = models.Sequential()
# model.add(layers.Dense(64, activation='relu', input_shape=(X_train_features.shape[1],)))
# model.add(layers.Dense(len(set(labels)), activation='softmax'))
#
# # Компиляция модели
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Обучение модели с указанием количества эпох
# model.fit(X_train_features, y_train, epochs=10)
#
# # Оценка модели
# test_loss, test_acc = model.evaluate(X_test_features, y_test)
# print(f'Точность на тестовом наборе: {test_acc * 100:.2f}%')
#
# # Сохранение модели
# model.save('voice_command_model.h5')
# print("Модель сохранена!")


# Уже лучше
# import json
# import os
# import librosa
# import numpy as np
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras import layers, models
#
# # Папка с аудиофайлами
# audio_folder = 'data/audio/hr_bot_clear/'
#
# # Списки для аудиофайлов и меток
# audio_files = []
# labels = []
#
# # Загрузка данных
# annotation_files = [
#     'data/annotations/hr_bot_clear.json',
# ]
#
# for annotation_file in annotation_files:
#     with open(annotation_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         for entry in data:
#             audio_filepath = os.path.join(audio_folder, entry['audio_filepath'])
#             label = entry['label']
#
#             audio, sr = librosa.load(audio_filepath,
#                                      sr=None)  # Указать, чтобы использовать оригинальную частоту дискретизации
#             audio_files.append(audio)
#             labels.append(label)
#
# print(f"Загружено {len(audio_files)} аудиофайлов с {len(set(labels))} уникальными метками.")
#
# # Преобразование меток в числовой формат
# label_to_index = {label: index for index, label in enumerate(set(labels))}
# y_numeric = [label_to_index[label] for label in labels]
#
# # Разделение на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test = train_test_split(audio_files, y_numeric, test_size=0.2, random_state=42)
#
#
# # Функция для извлечения признаков
# def extract_features(audio, sr):
#     # Паддинг или обрезка до фиксированной длины (например, 22050)
#     if len(audio) < 22050:
#         audio = np.pad(audio, (0, 22050 - len(audio)), 'constant')
#     else:
#         audio = audio[:22050]
#
#     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
#     return np.mean(mfccs.T, axis=0)
#
#
# X_train_features = np.array([extract_features(audio, sr) for audio in X_train])
# X_test_features = np.array([extract_features(audio, sr) for audio in X_test])
#
# # Преобразование меток в категории
# y_train = tf.keras.utils.to_categorical(y_train)
# y_test = tf.keras.utils.to_categorical(y_test)
#
# # Создание модели
# model = models.Sequential()
# model.add(layers.Dense(64, activation='relu', input_shape=(X_train_features.shape[1],)))
# model.add(layers.Dense(len(set(labels)), activation='softmax'))
#
# # Компиляция модели
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Обучение модели с указанием количества эпох
# model.fit(X_train_features, y_train, epochs=100, validation_split=0.2)  # Добавлен валидационный набор
#
# # Оценка модели
# test_loss, test_acc = model.evaluate(X_test_features, y_test)
# print(f'Точность на тестовом наборе: {test_acc * 100:.2f}%')
#
# # Сохранение модели
# model.save('voice_command_model.h5')
# print("Модель сохранена!")

import numpy as np
import librosa
import json
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Папка с аудиофайлами
audio_folder = 'data/audio/hr_bot_clear/'


# 1. Загрузка и преобразование аудио в спектрограммы
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


# 2. Подготовка данных для обучения
def prepare_data(json_path, n_mels=128, fixed_length=128):
    X = []
    y = []

    for annotation_file in json_path:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                audio_filepath = os.path.join(audio_folder, entry['audio_filepath'])
                spectrogram = load_audio_file(audio_filepath, n_mels=n_mels, fixed_length=fixed_length)
                spectrogram = np.expand_dims(spectrogram, axis=-1)  # Добавляем размерность для канала
                X.append(spectrogram)
                y.append(entry['label'])

    X = np.array(X)
    y = np.array(y)

    # Преобразование меток в формат one-hot encoding
    label_to_index = {label: index for index, label in enumerate(set(y))}
    y_numeric = [label_to_index[label] for label in y]
    y = to_categorical(y_numeric)

    return X, y, label_to_index


# 3. Создание модели
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# 4. Обучение модели
def train_model(json_path, n_mels=128, fixed_length=128):
    X, y, label_to_index = prepare_data(json_path, n_mels, fixed_length)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    num_classes = y_train.shape[1]
    model = create_model(input_shape, num_classes)

    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))
    model.save('hybrid_voice_command_model.h5')

    return model, label_to_index


# Функция для предсказания метки нового аудиофайла
def predict_audio_file(file_path, model, label_to_index):
    spectrogram = load_audio_file(file_path)
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Добавляем размерность для батча

    prediction = model.predict(spectrogram)
    predicted_class = np.argmax(prediction[0])

    index_to_label = {index: label for label, index in label_to_index.items()}

    return index_to_label[predicted_class]


# Основная функция для выполнения всего процесса
def main():
    # Путь к файлу с аннотациями
    json_path = ['data/annotations/hr_bot_clear.json']

    # Обучение модели
    model, label_to_index = train_model(json_path)

    # Оценка модели на валидационном наборе
    X_val, y_val = prepare_data(json_path)[0], prepare_data(json_path)[1]
    test_loss, test_acc = model.evaluate(X_val, y_val)
    print(f'Точность на валидационном наборе: {test_acc * 100:.2f}%')

    # Тестирование на новых аудиофайлах
    new_audio_file = '123.mp3'
    predicted_label = predict_audio_file(new_audio_file, model, label_to_index)
    print(f"Прогнозированная метка для нового файла: {predicted_label}")


if __name__ == "__main__":
    main()


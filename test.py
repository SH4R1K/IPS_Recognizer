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
import json
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib

# Папка с аудиофайлами
audio_folder = 'data/audio/hr_bot_clear/'

# Списки для аудиофайлов и меток
audio_files = []
labels = []

# Загрузка данных
annotation_files = [
    'data/annotations/hr_bot_clear.json',
]

for annotation_file in annotation_files:
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for entry in data:
            audio_filepath = os.path.join(audio_folder, entry['audio_filepath'])
            label = entry['label']

            audio, sr = librosa.load(audio_filepath)
            audio_files.append(audio)
            labels.append(label)

print(f"Загружено {len(audio_files)} аудиофайлов с {len(set(labels))} уникальными метками.")

X_train, X_test, y_train, y_test = train_test_split(audio_files, labels, test_size=0.2, random_state=42)

# Функция для извлечения признаков
def extract_features(audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

X_train_features = np.array([extract_features(audio) for audio in X_train])
X_test_features = np.array([extract_features(audio) for audio in X_test])

# Преобразование меток в категории
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Создание модели
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(X_train_features.shape[1],)))
model.add(layers.Dense(len(set(labels)), activation='softmax'))

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели с указанием количества эпох
model.fit(X_train_features, y_train, epochs=10)

# Оценка модели
test_loss, test_acc = model.evaluate(X_test_features, y_test)
print(f'Точность на тестовом наборе: {test_acc * 100:.2f}%')

# Сохранение модели
model.save('voice_command_model.h5')
print("Модель сохранена!")
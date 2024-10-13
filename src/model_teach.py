import json
import os
import numpy as np
import librosa
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
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
# Папка с аудиофайлами
audio_folder = 'data/hr_bot_merged/'
def visualize_predictions(X_val, y_val, y_pred, label_to_index):
    num_samples = len(X_val)

    plt.figure(figsize=(10, 6))

    for i in range(num_samples):
        actual_class = np.argmax(y_val[i])
        predicted_class = np.argmax(y_pred[i])

        # Используем индекс как x-координату, предсказанное значение как y
        plt.scatter(x=[i], y=[predicted_class], color='b', alpha=0.5, label='Predicted class' if i == 0 else "")

        # Отображаем фактическое значение
        plt.scatter(x=[i], y=[actual_class], color='r', alpha=0.5, linestyle='--', label='Actual class' if i == 0 else "")

    # Добавляем легенду и отображаем график
    plt.title('Comparison of Actual and Predicted Classes')
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
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
json_path = 'data/hr_bot_merged.json'
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

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=80)

y_pred = model.predict(X_val)
visualize_predictions(X_val, y_val, y_pred, label_to_index)

# Сохранение модели
model.save('my_module.h5')
print("Модель сохранена")
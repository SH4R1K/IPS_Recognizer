import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import librosa
import matplotlib.pyplot as plt

# Константы
SAMPLE_RATE = 16000
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
DURATION = 60  # Длительность аудио в секундах


def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(audio) < SAMPLE_RATE * DURATION:
        audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)))
    return librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT)


def prepare_data(data_dir):
    X = []
    y = []
    for file in os.listdir(data_dir):
        if file.endswith('.wav'):
            audio_path = os.path.join(data_dir, file)
            audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)

            # Calculate duration
            duration = len(audio) / SAMPLE_RATE

            # Pad audio if necessary
            if duration < DURATION:
                audio = np.pad(audio, (0, int(DURATION * SAMPLE_RATE - len(audio))))

            mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS,
                                                             hop_length=HOP_LENGTH, n_fft=N_FFT)

            # Reshape to (num_frames, num_mels, 1)
            X.append(mel_spectrogram.T[:, :, None])

            # Get label from filename
            label = file.split('.')[0]
            y.append(label)

    X = np.array(X)
    print(f"Shape before reshape: {X.shape}")

    # Reshape to (samples, num_mels, time_steps, channels)
    X = X.reshape((X.shape[0], N_MELS, int(SAMPLE_RATE / HOP_LENGTH * DURATION), 1))

    print(f"Shape after reshape: {X.shape}")

    y = to_categorical(y)

    return X, y


def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(np.unique(y)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Подготовка данных
data_dir = '.\data'
X, y = prepare_data(data_dir)

# Создание и обучение модели
model = create_model((N_MELS, int(SAMPLE_RATE / HOP_LENGTH * DURATION), 1))
history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

# Сохранение модели
model.save('speech_recognition_model.h5')

# Визуализация результатов обучения
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

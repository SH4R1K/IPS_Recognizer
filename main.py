import os
import numpy as np
import librosa
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Отключаем использование GPU, если необходимо
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Класс для загрузки данных
class AudioDataset(Dataset):
    def __init__(self, audio_data, audio_labels):
        self.audio_data = audio_data
        self.audio_labels = audio_labels

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        return self.audio_data[idx], self.audio_labels[idx]

def load_json_data(json_dir):
    data = []
    labels = []

    for label_file in os.listdir(json_dir):
        if label_file.endswith('.json'):
            label_path = os.path.join(json_dir, label_file)
            with open(label_path, 'r', encoding='utf-8') as f:
                data.append(json.load(f))
                labels.append(label_file.split('.')[0])  # Извлечение имени файла без расширения для меток

    return data, labels  # Возвращаем данные и метки

def load_data(data_dir, labels):
    audio_data = []
    audio_labels = []

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith('.wav'):
                    file_path = os.path.join(label_dir, file)
                    # Загрузка аудиофайла
                    audio, sr = librosa.load(file_path, sr=None)
                    audio_data.append(audio)
                    audio_labels.append(labels.index(label))  # Добавляем индекс метки

    return audio_data, audio_labels  # Возвращаем аудиоданные и метки

def preprocess_data(audio_data, sr=22050, max_length=16000):  # Установите max_length на 16000
    processed_data = []

    for audio in audio_data:
        # Обрезка или дополнение до max_length
        if len(audio) > max_length:
            audio = audio[:max_length]
        else:
            audio = np.pad(audio, (0, max_length - len(audio)), 'constant')

        # Преобразование в спектрограмму
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        # Преобразование в логарифмическую шкалу (опционально)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        processed_data.append(log_spectrogram)

    return np.array(processed_data)

# Пример загрузки данных
json_dir = 'D:\\pythonProject\\ESC_DATASET_v1.2\\annotation'  # Укажите путь к вашему JSON-файлу
data, labels = load_json_data(json_dir)  # Изменено на возврат меток
# Загрузка данных
data_dir = 'D:\\pythonProject\\ESC_DATASET_v1.2'  # Укажите путь к вашим данным
audio_data, audio_labels = load_data(data_dir, labels)  # Передаем метки в функцию

# Предобработка данных
X = preprocess_data(audio_data)

# Преобразуем данные в тензоры
X = torch.tensor(X, dtype=torch.float32)
audio_labels = torch.tensor(audio_labels, dtype=torch.long)

# Создание датасета и загрузчика данных
dataset = AudioDataset(X, audio_labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Определение модели
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)

        # Обновите входной размер на 16384
        self.fc1 = nn.Linear(64 * 32 * 8, 64)  # 64 * 32 * 8 = 16384
        self.fc2 = nn.Linear(64, len(set(audio_labels)))  # Убедитесь, что len(set(audio_labels)) соответствует количеству классов

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.unsqueeze(1))))  # Добавляем размерность для канала
        print(f'After conv1: {x.shape}')  # Отладочная строка
        x = self.pool(F.relu(self.conv2(x)))
        print(f'After conv2: {x.shape}')  # Отладочная строка
        x = torch.flatten(x, start_dim=1)  # Уплощаем тензор
        print(f'After flattening: {x.shape}')  # Отладочная строка
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Создание экземпляра модели
model = AudioClassifier()

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Устанавливаем модель в режим обучения
    running_loss = 0.0

    for inputs, labels in dataloader:
        optimizer.zero_grad()  # Обнуляем градиенты
        outputs = model(inputs)  # Прямой проход
        loss = criterion(outputs, labels)  # Вычисляем потерю
        loss.backward()  # Обратный проход
        optimizer.step()  # Обновляем параметры

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')

# Сохранение модели
torch.save(model.state_dict(), 'audio_classifier.pth')
print("Модель сохранена как 'audio_classifier.pth'")

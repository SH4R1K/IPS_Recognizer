from __future__ import annotations

import os
import argparse
import json
import torch
import torchaudio
from main import AudioClassifier, preprocess_data  # Импортируйте вашу модель


class Predictor:
    """Class for your model's predictions."""

    def __init__(self):
        self.model = AudioClassifier()  # Создайте экземпляр вашей модели
        self.model.load_state_dict(torch.load("D:\\pythonProject\\audio_classifier.pth"))  # Замените на путь к вашей модели
        self.model.eval()  # Установите модель в режим оценки

    def __call__(self, audio_path: str):
        # Загрузка аудиофайла
        waveform, sample_rate = torchaudio.load(audio_path)

        # Преобразование в тензор и добавление размерности для батча
        input_tensor = waveform.unsqueeze(0)  # Добавляем размерность для батча

        # Прогоните аудиофайл через модель
        with torch.no_grad():  # Отключаем градиенты для оценки
            output = self.model(input_tensor)  # Используем тензор PyTorch

        # Получение предсказаний
        predicted_label = torch.argmax(output, dim=1).item()  # Получаем индекс класса с максимальным значением
        predicted_text = "Your predicted text"  # Замените на вашу логику для получения текста, если необходимо
        predicted_attribute = -1  # Замените на вашу логику для получения атрибута, если необходимо

        result = {
            "audio": os.path.basename(audio_path),  # Audio file base name
            "text": predicted_text,  # Predicted text
            "label": predicted_label,  # Text class
            "attribute": predicted_attribute,  # Predicted attribute (if any, or -1)
        }
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get submission.")
    parser.add_argument(
        "--src",
        type=str,
        help="Path to the source audio files.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        help="Path to the output submission.",
    )
    args = parser.parse_args()
    predictor = Predictor()

    results = []
    for audio_file in os.listdir(args.src):
        audio_path = os.path.join(args.src, audio_file)
        if audio_file.endswith('.wav'):  # Убедитесь, что это аудиофайл
            result = predictor(audio_path)
            results.append(result)

    with open(
            os.path.join(args.dst, "submission.json"), "w", encoding="utf-8"
    ) as outfile:
        json.dump(results, outfile)
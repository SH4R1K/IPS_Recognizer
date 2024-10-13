from __future__ import annotations  # Для использования аннотаций типов (поддержка более новых возможностей типов)

import os  # Для работы с файловой системой
import argparse  # Для парсинга аргументов командной строки
import json  # Для работы с JSON-файлами

from src.ml_model import MLModel  # Импортируем модель из другого модуля


class Predictor:
    """
    Класс Predictor выполняет предсказание по аудиофайлам.
    """

    def __init__(self):
        # Инициализируем модель с помощью заранее обученного файла модели
        self.model = MLModel("my_module_backup.h5")

    def __call__(self, audio_path: str):
        # Метод __call__ позволяет вызывать объект как функцию
        prediction = self.model.predict(audio_path)
        result = {
            "audio": os.path.basename(audio_path),  # Извлекаем имя файла из полного пути
            "text": prediction["predicted_text"],  # Предсказанный текст
            "label": prediction["predicted_label"],  # Предсказанная метка
            "attribute": -1  # По умолчанию устанавливаем атрибут в -1 (может быть изменен в будущем)
        }
        return result

    # Пути к файлам с исходными данными (JSON и аудио)
    ground_truth_data_path_json = "./src/data/hr_bot_merged.json"
    ground_truth_data_path_audio = "./src/data/hr_bot_merged.json"


if __name__ == "__main__":
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(description="Get submission.")

    parser.add_argument(
        "--src",
        type=str,
        help="Path to the source audio files.",  # Путь к папке с аудиофайлами
    )
    parser.add_argument(
        "--dst",
        type=str,
        help="Path to the output submission.",  # Путь для сохранения результатов
    )
    parser.add_argument(
        "--metrics",
        type=bool,
        help="True if you need to get metrics.",  # Если True, то будут рассчитаны метрики модели
    )
    args = parser.parse_args()  # Получаем аргументы

    # Создаем экземпляр класса Predictor
    predictor = Predictor()

    results = []  # Список для хранения результатов предсказаний
    # Проходим по всем аудиофайлам в указанной папке
    for audio_path in os.listdir(args.src):
        result = predictor(os.path.join(args.src, audio_path))  # Получаем предсказание
        results.append(result)  # Добавляем результат в список

    # Читаем файл с исходными данными (ground truth)
    with open(predictor.ground_truth_data_path_json, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)  # Загружаем JSON с метками

    # Записываем предсказания в файл submission.json
    with open(
            os.path.join(args.dst, "submission.json"), "w", encoding="utf-8"
    ) as outfile:
        json.dump(results, outfile)  # Сохраняем результаты в JSON

    # Если указан флаг для расчета метрик
    if args.metrics:
        # Сохраняем метрики модели в файл submission_metrics.json
        with open(
                os.path.join(args.dst, "submission_metrics.json"), "w", encoding="utf-8"
        ) as outfile_metrics:
            json.dump(predictor.model.evaluate_model(ground_truth_data),
                      outfile_metrics)  # Рассчитываем и сохраняем метрики

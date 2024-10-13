# import os
# from pydub import AudioSegment
# from vosk import Model, KaldiRecognizer
# import wave
#
# # Установим путь к ffmpeg и ffprobe
# os.environ["PATH"] += os.pathsep + r"C:\Users\Meresk\Desktop\haha1\IPS_Recognizer\ffmpeg"
#
# def convert_audio_format(input_audio_path, output_audio_path):
#     # Загружаем аудиофайл
#     audio = AudioSegment.from_file(input_audio_path)
#
#     # Приведение к моно
#     audio = audio.set_channels(1)  # Устанавливаем 1 канал (моно)
#
#     # Установка частоты дискретизации
#     audio = audio.set_frame_rate(16000)  # Устанавливаем 16000 Гц
#
#     # Установка битности (16 бит = 2 байта)
#     audio = audio.set_sample_width(2)  # 2 байта = 16 бит
#
#     # Экспортируем файл с новыми параметрами
#     audio.export(output_audio_path, format="wav")
#
# def recognize_speech(file_path):
#     model = Model("vosk-model-small-ru-0.22")  # Путь к модели
#     recognizer = KaldiRecognizer(model, 16000)
#
#     wf = wave.open(file_path, "rb")
#     if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
#         print("Файл не соответствует требованиям: моно, 16 бит, 16000 Гц.")
#         return
#
#     text = ""
#     while True:
#         data = wf.readframes(4000)
#         if not data:
#             break
#         if recognizer.AcceptWaveform(data):
#             result = recognizer.Result()
#             if isinstance(result, dict) and 'text' in result:
#                 text += result['text'] + " "
#
#     final_result = recognizer.FinalResult()
#     if isinstance(final_result, str) and final_result.strip():
#         text += " " + final_result.strip()
#
#     return text.strip()
#
# # Пример использования
# input_file_path = '2.mp3'  # Путь к входному MP3 файлу
# converted_file_path = 'converted_audio.wav'
#
# # Конвертируем аудиофайл
# convert_audio_format(input_file_path, converted_file_path)
#
# # Распознаем речь
# recognized_text = recognize_speech(converted_file_path)
# print("Распознанный текст:", recognized_text)
#
# # Удаление временного файла
# os.remove(converted_file_path)

import os
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
import wave
import re


# Установим путь к ffmpeg и ffprobe
os.environ["PATH"] += os.pathsep + r"C:\Users\Meresk\Desktop\haha1\IPS_Recognizer\ffmpeg"

def convert_audio_format(input_audio_path, output_audio_path):
    # Загружаем аудиофайл
    audio = AudioSegment.from_file(input_audio_path)

    # Приведение к моно
    audio = audio.set_channels(1)  # Устанавливаем 1 канал (моно)

    # Установка частоты дискретизации
    audio = audio.set_frame_rate(16000)  # Устанавливаем 16000 Гц

    # Установка битности (16 бит = 2 байта)
    audio = audio.set_sample_width(2)  # 2 байта = 16 бит

    # Экспортируем файл с новыми параметрами
    audio.export(output_audio_path, format="wav")

def recognize_speech(file_path):
    model = Model("vosk-model-small-ru-0.22")  # Путь к модели
    recognizer = KaldiRecognizer(model, 16000)

    wf = wave.open(file_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
        print("Файл не соответствует требованиям: моно, 16 бит, 16000 Гц.")
        return ""

    text = ""
    while True:
        data = wf.readframes(4000)
        if not data:
            break
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            if isinstance(result, dict) and 'text' in result:
                text += result['text'] + " "

    final_result = recognizer.FinalResult()
    if isinstance(final_result, str) and final_result.strip():
        # Если `final_result` — это JSON, разбираем его.
        try:
            import json
            final_data = json.loads(final_result)
            if 'text' in final_data:
                text += " " + final_data['text'].strip()
        except json.JSONDecodeError:
            text += " " + final_result.strip()

    return text.strip()


def find_word_numbers(text):
    # Словарь для перевода чисел из слов к цифрам
    word_to_number = {
        'ноль': 0, 'один': 1, 'два': 2, 'три': 3, 'четыре': 4,
        'пять': 5, 'шесть': 6, 'семь': 7, 'восемь': 8, 'девять': 9,
        'десять': 10, 'одиннадцать': 11, 'двенадцать': 12,
        'тринадцать': 13, 'четырнадцать': 14, 'пятнадцать': 15,
        'шестнадцать': 16, 'семнадцать': 17, 'восемнадцать': 18,
        'девятнадцать': 19, 'двадцать': 20,
    }

    # Регулярное выражение для поиска чисел-слов
    pattern = r'\b(' + '|'.join(word_to_number.keys()) + r')\b'

    # Поиск всех совпадений
    match = re.search(pattern, text.lower())

    # Если найдено число, возвращаем его значение
    if match:
        return word_to_number[match.group()]

    # Если не найдено, возвращаем None
    return None


def process_audio(input_file_path):
    converted_file_path = 'converted_audio.wav'

    # Конвертируем аудиофайл
    convert_audio_format(input_file_path, converted_file_path)

    # Распознаем речь
    recognized_text = recognize_speech(converted_file_path)
    recognized_attribute = find_word_numbers(recognized_text)

    # Удаление временного файла
    os.remove(converted_file_path)

    return recognized_text, recognized_attribute

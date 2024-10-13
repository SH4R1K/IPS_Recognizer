
# **IPSRecognizer**

> **Проект создан в рамках окружного хакатона СЗФО**  
> Система управления локомотивом с использованием голосового управления.

---

## **Описание проекта**
**IPSRecognizer** — это инновационная система, предназначенная для управления локомотивом с помощью голосовых команд, подаваемых через индивидуальный пульт составителя. Основная цель проекта — улучшение безопасности и эффективности управления железнодорожными составами за счет снижения необходимости ручного управления и упрощения взаимодействия с оборудованием.

---

## **Основные функции**
- **Голосовое управление локомотивом**: использование предопределенных голосовых команд для выполнения ключевых операций.
- **Реализация с использованием машинного обучения**: система использует нейронные сети для распознавания команд и их сопоставления с действиями.

---

## **Стек технологий**

![Python](https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)  
![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)  
![Keras](https://img.shields.io/badge/-Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)  
![Librosa](https://img.shields.io/badge/-Librosa-FFBB00?style=for-the-badge&logo=librosa&logoColor=white)  
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)  
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

---

## **Архитектура проекта**

1. **Сбор данных**: для обучения модели используется заранее размеченный набор голосовых команд.
2. **Предобработка данных**: с помощью библиотеки Librosa выполняется извлечение признаков и нормализация аудиоданных.
3. **Модель машинного обучения**: на основе TensorFlow и Keras обучается нейронная сеть для классификации голосовых команд.
4. **Интеграция с пультом управления**: реализован интерфейс, который позволяет передавать распознанные команды на пульт управления локомотивом.
5. **Оценка и оптимизация**: проводится постоянная проверка качества модели и её дообучение на новых данных для повышения точности.

---

## **Установка и запуск проекта**

1. Клонируйте репозиторий:
    ```bash
    git clone https://github.com/SH4R1K/IPS_Recognizer.git
    ```
2. Установите зависимости:
    ```bash
    pip install -r requirements.txt
    ```
3. Запустите обучение модели:
    ```bash
    python src/model_teach.py
   ```
4. Запустите приложение:
    ```bash
    python get_submissions.py --src .\src\data\audio\test_files\ --dst . --metrics true
    ```
## Дополнительная информация
`get_submissions.py` поддерживает следующие параметры:
- --src определяет источник входных данных
- --dst определяет папку для вывода
- --metrics true/false определяет необходимо ли замерять метрики
---

## **Планы на будущее**

- Расширение набора поддерживаемых голосовых команд.
- Улучшение точности модели распознавания.
- Интеграция с другими системами управления.
- Поддержка многоязычности.


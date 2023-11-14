import pytesseract
import easyocr
from PIL import Image

class ImageRecognizer:
    def __init__(self):
        self.tesseract_config = r'--oem 3 --psm 6'  # конфигурация для TesseractOCR
        self.reader = easyocr.Reader(['en','ru'])  # EasyOCR с поддержкой английского и русского

    # метод распознавания - Tesseract OCR
    def annotate_images(self, image_paths, annotation_file):
        # создание пустого словаря для результатов
        annotations = {}
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        # проход по списку путей к изображениям
        for image_path in image_paths:
            # открытие каждого изображения
            img = Image.open(image_path)
            # распознавание текста
            text = pytesseract.image_to_string(img, config=self.tesseract_config)
            # сохранение результата в словарь
            annotations[image_path] = text.strip()

        # запись в результирующий файл
        with open(annotation_file, 'w') as file:
            for image_path, annotation in annotations.items():
                file.write(f"{image_path}: {annotation}\n")


    # вычисление точности
    def evaluate_accuracy(self, ground_truth, predictions):
        # счётчик корректных элементов
        correct = 0
        # общее количество распознанных элементов
        total = len(ground_truth)

        for image_path, true_text in ground_truth.items():
            predicted_text = predictions.get(image_path, '')
            # проверка, совпадает ли распознанный текст с ожидаемым
            if predicted_text == true_text:
                correct += 1

        # вычисление точности
        accuracy = correct / total
        return accuracy

    # метод распознавания - Tesseract
    def straight_recognition(self, image_paths):
        # создание пустого словаря predictions - это словарь, содержащий пути к изображениям в качестве ключей и текст, распознанный на каждом изображении, в качестве значений
        predictions = {}
        for image_path in image_paths:
            # открытие изображения
            img = Image.open(image_path)
            # распознавание текста
            text = pytesseract.image_to_string(img, config=self.tesseract_config)
            # сохранение результата в словарь
            predictions[image_path] = text.strip()
        return predictions

    # метод распознавания - EasyOCR
    def easyocr_recognition(self, image_paths):
        # создание пустого словаря predictions - это словарь, содержащий пути к изображениям в качестве ключей и текст, распознанный на каждом изображении, в качестве значений
        predictions = {}
        for image_path in image_paths:
            # открытие изображения
            img = Image.open(image_path)
            # распознавание текста
            result = self.reader.readtext(image_path)
            text = ' '.join([item[1] for item in result])
            # сохранение результата в словарь
            predictions[image_path] = text.strip()
        return predictions


    # Тестирование распознавания для указанного типа, оценивает точность и сохраняем предсказания в файл.
    def test_recognition(self, rec_type, val_type, image_paths, ground_truth_file):
        # выбор метода распознавания
        if rec_type == 'straight':
            predictions = self.straight_recognition(image_paths)
        elif rec_type == 'easyocr':
            predictions = self.easyocr_recognition(image_paths)
        else:
            raise ValueError(f"Неподдерживаемый метод распознавания: {rec_type}")

        # загрузка файла с правильными ответами для каждого изображения
        # создание пустого словаря ground_truth - это словарь, содержащий пути к изображениям в качестве ключей и соответствующий текст, который должен быть распознан на каждом изображении, в качестве значений
        ground_truth = {}
        with open(ground_truth_file, 'r') as file:
            for line in file:
                parts = line.split(':')
                image_path = parts[0].strip()
                # сохранение в словарь правильных ответов
                true_text = parts[1].strip()
                ground_truth[image_path] = true_text

        # оценка качества точности - полное совпадение
        if val_type == 'full_match':
            accuracy = self.evaluate_accuracy(ground_truth, predictions)

        # запись распознанных слов в файл
        predictions_file = f'{rec_type}_predictions.txt'
        with open(predictions_file, 'w') as file:
            for image_path, prediction in predictions.items():
                file.write(f"{image_path}: {prediction}\n")

        return accuracy

    # оценка качества распознавания - попарное сравнение
    def compare_predictions(self, ground_truth_file, straight_predictions_file, easyocr_predictions_file):
        # загрузка файла с правильными ответами для каждого изображения
        ground_truth = {}
        with open(ground_truth_file, 'r') as file:
            for line in file:
                parts = line.split(':')
                image_path = parts[0].strip()
                # сохранение в словарь правильных ответов
                true_text = parts[1].strip()
                ground_truth[image_path] = true_text

        # загрузка файла с распознанными словами от метода straight_recognition
        straight_predictions = {}
        with open(straight_predictions_file, 'r') as file:
            for line in file:
                parts = line.split(':')
                image_path = parts[0].strip()
                prediction_text = parts[1].strip()
                straight_predictions[image_path] = prediction_text

        # загрузка файла с распознанными словами от метода easyocr_recognition
        easyocr_predictions = {}
        with open(easyocr_predictions_file, 'r') as file:
            for line in file:
                parts = line.split(':')
                image_path = parts[0].strip()
                prediction_text = parts[1].strip()
                easyocr_predictions[image_path] = prediction_text

        # оценка качества распознавание - сравнение по словам
        straight_accuracy = self.evaluate_accuracy(ground_truth, straight_predictions)
        easyocr_accuracy = self.evaluate_accuracy(ground_truth, easyocr_predictions)

        return straight_accuracy, easyocr_accuracy

recognizer = ImageRecognizer()
# пути к изображениям
# image_paths = ['dataset/1.jpg', 'dataset/2.jpg', 'dataset/3.jpg', 'dataset/4.jpg', 'dataset/5.jpg']
image_paths = ['yandex-capchi/1.jpg', 'yandex-capchi/2.jpg','yandex-capchi/4.jpg', 'yandex-capchi/5.jpg','yandex-capchi/6.jpg', 'yandex-capchi/7.jpg','yandex-capchi/9.jpg', 'yandex-capchi/10.jpg','yandex-capchi/11.jpg']

# путь к файлу с правильными ответами для каждого изображения
# ground_truth_file = 'ground_truth.txt'
ground_truth_file = 'true_words.txt'

recognizer.annotate_images(image_paths, ground_truth_file)

recognition_type = 'straight'
validation_type = 'full_match'

accuracy = recognizer.test_recognition(recognition_type, validation_type, image_paths, ground_truth_file)
print('Оценка точности методом полного совпадения:')
print(f"Accuracy for {recognition_type} recognition: {accuracy * 100:.2f}%")

# пути с файлами с распознаными словами
straight_predictions_file = 'straight_predictions.txt'
easyocr_predictions_file = 'easyocr_predictions.txt'

# оценка точности методом попарных сравнений
straight_accuracy, easyocr_accuracy = recognizer.compare_predictions(ground_truth_file, straight_predictions_file, easyocr_predictions_file)
print("Оценка точности методом попарного сравнения слов:")
print(f"Straight Recognition Accuracy: {straight_accuracy * 100:.2f}%")
print(f"EasyOCR Recognition Accuracy: {easyocr_accuracy * 100:.2f}%")
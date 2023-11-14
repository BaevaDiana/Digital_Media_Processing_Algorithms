import pytesseract
import easyocr
from PIL import Image
import os

class ImageRecognizer:
    def __init__(self):
        self.tesseract_config = r'--oem 3 --psm 6'  # конфигурация для TesseractOCR
        self.reader = easyocr.Reader(['en', 'ru'])  # конфигурация для EasyOCR с поддержкой английского и русского

    #Tesseract, запись результатов в файл аннотаций.
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
        with open(annotation_file, 'w', encoding='utf-8',errors='replace') as file:
            for image_path, annotation in annotations.items():
                file.write(f"{image_path}: {annotation}\n")

    # вычисление точности распознавания
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
        # проход по списку путей к изображениям
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
        # проход по списку путей к изображениям
        for image_path in image_paths:
            # открытие изображения
            img = Image.open(image_path)
            # распознавание текста
            result = self.reader.readtext(image_path)
            text = ' '.join([item[1] for item in result])
            # распознавание текста
            predictions[image_path] = text.strip()

        return predictions

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
        # Load ground truth from the annotation file
        ground_truth = {}
        with open(ground_truth_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.split(':')
                if len(parts) >= 2:
                    # сохранение в словарь правильных ответов
                    image_path = parts[0].strip()
                    true_text = parts[1].strip()
                    ground_truth[image_path] = true_text
                else:
                    # обработка случая, когда в строке нет символа ':' или после ':' нет текста
                    print(f"Неправильный формат строки: {line}")

        # оценка качества точности - полное совпадение
        if val_type == 'full_match':
            accuracy = self.evaluate_accuracy(ground_truth, predictions)

        # запись распознанных слов в файл в кодировке UTF-8
        predictions_file = f'{rec_type}_predictions_2.txt'
        with open(predictions_file, 'w', encoding='utf-8') as file:
            for image_path, prediction in predictions.items():
                file.write(f"{image_path}: {prediction}\n")

        return accuracy

    # аугментация датасета
    def augment_dataset(self, original_path, augmented_path):
        # создание новой папки для нового датасета
        if not os.path.exists(augmented_path):
            os.makedirs(augmented_path)

        for image_path in os.listdir(original_path):
            if image_path.endswith(('.jpg', '.jpeg', '.png')):
                original_image = Image.open(os.path.join(original_path, image_path))

                original_image = original_image.convert('RGB')

                for angle in range(-20, 21):
                    rotated_image = original_image.rotate(angle)
                    rotated_image_path = os.path.join(augmented_path, f"{os.path.splitext(image_path)[0]}_{angle}.jpg")
                    rotated_image.save(rotated_image_path)

    def test_augmented_dataset(self, rec_type, val_type, augmented_path, ground_truth_file):

            # список всех изображений в аугментированном датасете
            augmented_images = [os.path.join(augmented_path, image) for image in os.listdir(augmented_path)]

            # распознавание на аугментированном датасете
            accuracy = self.test_recognition(rec_type, val_type, augmented_images, ground_truth_file)

            return accuracy

    # оценка качества распознавания - попарное сравнениеО
    def compare_predictions_wordwise(self, ground_truth_file, straight_predictions_file, easyocr_predictions_file):
        # загрузка файла с правильными ответами для каждого изображения
        ground_truth = {}
        with open(ground_truth_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.split(':')
                if len(parts) >= 2:
                    image_path = parts[0].strip()
                    # сохранение в словарь правильных ответов
                    true_text = parts[1].strip()
                    ground_truth[image_path] = true_text

        # загрузка файла с распознанными словами от метода straight_recognition
        straight_predictions = {}
        with open(straight_predictions_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.split(':')
                if len(parts) >= 2:
                    image_path = parts[0].strip()
                    prediction_text = parts[1].strip()
                    straight_predictions[image_path] = prediction_text
                #else:
                    #print(f"Invalid line format in {straight_predictions_file}: {line}")

        # загрузка файла с распознанными словами от метода easyocr_recognition
        easyocr_predictions = {}
        with open(easyocr_predictions_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.split(':')
                if len(parts) >= 2:
                    image_path = parts[0].strip()
                    prediction_text = parts[1].strip()
                    easyocr_predictions[image_path] = prediction_text
                #else:
                    #print(f"Invalid line format in {easyocr_predictions_file}: {line}")

        # оценка качества распознавание - сравнение по словам
        straight_accuracy = self.evaluate_accuracy_wordwise(ground_truth, straight_predictions)
        easyocr_accuracy = self.evaluate_accuracy_wordwise(ground_truth, easyocr_predictions)

        return straight_accuracy, easyocr_accuracy

    def evaluate_accuracy_wordwise(self, ground_truth, predictions):
        correct = 0
        total = len(ground_truth)

        for image_path, true_text in ground_truth.items():
            predicted_text = predictions.get(image_path, '')
            true_words = set(true_text.split())
            predicted_words = set(predicted_text.split())
            if true_words == predicted_words:
                correct += 1

        accuracy = correct / total
        return accuracy


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
recognizer = ImageRecognizer()

# пути к датасетам
original_dataset_path = 'dataset_1'
augmented_dataset_path = 'dataset2'
#  путь к файлу с правильными ответами для каждого изображения
ground_truth_file = 'true_words_2.txt'

recognition_type = 'straight'  #straight   easyocr
validation_type = 'full_match'

# аугментация датасета
recognizer.augment_dataset(original_dataset_path, augmented_dataset_path)

# распознавание на аугментированном датасете
accuracy_augmented = recognizer.test_augmented_dataset(recognition_type, validation_type, augmented_dataset_path, ground_truth_file)
print('Оценка точности методом полного совпадения:')
print(f"Accuracy for {recognition_type} recognition on augmented dataset: {accuracy_augmented * 100:.2f}%")

# пути с файлами с распознаными словами
straight_predictions_file = 'straight_predictions_2.txt'
easyocr_predictions_file = 'easyocr_predictions_2.txt'

# сравнение по словам
straight_accuracy_wordwise, easyocr_accuracy_wordwise = recognizer.compare_predictions_wordwise(
    ground_truth_file, straight_predictions_file, easyocr_predictions_file
)
print("Оценка точности методом попарного сравнения слов:")
print(f"Straight Recognition Wordwise Accuracy: {straight_accuracy_wordwise * 100:.2f}%")
print(f"EasyOCR Recognition Wordwise Accuracy: {easyocr_accuracy_wordwise * 100:.2f}%")
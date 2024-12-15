import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance

# Пути к шрифтам
FONT_PATH = [
    "Fonts/calibri.ttf",
    "Fonts/arial.ttf",
    "Fonts/MathCadUniMath.otf",
    "Fonts/times.ttf",
    "Fonts/Courgette-Regular.ttf",
    "Fonts/cambriai.ttf"
]

# Папка для датасета
DATASET_FOLDER = "dataset"
os.makedirs(DATASET_FOLDER, exist_ok=True)
# Функция для добавления фонового шума
def add_background_noise(img):
    noise = Image.effect_noise(img.size, random.randint(5, 15))  # Интенсивность шума
    noise = ImageOps.autocontrast(noise)
    img = Image.composite(img, noise, img)
    return img

# Генерация датасета
def generate_dataset():
    for digit in range(0, 10):  # Числа от 0 до 9
        digit_folder = os.path.join(DATASET_FOLDER, str(digit))
        os.makedirs(digit_folder, exist_ok=True)
        if digit not in [8,9]:
            n = 100
        else:
            n = 200

        for schrift in range(len(FONT_PATH)):  # По всем шрифтам
            for i in range(n):  # Генерируем 100 изображений на каждую цифру
                # Создаём белое изображение размером 32x32
                img = Image.new("L", (32, 32), color=255)  # "L" означает градации серого
                draw = ImageDraw.Draw(img)

                # Загружаем шрифт
                font_size = random.randint(35, 40)  # Случайный размер шрифта
                font = ImageFont.truetype(FONT_PATH[schrift], font_size)

                # Вычисляем размеры текста
                bbox = draw.textbbox((0, 0), str(digit), font=font)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

                # Случайное смещение текста
                offset_x = random.randrange(-3, 3)
                offset_y = random.randrange(-2, 2)
                position = ((32 - w) / 2 + offset_x, ((32 - h)-10) / 2 + offset_y)

                # Рисуем текст
                draw.text(position, str(digit), fill=0, font=font)
                # Случайные аффинные преобразования

                # Случайное вращение
                angle = random.randint(-15, 15)  # Угол поворота
                img = img.rotate(angle, fillcolor=255)

                # Добавление фонового шума
                if random.random() > 0.5:  # 50% вероятности
                    img = add_background_noise(img)

                # Сохранение изображения
                img_path = os.path.join(digit_folder, f"{schrift}_{digit}_{i}.png")
                img.save(img_path)
    print("Датасет сгенерирован.")
# Генерация датасета
generate_dataset()

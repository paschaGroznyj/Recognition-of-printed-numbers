import tensorflow as tf
import cv2
import numpy as np

# Загрузка модели
model = tf.keras.models.load_model("my_model_numbers.keras")


def letters_extract(image_file: str, out_size=28):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
    cv2.imshow('1', img_erode)
    cv2.waitKey(1000)
    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
        # hierarchy[i][0]: the index of the next contour of the same level
        # hierarchy[i][1]: the index of the previous contour of the same level
        # hierarchy[i][2]: the index of the first child
        # hierarchy[i][3]: the index of the parent
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            # print(letter_crop.shape)

            # Resize letter canvas to square
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h: #Если ширина больше высоты
                # Enlarge image top-bottom
                # ------
                # ======
                # ------
                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Enlarge image left-right
                # --||--
                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop
            # cv2.imshow('1', letter_square)
            # cv2.waitKey(1000)
            # Resize letter to 28x28 and add letter and its X-coordinate
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))
    # cv2.imshow("1", letters[1][2])
    # cv2.imshow("2", letters[2][2])
    # cv2.imshow("3", letters[3][2])
    # cv2.imshow("4", letters[4][2])

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)
    return letters


def recognize_digits(img):
    """
    Распознает числа на изображении.
    :param img: входное изображение (OpenCV, cv2)
    :return: строка с числами
    """

    # Преобразование изображения в градации серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Бинаризация изображения
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Находим контуры чисел
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digits = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        digit_img = binary[y:y + h, x:x + w]

        # Изменяем размер изображения под модель
        digit_img = cv2.resize(digit_img, (28, 28))
        digit_img = digit_img / 255.0
        digit_img = digit_img.reshape(1, 28, 28, 1)



        # Прогноз
        prediction = np.argmax(model.predict(digit_img), axis=-1)
        digits.append((x, prediction[0]))

    # Сортируем числа по их позиции
    digits.sort(key=lambda x: x[0])
    result = "".join(str(d[1]) for d in digits)
    return result

letters = letters_extract('img_11.png')
for i in range(len(letters)):
    cv2.imshow('1', letters[i][2])
    cv2.waitKey(1000)
    print(recognize_digits(letters[i][2]))

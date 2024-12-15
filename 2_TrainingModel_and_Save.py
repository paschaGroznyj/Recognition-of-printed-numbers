import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

x_train = []
y_train = []

for i in range(0, 10):  # По всем папкам 0-9
    if i not in [8, 9]:
        n = 100
    else:
        n = 200
    for font in range(6):
        for j in range(n):  # По всем изображениям 0-49
        # Загружаем изображение в градациях серого
            img = cv2.imread(f"dataset/{i}/{font}_{i}_{j}.png", 0)
            # Преобразуем изображение в одну строку (flatten)
            img_flattened = img.flatten()  # Это эквивалентно reshape(-1)

            # Добавляем в список x_train
            x_train.append(img_flattened)

            # Добавляем метку класса в y_train
            y_train.append(i)

# Преобразуем списки в массивы NumPy
x_train = np.array(x_train, dtype=np.uint8)  # Преобразуем в uint8 для экономии памяти
y_train = np.array(y_train, dtype=np.uint8)
print(y_train)
x_train = x_train.reshape(-1, 32, 32, 1) #Создаем трехмерный массив с глубиной
#(количество примеров, 28, 28, 1)
x_train = x_train / 255.0

y_train_cat = keras.utils.to_categorical(y_train, 10)

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),  # Добавьте Dropout для уменьшения переобучения
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
# opt = keras.optimizers.Adam(learning_rate=0.01)
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# model = keras.models.Sequential([
#     keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(10, activation='softmax')
# ])
# model = keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=(32, 32, 1)))
# model.add(keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_cat, batch_size=40, epochs=6, validation_split=0.1)

model.save('print_letters.keras')




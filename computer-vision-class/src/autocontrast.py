import os.path
from sys import argv

import cv2
import numpy as np


def autocontrast(src_path, dst_path, white_perc, black_perc):
    # Считываем изображение и проверяем, что оно считалось
    img = cv2.imread(src_path)
    assert img is not None
    # Создаем гистограмму изображения, она понадобится для отрезания долей темных и светлых пикселей
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    table = np.array([j for j in range(256)]).astype("uint8")
    # Подсчитываем кол-во темных и кол-во светлых пикселей в зависимости от долей
    a = img.shape[0] * img.shape[1] * float(white_perc)
    b = img.shape[0] * img.shape[1] * float(black_perc)
    k = 0
    l = 0
    max_val = 0
    min_val = 0
    # Преобразуем темные пиксели, а также находим минимум для линейной коррекции
    for i in range(0, 256):
        if (k < b):
            k = k + int(hist[i])
            table[i] = 0
            min_val = i + 1
    # Преобразуем светлые пиксели, а также находим максимум для линейной коррекции
    for i in range(255, 0, -1):
        if (l < a):
            l = l + int(hist[i])
            table[i] = 255
            max_val = i - 1
    # Линейная коррекция оставшихся пикселей
    for i in range(0, 256):
        if (table[i] != 0 and table[i] != 255):
            table[i] = int((table[i] - min_val) * (255.0 - 0.0) / (max_val - min_val))
    # Делаем Look Up Table и сохраняем изменненное изображение
    img1 = cv2.LUT(img, table)
    cv2.imwrite(dst_path, img1)


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = float(argv[3])
    argv[4] = float(argv[4])

    assert 0 <= argv[3] < 1
    assert 0 <= argv[4] < 1

    autocontrast(*argv[1:])

import os.path
from sys import argv

import cv2
import numpy as np


def otsu(src_path, dst_path):
    img = cv2.imread(src_path, 0)  # Считываем одноканальное изображение
    assert img is not None

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # Строим гистограмму изображения
    size = img.size  # Находим кол-во пикселей в изображении для определения вероятностей появления тона
    hist_prob = hist.ravel() / size  # Строим распределение вероятностей
    hist_cumsum = hist_prob.cumsum()  # Находим кумулятивную сумму распр.вероятностей она понадобится для определения весов

    table = np.arange(256)
    a, b = img.shape
    max_sigmab = -1  # Переменная, которая будет хранить максимальную межклассовую дисперсию
    t = -1  # Порог

    hist_table = table * hist_prob  # Массив (i * p(i))для вычисления математического ожидания
    hist_table_cumsum = hist_table.cumsum()  # Кумулятивная сумма для вычисления математического ожидания

    for i in range(1, 256):
        weight0 = hist_cumsum[i - 1]  # Каждая ячейка в данном массиве равна сумме всех элементов слева
        weight1 = hist_cumsum[-1] - weight0
        mu0 = hist_table_cumsum[i - 1] / weight0 if weight0 > 0 else 0  # Находим мат.ожидания
        mu1 = (hist_table_cumsum[-1] - hist_table_cumsum[i - 1]) / weight1 if weight1 > 0 else 0
        sigmab = weight0 * weight1 * ((mu0 - mu1) ** 2)  # Находим межклассовую дисперсию
        # Находим максимальную межклассовую дисперсию, значение t будет искомым порогом
        if (sigmab > max_sigmab):
            max_sigmab = sigmab
            t = i
    # Бинаризуем изображение
    for i in range(0, a):
        for j in range(0, b):
            if (img[i, j] < t):
                img[i, j] = 0
            else:
                img[i, j] = 255
    # Записываем изображение
    cv2.imwrite(dst_path, img)


if __name__ == '__main__':
    assert len(argv) == 3
    assert os.path.exists(argv[1])
    otsu(*argv[1:])

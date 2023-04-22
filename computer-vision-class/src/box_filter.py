import os.path
from sys import argv

import cv2
import numpy as np


def box_flter(src_path, dst_path, w, h):
    # Считываем изображение в одноканальном режиме и проверяем, что оно считалось.
    img = cv2.imread(src_path, 0)
    assert img is not None
    # Определеяем размеры изображения.
    a, b = img.shape
    # Создаем кумулятивную матрицу, каждый элемент которой представляет сумму всех элементов левее и выше в матрице исходного изображения.
    # Сложность не более O(N).
    img_filt = np.cumsum(np.cumsum(img, axis=0), axis=1, dtype='int64')
    # Окно размера w x h имеет центр в нижней правой точке. Относительно этой точки будет применяться фильтр.
    # Происходит усреднение значения пикселя в окрестности окна, используя кумулятивную сумму, посчитанную ранее.
    # Общая сложность не более O(N), нет зависимости производительности алгоритма от w и h
    for i in range(int(h), a):
        for j in range(int(w), b):
            img[i, j] = (img_filt[i, j] - img_filt[i, j - int(w)] - img_filt[i - int(h), j] + img_filt[
                i - int(h), j - int(w)]) / (int(w) * int(h))
    # Происходит сохранения преобразованного изображения с обрезанием необработанных краев (срезы можно удалить)
    cv2.imwrite(dst_path, img[int(h): a, int(w): b])


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = int(argv[3])
    argv[4] = int(argv[4])
    assert argv[3] > 0
    assert argv[4] > 0

    box_flter(*argv[1:])

# coding=utf-8
from __future__ import print_function
from sys import argv
import os.path

import cv2
import numpy as np


def gamma_correction(src_path, dst_path, a, b):
    # Считываем изображение и проверяем, что оно считалось
    img = cv2.imread(src_path)
    assert img is not None
    # Создаем массив для преобразования каждого пикселя у изображения
    table = np.array([(max(0.0, min(1.0, float(a) * ((i / 255.0) ** float(b))))) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # Делаем Look Up Table и сохраняем изменненное изображение
    img1 = cv2.LUT(img, table)
    cv2.imwrite(dst_path, img1)


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = float(argv[3])
    argv[4] = float(argv[4])

    gamma_correction(*argv[1:])

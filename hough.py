from __future__ import print_function
from sys import argv
import cv2
import numpy as np


def gradient_img(img):
    hor_grad = (img[1:, :] - img[:-1, :])[:, :-1]
    ver_grad = (img[:, 1:] - img[:, :-1])[:-1:, :]
    magnitude = np.sqrt(hor_grad ** 2 + ver_grad ** 2)

    return magnitude


def hough_transform(img, theta, rho):
    # Считываем изображение в одноканальном режими, получаем высоту и ширину изображения
    cv2.imwrite("temp.png", img)
    img = cv2.imread("temp.png", 0)
    height, width = img.shape
    # Находим максимальную длину нормали от начала координат до точки
    Rr = np.sqrt(height ** 2 + width ** 2)
    R = int(round(Rr))
    # Заполняем ось углов и расстояний на основании входных аргументов
    rhos = np.arange(0, R + rho, rho)
    thetas = np.arange(-np.pi, np.pi / 2 + theta / 2, theta)
    # Создаем пространство Хафа
    ht_map = np.zeros((len(rhos), len(thetas)))
    # Для каждого пикселя находим r
    # Заполняем пространство Хафа с учетом того, что каждая точка изображения порождает кривую в пространстве Хафа
    for i in range(0, height):
        for j in range(0, width):
            for itheta in xrange(len(thetas)):
                r = j * np.cos(thetas[itheta]) + i * np.sin(thetas[itheta])
                if r >= 0:
                    ht_map[np.argmin(np.abs(rhos - r))][itheta] += img[i, j]
    # Нормализуем пространство для наглядности отображения кривых
    cv2.normalize(ht_map, ht_map, 0.0, 255.0, norm_type=cv2.NORM_MINMAX)
    #Возвращаем значения 
    return ht_map, thetas, rhos


def get_lines(ht_map, n_lines, thetas, rhos, min_delta_rho, min_delta_theta):
    pass


if __name__ == '__main__':
    assert len(argv) == 9
    src_path, dst_ht_path, dst_lines_path, theta, rho,\
        n_lines, min_delta_rho, min_delta_theta = argv[1:]

    theta = float(theta)
    rho = float(rho)
    n_lines = int(n_lines)
    min_delta_rho = float(min_delta_rho)
    min_delta_theta = float(min_delta_theta)

    assert theta > 0.0
    assert rho > 0.0
    assert n_lines > 0
    assert min_delta_rho > 0.0
    assert min_delta_theta > 0.0

    image = cv2.imread(src_path, 0)
    assert image is not None

    image = image.astype(float)
    gradient = gradient_img(image)

    ht_map, thetas, rhos = hough_transform(gradient, theta, rho)
    cv2.imwrite(dst_ht_path, ht_map)

    lines = get_lines(ht_map, thetas, rhos, n_lines, min_delta_rho, min_delta_theta)
    with open(dst_lines_path, 'w') as fout:
        for line in lines:
            fout.write('%0.3f, %0.3f\n' % line)

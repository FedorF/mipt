import json
import math
import random
import os.path
from sys import argv

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import scipy


def generate_data(img_size, line_params, n_points, sigma, inlier_ratio):
    w = img_size[0]
    h = img_size[1]

    a = line_params[0]
    b = line_params[1]
    c = line_params[2]
    n_inliers = int(n_points * inlier_ratio)
    n_outliers = int(n_points * (1 - inlier_ratio))

    l = np.random.rand(n_inliers) * (w ** 2 + h ** 2) ** (.5)
    l_1 = np.random.normal(loc=0.0, scale=sigma, size=n_inliers)

    x = - c / a + l * np.cos(np.arctan(- a / b)) - l_1 * np.sin(np.arctan(- a / b))
    y = - c / b + l * np.sin(np.arctan(- a / b)) + l_1 * np.cos(np.arctan(- a / b))

    solt_x = np.random.uniform(low=0.0, high=w, size=n_outliers)
    solt_y = np.random.uniform(low=0.0, high=h, size=n_outliers)

    return np.array(zip(np.append(x, solt_x), np.append(y, solt_y)))


def compute_ransac_thresh(alpha, sigma):
    return math.sqrt(scipy.stats.chi2.ppf(alpha, 1)) * sigma


def compute_ransac_iter_count(conv_prob, inlier_ratio):
    return int(math.log(1 - conv_prob) / math.log(1 - inlier_ratio ** 2)) + 1


def compute_line_ransac(data, inlier_ratio, t, n):
    best_line = []
    bestInNum = 0

    for i in range(n):

        p1 = random.choice(data)
        p2 = random.choice(data)

        inliers = 0
        line_points = []

        for p3 in data:

            d = norm(np.cross(p2 - p1, p1 - p3)) / norm(p2 - p1)

            if (d > 0 and d <= t):
                line_points.append(p3)
                inliers += 1

        if (inliers >= inlier_ratio * len(data) and inliers > bestInNum):
            bestInNum = inliers
            best_line = list(line_points)

    mean = lambda x: sum(x) / len(x)
    power = lambda x, y: np.array(x) * np.array(y)

    if (len(best_line) > 0):
        x = [x[0] for x in best_line]
        y = [y[1] for y in best_line]

        b = (mean(power(x, y)) - mean(x) * mean(y)) / (mean(power(x, x)) - mean(x) ** 2)
        a = mean(y) - b * mean(x)

        y_line = [b * x1 + a for x1 in x]

        return zip(x, y_line)

    return None


def main():
    print(argv)
    assert len(argv) == 2
    assert os.path.exists(argv[1])

    with open(argv[1]) as fin:
        params = json.load(fin)

    """
    params:
    line_params: (a,b,c) - line params (ax+by+c=0)
    img_size: (w, h) - size of the image
    n_points: count of points to be used

    sigma - Gaussian noise
    alpha - probability of point is an inlier

    inlier_ratio - ratio of inliers in the data
    conv_prob - probability of convergence
    """

    data = generate_data((params['w'], params['h']),
                         (params['a'], params['b'], params['c']),
                         params['n_points'], params['sigma'],
                         params['inlier_ratio'])

    t = compute_ransac_thresh(params['alpha'], params['sigma'])
    n = compute_ransac_iter_count(params['conv_prob'], params['inlier_ratio'])
    inlier_ratio = params['inlier_ratio']

    detected_line = compute_line_ransac(data, inlier_ratio, t, n)

    x = [x[0] for x in data]
    y = [y[1] for y in data]

    x_line = [x[0] for x in detected_line]
    y_line = [x[1] for x in detected_line]

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, 'o')
    plt.plot(x_line, y_line, 'k')
    plt.show()


if __name__ == '__main__':
    main()

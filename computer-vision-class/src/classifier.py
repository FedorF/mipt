"""
1) В качестве признаков, я использовал гистограму направленных градиентов (HOG). Для каждого изображения из train-выборки я получил вектор гистограмы.
2) Далее эти вектора использовались как объекты для обучения. Обучил я две модели, и качество измерял с помощью кросс-валидации и меры F.
3) SVM дал следующий усредненный скор на кросс-валидации: 0.78261 (+/- 0.01162)
 GradientBoosting: 0.99943 (+/- 0.00128)
4) Далее я получил HOG для изображений из test-выборки, и предсказал классы с помощью обученной модели GradientBoosting.
К моему удивлению score на Kaggle оказался таким: 0.84800

"""

import csv
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm


def my_hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    bin_n = 16

    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n * ang / (2 * np.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    norma = np.sum(hist * hist)
    hist = hist / np.sqrt(norma)
    return hist


with open('train_labels.csv') as labels:
    reader = csv.DictReader(labels)
    img_files = []
    y_labels = []
    for row in reader:
        img_files.append('./Train/' + row['category'] + '/' + row['id'])
        y_labels.append(int(row['category']))

print(len(y_labels), len(img_files))

x_data = []

for filename in img_files:
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hog_desc = my_hog(img)
    x_data.append(hog_desc)

print(len(x_data))

clf = svm.SVC()
scores = cross_val_score(clf, x_data, y_labels, scoring=('f1_weighted'), cv=6)
print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

gb = GradientBoostingClassifier()
scores = cross_val_score(gb, x_data, y_labels, scoring=('f1_weighted'), cv=6)
print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

test_files = os.listdir('Test')
print(len(test_files))
print(test_files[:10])

descriptors = []

for filename in test_files:
    img = cv2.imread('Test/' + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    descriptors.append(my_hog(img))

gb.fit(x_data, y_labels)
label = gb.predict(descriptors)

df = pd.DataFrame()
df['# fname'] = test_files
df['class'] = label
df.to_csv('result.csv', index=False, header=True)

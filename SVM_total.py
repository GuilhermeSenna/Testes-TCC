from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
import sys
from skimage.filters import sobel
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn import preprocessing
import timeit
from skimage.measure import shannon_entropy
import scipy.stats as fo
from skimage.filters import gabor
from radiomics import glcm

# Resize images to
SIZE = 300

# Numpy Array (explicação)
# The main benefits of using NumPy arrays should be smaller memory consumption and better runtime behavior.

def train_set():
    # Imagens de treino e suas classes
    train_images = []
    train_labels = []
    train_medias = []
    media_RGB = []

    # Varre a pasta de treino pegando as fotos da pasta
    for directory_path in glob.glob("amendoas/train/*"):
        label = directory_path.split("\\")[-1]
        for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
            #
            # imagem_colorida = cv2.imread(img_path)
            # # train_medias.append(cv2.mean(imagem_colorida))
            # imagem_colorida = np.array(imagem_colorida, dtype=float)
            # imagem_colorida[imagem_colorida == 0] = np.nan
            #
            # train_medias.append(np.nanmean(imagem_colorida, axis=(0, 1)))

            media_RGB = cv2.mean(cv2.imread(img_path))
            train_medias.append(media_RGB)

            img = cv2.imread(img_path, 0)  # Reading color images
            img = cv2.resize(img, (SIZE, SIZE))  # Resize images
            train_images.append(img)
            train_labels.append(label)

            # print(media_RGB)

            # media_cinza = cv2.mean(img)[0]
            #
            # media_RGB[0] += media_cinza
            # media_RGB[1] += media_cinza
            # media_RGB[2] += media_cinza



            # print(media_RGB)

    # Coleção total de fotos de treino e classes em NP.array
    return np.array(train_images), np.array(train_labels), np.array(train_medias)


def test_set():
    # Imagens de teste e suas classes
    test_images = []
    test_labels = []
    test_medias = []
    media_RGB = []

    # Varre a pasta de teste pegando as fotos da pasta
    for directory_path in glob.glob("amendoas/test/*"):
        fruit_label = directory_path.split("\\")[-1]
        for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):

            #
            # imagem_colorida = cv2.imread(img_path)
            # # train_medias.append(cv2.mean(imagem_colorida))
            # imagem_colorida = np.array(imagem_colorida, dtype=float)
            # imagem_colorida[imagem_colorida == 0] = np.nan
            # test_medias.append(np.nanmean(imagem_colorida, axis=(0, 1)))

            media_RGB = cv2.mean(cv2.imread(img_path))
            test_medias.append(media_RGB)

            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (SIZE, SIZE))  # Resize images
            test_images.append(img)
            test_labels.append(fruit_label)

            # print(media_RGB)
            #
            # media_cinza = cv2.mean(img)[0]
            #
            # media_RGB[0] += media_cinza
            # media_RGB[1] += media_cinza
            # media_RGB[2] += media_cinza

            # print(media_RGB)

    # Coleção total de fotos de teste e classes em NP.array
    return np.array(test_images), np.array(test_labels), np.array(test_medias)


def pre_processing(le, train_labels, test_labels):
    # Codifica os labels em valores entre 0 e n_classes - 1 (inteiros)
    # Tanto pro treino quanto pro teste
    le.fit(train_labels)
    train_labels_encoded = le.transform(train_labels)
    le.fit(test_labels)
    test_labels_encoded = le.transform(test_labels)

    return train_labels_encoded, test_labels_encoded

# print(iris)
#
# X = iris.data[:, :2]
# y = iris.target
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=101)

train_images, train_labels, medias_RGB_treino = train_set()
test_images, test_labels, medias_RGB_teste = test_set()

X_train, X_test = train_images, test_images

le = preprocessing.LabelEncoder()
y_train, y_test = pre_processing(le, train_labels, test_labels)  # Codificando nome das Classes para inteiros (0 ... nº classes - 1)


nsamples, nx, ny = X_train.shape
d2_train_dataset = X_train.reshape((nsamples,nx*ny))

print(d2_train_dataset)

rbf = svm.SVC(decision_function_shape='ovo').fit(d2_train_dataset, y_train)
poly = svm.SVC(decision_function_shape='ovo').fit(d2_train_dataset, y_train)

nsamples, nx, ny = X_test.shape
d2_test_dataset = X_test.reshape((nsamples,nx*ny))

poly_pred = poly.predict(d2_test_dataset)
rbf_pred = rbf.predict(d2_test_dataset)

poly_accuracy = accuracy_score(y_test, poly_pred)
poly_f1 = f1_score(y_test, poly_pred, average='weighted')
print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

rbf_accuracy = accuracy_score(y_test, rbf_pred)
rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))

from sklearn import metrics
from sklearn.metrics import confusion_matrix

test_prediction = le.inverse_transform(poly_pred)

print(metrics.classification_report(test_labels, test_prediction))

cm = confusion_matrix(test_labels, test_prediction)

fig, ax = plt.subplots(figsize=(6, 6))  # Sample figsize in inches
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)

plt.show()
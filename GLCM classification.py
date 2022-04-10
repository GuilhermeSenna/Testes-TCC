# https://youtu.be/5x-CIHRmMNY
"""
@author: Sreenivas Bhattiprolu
skimage.feature.greycomatrix(image, distances, angles, levels=None, symmetric=False, normed=False)
distances - List of pixel pair distance offsets.
angles - List of pixel pair angles in radians.
skimage.feature.greycoprops(P, prop)
prop: The property of the GLCM to compute.
{‘contrast’, ‘dissimilarity’, ‘homogeneity’, ‘energy’, ‘correlation’, ‘ASM’}
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
import sys
from skimage.filters import sobel
from skimage.feature import graycomatrix, graycoprops
from sklearn import preprocessing
import timeit
from skimage.measure import shannon_entropy

# print(os.listdir("images/natural/"))

# Resize images to
# SIZE = 128

# Numpy Array (explicação)
# The main benefits of using NumPy arrays should be smaller memory consumption and better runtime behavior.


def train_set():
    # Imagens de treino e suas classes
    train_images = []
    train_labels = []

    # Varre a pasta de treino pegando as fotos da pasta
    for directory_path in glob.glob("folhas/train/*"):
        label = directory_path.split("\\")[-1]
        for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
            img = cv2.imread(img_path, 0)  # Reading color images
            img = cv2.resize(img, (256, 256))  # Resize images
            train_images.append(img)
            train_labels.append(label)

    # Coleção total de fotos de treino e classes em NP.array
    return np.array(train_images), np.array(train_labels)


def test_set():
    # Imagens de teste e suas classes
    test_images = []
    test_labels = []

    # Varre a pasta de teste pegando as fotos da pasta
    for directory_path in glob.glob("folhas/test/*"):
        fruit_label = directory_path.split("\\")[-1]
        for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (256, 256))  # Resize images
            test_images.append(img)
            test_labels.append(fruit_label)

    # Coleção total de fotos de teste e classes em NP.array
    return np.array(test_images), np.array(test_labels)


def pre_processing(le, train_labels, test_labels):
    # Codifica os labels em valores entre 0 e n_classes - 1 (inteiros)
    # Tanto pro treino quanto pro teste
    le.fit(train_labels)
    train_labels_encoded = le.transform(train_labels)
    le.fit(test_labels)
    test_labels_encoded = le.transform(test_labels)

    return train_labels_encoded, test_labels_encoded


###################################################################
# FEATURE EXTRACTOR function
# input shape is (n, x, y, c) - number of images, x, y, and channels
def feature_extractor(dataset):
    image_dataset = pd.DataFrame()

    for image in range(dataset.shape[0]):  # iterate through each file

        # Dataframe temporário para manter os valores
        # Reseta o dataframe a cada loop
        df = pd.DataFrame()

        # Obtendo a imagem do dataset
        img = dataset[image]

        # Ângulos
        # pi/4 - 45°
        # pi/2 - 90º
        configs = [
            {'distancia': [1], 'angulo': [0]},
            {'distancia': [2], 'angulo': [0]},
            {'distancia': [3], 'angulo': [0]},
            {'distancia': [1], 'angulo': [np.pi / 4]},
            {'distancia': [1], 'angulo': [np.pi / 2]},
        ]

        # Adicionando dados ao Dataframe

        # Atributos considerados:
        # Energia, Correlação, dissimilaridade, homogeneidade, contraste e entropia
        for n, config in enumerate(configs):
            GLCM = graycomatrix(img, config["distancia"], config["angulo"])
            GLCM_Energy = graycoprops(GLCM, 'energy')[0]
            df['Energy' + str(n+1)] = GLCM_Energy
            GLCM_corr = graycoprops(GLCM, 'correlation')[0]
            df['Corr' + str(n+1)] = GLCM_corr
            GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
            df['Diss_sim' + str(n+1)] = GLCM_diss
            GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]
            df['Homogen' + str(n+1)] = GLCM_hom
            GLCM_contr = graycoprops(GLCM, 'contrast')[0]
            df['Contrast' + str(n+1)] = GLCM_contr
            entropy = shannon_entropy(img)
            df['Entropy' + str(n+1)] = entropy

        # Append features from current image to the dataset
        image_dataset = image_dataset.append(df)

    return image_dataset


# Monta a coleção de imagens de treino e teste
train_images, train_labels = train_set()
test_images, test_labels = test_set()

# Atribui os valores para a padronização utilizada
# x_train: Imagens de treino
# y_train: Classes de treino

# x_test: Imagens de teste
# y_test: Classes de teste

x_train, x_test = train_images, test_images

# Instância do codificador
le = preprocessing.LabelEncoder()
y_train, y_test = pre_processing(le, train_labels, test_labels)  # Classes para inteiros (0 ... nº classes - 1)

# Normalize pixel values to between 0 and 1
# x_train, x_test = x_train / 255.0, x_test / 255.0

####################################################################
# Extração de atributos das imagens de treino
image_features = feature_extractor(x_train)
X_for_ML = image_features

"""
Reshape to a vector for Random Forest / SVM training
n_features = image_features.shape[1]
image_features = np.expand_dims(image_features, axis=0)
X_for_ML = np.reshape(image_features, (x_train.shape[0], -1))  #Reshape to #images, features

Define the classifier
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)

Can also use SVM but RF is faster and may be more accurate.
from sklearn import svm
SVM_model = svm.SVC(decision_function_shape='ovo')  #For multiclass classification
SVM_model.fit(X_for_ML, y_train)

Fit the model on training data
RF_model.fit(X_for_ML, y_train) #For sklearn no one hot encoding
"""

import lightgbm as lgb

# Class names for LGBM start at 0 so reassigning labels from 1,2,3,4 to 0,1,2,3
d_train = lgb.Dataset(X_for_ML, label=y_train)

# https://lightgbm.readthedocs.io/en/latest/Parameters.html

lgbm_params = {'learning_rate': 0.05,
               'boosting_type': 'goss',
               'objective': 'multiclass',
               'metric': 'multi_logloss',
               'num_leaves': 100,
               'max_depth': 10,
               'num_class': 4}

lgb_model = lgb.train(lgbm_params, d_train, 100)  # 50 iterations. Increase iterations for small learning rates

# Predict on Test data
# Extract features from test data and reshape, just like training data
test_features = feature_extractor(x_test)
test_features = np.expand_dims(test_features, axis=0)
test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))

# Predict on test
test_prediction = lgb_model.predict(test_for_RF)
test_prediction = np.argmax(test_prediction, axis=1)
# Inverse le transform to get original label back.
test_prediction = le.inverse_transform(test_prediction)

# Print overall accuracy
from sklearn import metrics

print("Accuracy = ", metrics.accuracy_score(test_labels, test_prediction))

# Print confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, test_prediction)

# print(train_labels)

fig, ax = plt.subplots(figsize=(6, 6))  # Sample figsize in inches
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)

# Check results on a few random images
import random

for c in range(10):
    n = random.randint(0, x_test.shape[0] - 1)  # Select the index of image to be loaded for testing
    img = x_test[n]
    plt.imshow(img)
    plt.show()

    # Extract features and reshape to right dimensions
    input_img = np.expand_dims(img, axis=0)  # Expand dims so the input is (num images, x, y, c)
    input_img_features = feature_extractor(input_img)
    input_img_features = np.expand_dims(input_img_features, axis=0)
    input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))
    # Predict
    img_prediction = lgb_model.predict(input_img_for_RF)
    img_prediction = np.argmax(img_prediction, axis=1)
    img_prediction = le.inverse_transform([img_prediction])  # Reverse the label encoder to original name
    print("The prediction for this image is: ", img_prediction)
    print("The actual label for this image is: ", test_labels[n])
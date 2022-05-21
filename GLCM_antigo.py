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

SIZE = 256

# Capture images and labels into arrays.
# Start by creating empty lists.
train_images = []
train_labels = []
# for directory_path in glob.glob("cell_images/train/*"):
for directory_path in glob.glob("folhas/train/*"):
    label = directory_path.split("\\")[-1]
    # print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        # print(img_path)
        img = cv2.imread(img_path, 0)  # Reading color images
        img = cv2.resize(img, (SIZE, SIZE))  # Resize images
        train_images.append(img)
        train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_images = []
test_labels = []
# for directory_path in glob.glob("cell_images/test/*"):
for directory_path in glob.glob("folhas/test/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (SIZE, SIZE))
        test_images.append(img)
        test_labels.append(fruit_label)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded


def feature_extractor(dataset):
    image_dataset = pd.DataFrame()

    for image in range(dataset.shape[0]):  # iterate through each file
        # print(image)

        df = pd.DataFrame()  # Temporary data frame to capture information for each loop.
        # Reset dataframe to blank after each loop.

        img = dataset[image, :, :]
        ################################################################
        # START ADDING DATA TO THE DATAFRAME

        # Full image
        # GLCM = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
        GLCM = graycomatrix(img, [1], [0])
        GLCM_Energy = graycoprops(GLCM, 'energy')[0]
        df['Energy'] = GLCM_Energy
        GLCM_corr = graycoprops(GLCM, 'correlation')[0]
        df['Corr'] = GLCM_corr
        GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
        df['Diss_sim'] = GLCM_diss
        GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]
        df['Homogen'] = GLCM_hom
        GLCM_contr = graycoprops(GLCM, 'contrast')[0]
        df['Contrast'] = GLCM_contr

        GLCM2 = graycomatrix(img, [3], [0])
        GLCM_Energy2 = graycoprops(GLCM2, 'energy')[0]
        df['Energy2'] = GLCM_Energy2
        GLCM_corr2 = graycoprops(GLCM2, 'correlation')[0]
        df['Corr2'] = GLCM_corr2
        GLCM_diss2 = graycoprops(GLCM2, 'dissimilarity')[0]
        df['Diss_sim2'] = GLCM_diss2
        GLCM_hom2 = graycoprops(GLCM2, 'homogeneity')[0]
        df['Homogen2'] = GLCM_hom2
        GLCM_contr2 = graycoprops(GLCM2, 'contrast')[0]
        df['Contrast2'] = GLCM_contr2

        GLCM3 = graycomatrix(img, [5], [0])
        GLCM_Energy3 = graycoprops(GLCM3, 'energy')[0]
        df['Energy3'] = GLCM_Energy3
        GLCM_corr3 = graycoprops(GLCM3, 'correlation')[0]
        df['Corr3'] = GLCM_corr3
        GLCM_diss3 = graycoprops(GLCM3, 'dissimilarity')[0]
        df['Diss_sim3'] = GLCM_diss3
        GLCM_hom3 = graycoprops(GLCM3, 'homogeneity')[0]
        df['Homogen3'] = GLCM_hom3
        GLCM_contr3 = graycoprops(GLCM3, 'contrast')[0]
        df['Contrast3'] = GLCM_contr3

        GLCM4 = graycomatrix(img, [0], [np.pi / 4])
        GLCM_Energy4 = graycoprops(GLCM4, 'energy')[0]
        df['Energy4'] = GLCM_Energy4
        GLCM_corr4 = graycoprops(GLCM4, 'correlation')[0]
        df['Corr4'] = GLCM_corr4
        GLCM_diss4 = graycoprops(GLCM4, 'dissimilarity')[0]
        df['Diss_sim4'] = GLCM_diss4
        GLCM_hom4 = graycoprops(GLCM4, 'homogeneity')[0]
        df['Homogen4'] = GLCM_hom4
        GLCM_contr4 = graycoprops(GLCM4, 'contrast')[0]
        df['Contrast4'] = GLCM_contr4

        GLCM5 = graycomatrix(img, [0], [np.pi / 2])
        GLCM_Energy5 = graycoprops(GLCM5, 'energy')[0]
        df['Energy5'] = GLCM_Energy5
        GLCM_corr5 = graycoprops(GLCM5, 'correlation')[0]
        df['Corr5'] = GLCM_corr5
        GLCM_diss5 = graycoprops(GLCM5, 'dissimilarity')[0]
        df['Diss_sim5'] = GLCM_diss5
        GLCM_hom5 = graycoprops(GLCM5, 'homogeneity')[0]
        df['Homogen5'] = GLCM_hom5
        GLCM_contr5 = graycoprops(GLCM5, 'contrast')[0]
        df['Contrast5'] = GLCM_contr5

        # Add more filters as needed
        # entropy = shannon_entropy(img)
        # df['Entropy'] = entropy

        image_dataset = image_dataset.append(df)

    return image_dataset

# Numpy Array (explicação)
# The main benefits of using NumPy arrays should be smaller memory consumption and better runtime behavior.


####################################################################
# Extract features from training images
image_features = feature_extractor(x_train)
X_for_ML = image_features
# Reshape to a vector for Random Forest / SVM training
# n_features = image_features.shape[1]
# image_features = np.expand_dims(image_features, axis=0)
# X_for_ML = np.reshape(image_features, (x_train.shape[0], -1))  #Reshape to #images, features



import lightgbm as lgb

# Class names for LGBM start at 0 so reassigning labels from 1,2,3,4 to 0,1,2,3
d_train = lgb.Dataset(X_for_ML, label=y_train)

# https://lightgbm.readthedocs.io/en/latest/Parameters.html


lgbm_params = {
                'learning_rate': 0.05,
               'boosting_type': 'goss',
               'objective': 'multiclass',
               'metric': 'multi_logloss',
               'num_leaves': 100,
               'max_depth': 10,            # Máxima profundidade da árvore
               'num_class': 2  # Nº de classes utilizadas - Obtém dinamicamente com base dos valores distintos da lista
               }

# 100 iterações
lgb_model = lgb.train(lgbm_params, d_train, 100)

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
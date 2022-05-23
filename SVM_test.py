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
    for directory_path in glob.glob("amendoas/train_/*"):
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
    for directory_path in glob.glob("amendoas/test_/*"):
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


###################################################################
# FEATURE EXTRACTOR function
# input shape is (n, x, y, c) - number of images, x, y, and channels
def feature_extractor(dataset, medias):
    image_dataset = pd.DataFrame()

    for i, image in enumerate(range(dataset.shape[0])):  # iterate through each file
        # Dataframe temporário para manter os valores
        # Reseta o dataframe a cada loop
        df = pd.DataFrame()

        # Obtendo a imagem do dataset
        img = dataset[image, :, :]

        # Ângulos
        # pi/4 - 45°
        # pi/2 - 90º
        configs = [
            {'distancia': [1], 'angulo': [0]},
            {'distancia': [3], 'angulo': [0]},
            {'distancia': [5], 'angulo': [0]},
            {'distancia': [1], 'angulo': [np.pi / 4]},
            {'distancia': [1], 'angulo': [np.pi / 2]},
        ]

        # Adicionando dados ao Dataframe

        # Atributos considerados:
        # Energia, Correlação, dissimilaridade, homogeneidade, contraste e entropia

        n_black = cv2.countNonZero(img)
        height, width = img.shape
        n_total = height * width

        # print(f'R: {medias[i][0]}; G: {medias[i][1]}; B: {medias[i][2]}')

        # print(medias[i])

        # media_cinza = cv2.mean(img)[0]
        #

        R = medias[i][0]
        G = medias[i][1]
        B = medias[i][2]

        df['R'] = [R]
        df['G'] = [G]
        df['B'] = [B]
        #
        df['skew'] = [fo.skew(img.reshape(-1))]
        df['kurtosis'] = [fo.kurtosis(img.reshape(-1))]
        # # df['variation' + str(n + 1)] = fo.variation(img.reshape(-1))
        df['entropy'] = [fo.entropy(img.reshape(-1))]
        # # df['media'] = np.average(img)
        area = n_black / n_total
        df['Area'] = [area]
        # # df['max' + str(n+1)] = np.max(img)
        entropy = shannon_entropy(img)
        df['Entropy'] = entropy


        # df['RGB'] = medias[i][0] + medias[i][1] + medias[i][2] / 3

        # print(cv2.mean(img)[0])
        # df['cinza'] = cv2.mean(img)[0]

        # print(medias[i][0])


        for n, config in enumerate(configs):
            pass

            # df['Energy' + str(n+1)] = 1
            #
            GLCM = graycomatrix(img, config["distancia"], config["angulo"])

            #
            GLCM_Energy = graycoprops(GLCM, 'energy')[0]
            df['Energy' + str(n+1)] = GLCM_Energy
            #
            GLCM_corr = graycoprops(GLCM, 'correlation')[0]
            df['Corr' + str(n+1)] = GLCM_corr
            #
            GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
            df['Diss_sim' + str(n+1)] = GLCM_diss
            #
            GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]
            df['Homogen' + str(n+1)] = GLCM_hom
            #
            GLCM_contr = graycoprops(GLCM, 'contrast')[0]
            df['Contrast' + str(n+1)] = GLCM_contr
            #
            # df['max' + str(n + 1)] = np.average(GLCM)
            # print(skew(img.reshape(-1)))

            # print(fo.describe(img.reshape(-1)))

            # print("Percentage of dark pixels:")
            # print()

            # df['min' + str(n + 1)] = np.min(GLCM)

            # print()
            # print(medias[i][1])
            # print(medias[i][2])
            #



            # print(fo.cumfreq(img.reshape(-1)).binsize)

            # print(fo.tvar(img.reshape(-1)))

            # for m,sk in enumerate(skew(img)):
            #     df['Skew' + str(n+1) + str(m+1)] = skew(img)

            # print(f' {GLCM_Energy} - {GLCM_corr} - {GLCM_diss} - {GLCM_hom} - {GLCM_contr} - {entropy}')

            #
            # GLCM_ASM = graycoprops(GLCM, 'ASM')[0]
            # df['ASM' + str(n + 1)] = GLCM_ASM
            #

            # print(img)
            # entropy = glcm.RadiomicsGLCM.getAutocorrelationFeatureValue(img)
            # df['Entropy' + str(n+1)] = entropy


        # Append features from current image to the dataset
        #


        #
        #
        # feat_lbp = local_binary_pattern(img, 5, 2, 'uniform')
        # lbp_hist, _ = np.histogram(feat_lbp, 8)
        # lbp_hist = np.array(lbp_hist, dtype=float)
        # lbp_prob = np.divide(lbp_hist, np.sum(lbp_hist))
        # lbp_energy = np.nansum(lbp_prob ** 2)
        # lbp_entropy = -np.nansum(np.multiply(lbp_prob, np.log2(lbp_prob)))
        #
        # df['lbp_energy'] = [lbp_energy]
        # df['lbp_entropy'] = [lbp_entropy]
        # #
        # gaborFilt_real, gaborFilt_imag = gabor(img, frequency=0.6)
        # gaborFilt = (gaborFilt_real ** 2 + gaborFilt_imag ** 2) // 2
        # #
        # gabor_hist, _ = np.histogram(gaborFilt, 8)
        # gabor_hist = np.array(gabor_hist, dtype=float)
        # gabor_prob = np.divide(gabor_hist, np.sum(gabor_hist))
        # gabor_energy = np.sum(gabor_prob ** 2)
        # gabor_entropy = -np.sum(np.multiply(gabor_prob, np.log2(gabor_prob)))
        #
        # df['gabor_energy'] = [gabor_energy]
        # df['gabor_entropy'] = [gabor_entropy]
        #

        # print(df)

        image_dataset = image_dataset.append(df)

    # print(image_dataset)

    return image_dataset


# Monta a coleção de imagens de treino e teste
train_images, train_labels, medias_RGB_treino = train_set()
test_images, test_labels, medias_RGB_teste = test_set()

x_train, x_test = train_images, test_images

le = preprocessing.LabelEncoder()
y_train, y_test = pre_processing(le, train_labels, test_labels)  # Codificando nome das Classes para inteiros (0 ... nº classes - 1)

image_features = feature_extractor(x_train, medias_RGB_treino)
X_for_ML = image_features

classifier = svm.SVC(kernel='rbf', random_state = 1)
classifier.fit(X_for_ML, y_train)

Y_pred = classifier.predict(X_for_ML)

test_set["Predictions"] = Y_pred

# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=101)
#
# rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_for_ML, y_train)
# poly = svm.SVC(decision_function_shape='ovo').fit(X_for_ML, y_train)
#
# test_features = feature_extractor(x_test, medias_RGB_teste)
# test_features = np.expand_dims(test_features, axis=0)
# test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))
#
# # test_prediction = SVM_model.predict(test_for_RF)
# # test_prediction = np.argmax(test_prediction, axis=1)
#
# poly_pred = poly.predict(test_for_RF)
# rbf_pred = rbf.predict(test_for_RF)
#
# poly_accuracy = accuracy_score(y_test, poly_pred)
# poly_f1 = f1_score(y_test, poly_pred, average='weighted')
# print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
# print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))
#
# rbf_accuracy = accuracy_score(y_test, rbf_pred)
# rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
# print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
# print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))
#
# test_prediction = le.inverse_transform(poly_pred)
#
# from sklearn.metrics import confusion_matrix
#
# cm = confusion_matrix(test_labels, poly_pred)
#
# fig, ax = plt.subplots(figsize=(6, 6))  # Sample figsize in inches
# sns.set(font_scale=1.6)
# sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)
# plt.show()
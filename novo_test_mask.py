import cv2
import numpy as np
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image

path = r'C:\Users\guilh\PycharmProjects\Novo-TCC\amendoas\test\Agglutinated\A_094_p7_v3.JPG'

imagem_colorida = cv2.imread(path)
black_pixels_mask = np.all(imagem_colorida == [0, 0, 0], axis=-1)
non_black_pixels_mask = np.any(imagem_colorida != [0, 0, 0], axis=-1)

image_copy = imagem_colorida.copy()

print(cv2.mean(imagem_colorida))
print(cv2.mean(image_copy[non_black_pixels_mask]))

# image_copy[black_pixels_mask] = [255, 255, 255]
# image_copy[non_black_pixels_mask] = [0, 0, 0]

# plt.imshow(image_copy)
# plt.show()

#
#
#

train_images = []
train_labels = []
train_medias = []

red = []
green = []
blue = []

# Varre a pasta de treino pegando as fotos da pasta

# pasta = "amendoas/train/Agglutinated/*"
pasta = "amendoas/mais_testes/*"

# for directory_path in glob.glob(pasta):
#     label = directory_path.split("\\")[-1]
#     # print()
#
#     # result = cv2.mean(cv2.imread(directory_path))
#     imagem_colorida = cv2.imread(directory_path)
#
#     image_copy = imagem_colorida.copy()
#
#
#     # black_pixels_mask = np.all(imagem_colorida == [0, 0, 0], axis=-1)
#     non_black_pixels_mask = np.any(imagem_colorida != [0, 0, 0], axis=-1)
#     # result = cv2.mean(image_copy[non_black_pixels_mask])
#     result = np.mean(imagem_colorida, axis=(0, 1))
#
#     # print(image_copy[non_black_pixels_mask])
#
#     print(image_copy[non_black_pixels_mask])
#
#     plt.imshow(image_copy[non_black_pixels_mask])
#     plt.show()
#     break
#
#     print(np.mean(image_copy[non_black_pixels_mask], axis=(0, 1)))
#
#     # print(result)
#     # print()
#
#
#
#     # print(result)
#     red.append(result[0])
#     green.append(result[1])
#     blue.append(result[2])
#     # for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
#         #
#         # imagem_colorida = cv2.imread(img_path)
#         # # train_medias.append(cv2.mean(imagem_colorida))
#         # imagem_colorida = np.array(imagem_colorida, dtype=float)
#         # imagem_colorida[imagem_colorida == 0] = np.nan
#
#         # train_medias.append(np.nanmean(imagem_colorida, axis=(0, 1)))
#
#
#
#         # train_medias.append()
#
#         # img = cv2.imread(img_path, 0)  # Reading color images
#         # img = cv2.resize(img, (SIZE, SIZE))  # Resize images
#         # train_images.append(img)
#         # train_labels.append(label)

# plt.imshow(image_copy)
# plt.show()

# print(train_medias)
# print(train_medias.sort())
# train_medias.sort()
# print(train_medias)
#
# print(red)
# print(green)
# print(blue)
#
# cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100
#
# print(f'Red - {cv(red)}')
# print(f'Green - {cv(green)}')
# print(f'Blue - {cv(blue)}')

# Sem pegar os pixeis de interesse
# Red - 29.653654653514877
# Green - 27.430025479557152
# Blue - 28.403647506956233

teste = [ 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4]

plt.hist(teste, density=True)  # density=False would make counts
plt.title("teste1")
plt.ylabel('Probability')
plt.xlabel('Data')
plt.show()

# plt.hist(green, density=True, bins=30)  # density=False would make counts
# plt.title("teste2")
# plt.ylabel('Probability')
# plt.xlabel('Data')
# plt.show()
#
# plt.hist(blue, density=True, bins=30)  # density=False would make counts
# plt.title("teste3")
# plt.ylabel('Probability')
# plt.xlabel('Data')
# plt.show()


# [...,::-1]

# print(imagem_colorida)
#
# plt.imshow(imagem_colorida)
# # plt.imshow(imagem_colorida, cmap='gray')
# plt.show()
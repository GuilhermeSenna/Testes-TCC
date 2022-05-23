import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from random import randint


path = r'C:\Users\guilh\PycharmProjects\Novo-TCC\amendoas\test\Aglutinada\A_095_p7_v3.JPG'

tamanho = 4096

imagem_colorida = cv2.imread(path)
# imagem_colorida_redimensionada = cv2.resize(imagem_colorida, (tamanho, tamanho ))
imagem_cinza = cv2.cvtColor(imagem_colorida, cv2.COLOR_RGB2GRAY)

blur = cv2.blur(imagem_cinza, (15,15))
plt.imshow(blur, cmap='gray')
plt.show()

ret, thresh = cv2.threshold(blur, 10, 255, 0) # threshold (essa etapa seria equivalente a segmentação por redes neurais)
plt.imshow(thresh, cmap='gray')
plt.show()

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # acha os contornos
for i, c in enumerate(contours): # desenha os contornos na imagem com cores aleatórias para cada contorno
    img_contourn = cv2.drawContours(imagem_cinza, contours, i, (randint(0,255),randint(0,255),randint(0,255)), 3)
plt.imshow(img_contourn) # perceba que um contorno é sempre o retangulo mais ext
plt.show()

print('Contorno retangular externo:', contours[0], 'possui %i pontos' % len(contours[0]))

area = cv2.contourArea(contours[0])
print('A área em pixels é de %i pixels.' % area)
#
plt.imshow(imagem_cinza, cmap='gray')
plt.show()

width, height = Image.open(path).size
# print(width*height)

coordenadas_naozero = np.nonzero(imagem_colorida)

image_array = np.array(imagem_colorida)

# print(np.where(image_array == 0))

# image_array[image_array == 0] = np.nan


# print(image_array)

# print(len(coordenadas_naozero))
#
# print(len(coordenadas_naozero[0]))
# print(len(coordenadas_naozero[1]))
# print(len(coordenadas_naozero[2]))

# 3124893

R = 0
G = 0
B = 0

pixeis_coloridos = np.nonzero(imagem_colorida)

# for pixel in imagem_colorida:
#     print(pixel)
# print(imagem_colorida)

#
# print(np.mean(imagem_colorida.sum(1)/(imagem_colorida != 0).sum(1)))

# print(np.true_divide(imagem_colorida.sum(1),(imagem_colorida!=0).sum(1)))

# print(type(imagem_colorida))

# print(pixeis_coloridos[2])

imagem_colorida = np.array(imagem_colorida, dtype=float)
imagem_colorida[imagem_colorida == 0] = np.nan

# imagem_colorida[imagem_colorida == 0] = np.nan

RGB_mean = np.nanmean(imagem_colorida, axis=(0, 1))

# print(np.mean(imagem_colorida, axis=(0, 1)))
# print(cv2.mean(imagem_colorida))

print(np.average(imagem_colorida))


# plt.imshow(imagem_cinza)
# plt.show()
# print(imagem_colorida)

# (32.44997724195206, 69.25254762184414, 58.09691866310849, 0.0)
# Imagem original

# (32.42942810058594, 69.23171997070312, 58.06938171386719, 0.0)
# Redimensionada para 256

# (32.345130920410156, 69.14846420288086, 57.99433517456055, 0.0)
# 512

# (32.345130920410156, 69.14846420288086, 57.99433517456055, 0.0)
# 1024

# print(imagem_colorida.sum(axis=2))

# print(cv2.findNonZero(imagem_cinza))
#
#
# print(cv2.mean(imagem_colorida))
# print(cv2.countNonZero(imagem_colorida))


"""
@author: Sreenivas Bhattiprolu
https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html
skimage.feature.greycomatrix(image, distances, angles, levels=None, symmetric=False, normed=False)
distances - List of pixel pair distance offsets.
angles - List of pixel pair angles in radians.
skimage.feature.greycoprops(P, prop)
prop: The property of the GLCM to compute.
{‘contrast’, ‘dissimilarity’, ‘homogeneity’, ‘energy’, ‘correlation’, ‘ASM’}
"""

import matplotlib.pyplot as plt

# Gerar os GLCM (matrizes) e extrair as características / quantificar as características
from skimage.feature import graycomatrix, graycoprops
from skimage import io
import cv2
import numpy as np


def GLCM(path, patch, semente, fundo, plot):
    # Patch pequeno devido ao tamanho da imagem
    PATCH_SIZE = patch

    # image = io.imread('images_resized/Cocoa Beans/Bean_Fraction_Cocoa/BF 01_1.jpg', as_gray=True)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plt.imshow(image, cmap='gray')

    if plot:
        plt.imshow(image, cmap='gray')
        plt.show()

    #Full image
    # GLCM = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
    # a= graycoprops(GLCM, 'energy')[0, 0]

    # Y, X

    # select some patches from grassy areas of the image
    cell_locations = semente
    cell_patches = []
    for loc in cell_locations:
        cell_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                   loc[1]:loc[1] + PATCH_SIZE])


    # select some patches from sky areas of the image
    scratch_locations = fundo
    scratch_patches = []
    for loc in scratch_locations:
        scratch_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                 loc[1]:loc[1] + PATCH_SIZE])

    # compute some GLCM properties each patch
    diss_sim = []
    corr = []
    homogen = []
    energy = []
    contrast = []
    for patch in (cell_patches + scratch_patches):
        glcm = graycomatrix(patch, distances=[2], angles=[0], levels=256, symmetric=True, normed=True)
        diss_sim.append(graycoprops(glcm, 'dissimilarity')[0, 0]) #[0,0] to convert array to value
        corr.append(graycoprops(glcm, 'correlation')[0, 0])
        homogen.append(graycoprops(glcm, 'homogeneity')[0, 0])
        energy.append(graycoprops(glcm, 'energy')[0, 0])
        contrast.append(graycoprops(glcm, 'contrast')[0, 0])


    # print(energy)
    # print(contrast)


    # OPTIONAL PLOTTING for Visualization of points and patches
    # create the figure
    fig = plt.figure(figsize=(8, 8))

    fig.text(2, 0.5, 'matplotlib')

    # display original image with locations of patches
    ax = fig.add_subplot(3, 2, 1)
    ax.imshow(image, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    for (y, x) in cell_locations:
        ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
    for (y, x) in scratch_locations:
        ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
    ax.set_xlabel('Imagem original')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')

    # for each patch, plot (dissimilarity, correlation)
    ax = fig.add_subplot(3, 2, 2)
    ax.plot(diss_sim[:len(cell_patches)], corr[:len(cell_patches)], 'go',
            label='Sementes')
    ax.plot(diss_sim[len(cell_patches):], corr[len(cell_patches):], 'bo',
            label='Fundo')
    ax.set_xlabel('GLCM Dissimilarity')
    ax.set_ylabel('GLCM Correlation')
    ax.legend()

    # display the image patches
    for i, patch in enumerate(cell_patches):
        ax = fig.add_subplot(3, len(cell_patches), len(cell_patches)*1 + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray,
                  vmin=0, vmax=255)
        ax.set_xlabel('Semente %d' % (i + 1))

    for i, patch in enumerate(scratch_patches):
        ax = fig.add_subplot(3, len(scratch_patches), len(scratch_patches)*2 + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray,
                  vmin=0, vmax=255)
        ax.set_xlabel('Fundo %d' % (i + 1))


    # display the patches and plot
    fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()


# GLCM(
#     'images_resized/Cocoa Beans/Bean_Fraction_Cocoa/BF 01_1.jpg',
#     6,
#     [(17, 3), (13, 6), (16, 18), (15, 12)],
#     [(5, 5), (25, 12), (9, 23), (26, 24)],
#     True
# )

# GLCM(
#     'images_resized/Cocoa Beans/Broken_Beans_Cocoa/BR01_01.jpg',
#     5,
#     [(6, 8), (8, 20), (27, 13), (20, 12)],
#     [(5, 0), (1, 28), (28, 1), (28, 28)],
#     False
# )

# GLCM(
#     'images_resized/Cocoa Beans/Fermented_Cocoa/FR01_01.jpg',
#     6,
#     [(13, 3), (13, 12), (13, 17), (13, 23)],
#     [(3, 0), (1, 28), (28, 1), (28, 28)],
#     False
# )


# GLCM(
#     'images_resized/Cocoa Beans/Moldy_Cocoa/MD01_01.jpg',
#     6,
#     [(16, 3), (13, 12), (6, 17), (18, 25)],
#     [(3, 0), (1, 28), (28, 1), (28, 28)],
#     False
# )

GLCM(
    'images_resized/Cocoa Beans/Unfermented_Cocoa/UF05_01.jpg',
    6,
    [(12, 3), (13, 8), (13, 17), (14, 25)],
    [(0, 0), (1, 28), (28, 1), (28, 28)],
    False
)

# GLCM(
#     'images_resized/Cocoa Beans/Whole_Beans_Cocoa/WB01_01.jpg',
#     6,
#     [(5, 12), (10, 16), (17, 15), (24, 14)],
#     [(0, 0), (1, 28), (28, 1), (28, 28)],
#     False
# )
from image_process.fast_GLCM import *

import numpy as np
from skimage import data
from matplotlib import pyplot as plt
import cv2

def main():
    pass


if __name__ == '__main__':
    main()

    # img =
    img = cv2.imread(r'C:\Users\guilh\PycharmProjects\Novo-TCC\amendoas\test\Agglutinated\A_081.JPG_p4_v1.JPG', 0)
    h,w = img.shape
    glcm_mean = fast_glcm_contrast(img)

    print(sum(glcm_mean))

    plt.imshow(glcm_mean)
    plt.tight_layout()
    plt.show()
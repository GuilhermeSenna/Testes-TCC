import glob
import os
import cv2
from PIL import Image
import numpy as np


images = []

for directory_path in glob.glob("folhas/train/*"):
    label = directory_path.split("\\")[-1]
    # print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.JPG")):
        # print(img_path)
        # img = cv2.imread(img_path)  # Reading color images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = Image.open(img_path)

        imResize = img.resize((256, 256), Image.ANTIALIAS)
        imResize.save(img_path, 'JPEG', quality=100)
        images.append(img.size)


print(sorted(images, key=lambda tup: tup[1]))
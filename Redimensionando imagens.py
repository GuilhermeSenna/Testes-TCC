from PIL import Image
import os, sys
import glob

actual_path = os.path.abspath(os.getcwd())
actual_path.replace('\\', '/')

path_images = '/images/'
path_images_resize = '/images_resized/'

# class_ = 'Cocoa Beans/Bean_Fraction_Cocoa/'
# class_ = 'Cocoa Beans/Broken_Beans_Cocoa/'
# class_ = 'Cocoa Beans/Fermented_Cocoa/'
# class_ = 'Cocoa Beans/Moldy_Cocoa/'
# class_ = 'Cocoa Beans/Unfermented_Cocoa/'
class_ = 'Cocoa Beans/Whole_Beans_Cocoa/'

# C:/Users/guilh/PycharmProjects/Novo-TCC/images/Cocoa Beans/Bean_Fraction_Cocoa/
path = actual_path.replace('\\', '/') + path_images + class_
path_resize = actual_path.replace('\\', '/') + path_images_resize + class_

for image in os.listdir(path):
    if os.path.isfile(path + image):
        im = Image.open(path + image)

        imResize = im.resize((34, 34), Image.ANTIALIAS)
        imResize.save(path_resize + image, 'JPEG', quality=100)

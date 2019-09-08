# coding: utf-8

'''
Sample script for inference process
'''

from pathlib import Path
import pathlib
import numpy as np
from PIL import Image
from keras.models import load_model


def crop_resize (image_path):

    '''
    Resize input image
    '''

    image_shape = (32, 32, 3)

    #======= Crop image =======

    image = Image.open(image_path)
    length = min(image.size)
    crop = image.crop((0, 0, length, length))

    #======= Resize image =======
    
    resized = crop.resize(image_shape[:2])
    img = np.array(resized).astype("float32")
    img /= 255

    return img


if __name__ == '__main__':

    '''
    main function
    '''

    #======= Set folder path =======

    model_path = "models/model_file.hdf5"
    images_folder = "images"

    #======= Search input images =======


    folder = Path(images_folder)
    image_paths = [str(f) for f in folder.glob("*.png")]

    #======= Convert image files intpu np array =======

    images = [crop_resize(p) for p in image_paths]
    images = np.asarray(images)

    #======= Execute inference process =======

    model = load_model(model_path)
    predicted = model.predict_classes(images)
    proba = model.predict_proba(images)


    #======= Display inference result =======

    i=0

    for path in image_paths:

        print("{0}: {1}, {2}".format(path, predicted[i], proba[i][predicted[i]]))
        i += 1

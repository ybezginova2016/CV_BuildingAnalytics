import os

import cv2
import numpy as np
import tensorflow as tf
import webcolors
from tensorflow import keras

color_mapping = {
    (0, 0, 0): 1,    # background
    (255, 0, 0): 2,  # facade
    (0, 255, 0): 10, # window
    (0, 0, 255): 5,  # door
    (255, 255, 0): 11, # cornice
    (255, 128, 0): 3, # sill
    (128, 255, 0): 4, # balcony
    (255, 0, 255): 6, # blind
    (128, 0, 255): 8, # deco
    (0, 255, 255): 7, # molding
    (255, 0, 128): 12, # pillar
    (255, 255, 255): 9 # shop
}

COLORS = []
for color in color_mapping:
    rgb = color
    css3_name = None
    try:
        css3_name = webcolors.rgb_to_name(rgb)
    except ValueError:
        pass
    if css3_name:
        COLORS.append(css3_name)

    # определим, какой класс представлен каким цветом. Нас интересуют фасады под классом 2
    class_color_mapping = {}
    for color in COLORS:
        for rgb, class_label in color_mapping.items():
            if webcolors.name_to_rgb(color) == rgb:
                class_color_mapping[color] = class_label
                break

class DataGen(keras.utils.Sequence):
    def __init__(self, image_paths, batch_size=16, image_size=(128, 128), augment=False):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.class_color_mapping = class_color_mapping
        self.on_epoch_end()

    def augment(self, input_image, input_mask):
        if tf.random.uniform(()) > 0.5:
            # Random flipping of the image and mask
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        return input_image, input_mask

    def __load__(self, id_name):
        image_path = id_name
        mask_path = None

        ## Check if the file is an image or mask
        if image_path.endswith(".jpg"):
            mask_path = os.path.splitext(image_path)[0] + ".png"
        elif image_path.endswith(".png"):
            mask_path = image_path
            image_path = os.path.splitext(image_path)[0] + ".jpg"

        ## Reading Image
        image = cv2.imread(image_path, 1) # different colors for mask!
        image = cv2.resize(image, self.image_size)

        mask = np.zeros((self.image_size[0], self.image_size[1]))

        ## Reading Mask
        if mask_path is not None:
            mask_image = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            mask_image = cv2.resize(mask_image, self.image_size)

        # Convert mask image to class labels using color mapping
        for color_name, class_label in self.class_color_mapping.items():
            color_rgb = webcolors.name_to_rgb(color_name)
            color_mask = np.all(mask_image == color_rgb, axis=-1)
            mask[color_mask] = class_label

        # # Reading Mask
        # if mask_path is not None:
        #     mask_image = cv2.imread(mask_path, 0)
        #     mask_image = cv2.resize(mask_image, self.image_size)
        #     mask = mask_image

        ## Normalizing
        image = image / 255.
        mask = mask / 255.

        if self.augment:
            input_image, input_mask = self.augment(image, mask)
            image = input_image
            mask = input_mask

        return image, mask[..., np.newaxis]

    def __getitem__(self, index):
        """
        Here, we loop through each image in the batch and generate 30 augmented images with masks
        by calling __load__() multiple times, and then append them to the image and mask arrays.
        With this modification, the DataGen class will generate 30 * len(self.image_paths)
        images and masks for training, effectively increasing the size of the dataset.

        """
        if(index+1)*self.batch_size > len(self.image_paths):
            self.batch_size = len(self.image_paths) - index*self.batch_size

        image_batch = self.image_paths[index*self.batch_size : (index+1)*self.batch_size]

        image = []
        mask  = []

        for i in range(len(image_batch)):
            for j in range(5):
                _img, _mask = self.__load__(image_batch[i])
                image.append(_img)
                mask.append(_mask)

        image = np.array(image)
        mask  = np.array(mask)

        return image, mask

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.image_paths)/float(self.batch_size)))
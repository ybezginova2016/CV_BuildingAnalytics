from DataGen import DataGen
import os
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

image_size = (128, 128)
batch_size = 16
CLASSES = 12

test = "C:\\Users\\HOME\\PycharmProjects\\CV_BuildingAnalytics\\test"

test_path = os.path.abspath(test)
print(os.listdir(test_path))

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - (dice_coefficient(y_true, y_pred))

# Load the saved model with the best weights
model = load_model("C:\\Users\\HOME\\PycharmProjects\\CV_BuildingAnalytics\\model_checkpoint.h5",
                   custom_objects={'dice_coefficient': dice_coefficient, 'dice_coef_loss': dice_coef_loss})

# Get list of image file paths in test path
test_ids = sorted([os.path.join(test, filename)
                    for filename in os.listdir(test)
                    if filename.endswith('.jpg') or filename.endswith('.png')])

test_gen = DataGen(test_ids, batch_size=batch_size, image_size=image_size)
test_steps = int(np.ceil(len(test_ids) / batch_size))

# Evaluate the model on the test set
test_loss, test_dice_coef, test_accuracy= model.evaluate(test_gen, steps=test_steps)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test Dice Coefficient:", test_dice_coef)

"""
Test Loss: 0.036116428673267365
Test Accuracy: 0.48137566447257996
Test Dice Coefficient: 0.007399423513561487
"""
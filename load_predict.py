from DataGen import DataGen
import os
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2

image_size = (128, 128)
batch_size = 16

test_jpg_path = "C:\\Users\\HOME\\PycharmProjects\\CV_BuildingAnalytics\\cmp_b0366.jpg"
test_png_path = "C:\\Users\\HOME\\PycharmProjects\\CV_BuildingAnalytics\\cmp_b0366.png"

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.cast(y_true, 'float64')
    y_pred_f = tf.keras.backend.cast(y_pred, 'float64')
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - (dice_coefficient(y_true, y_pred))

# Load the saved model with the best weights
model = load_model("C:\\Users\\HOME\\PycharmProjects\\CV_BuildingAnalytics\\model_checkpoint.h5",
                   custom_objects={'dice_coefficient': dice_coefficient, 'dice_coef_loss': dice_coef_loss})

# Create a DataGen object for test images
test_gen = DataGen([test_jpg_path], batch_size=1, image_size=image_size)

# Get the test image and mask batch
test_batch = test_gen.__getitem__(0)
test_jpg = test_batch[0]
test_png = test_batch[1]

# Make predictions on the test image batch
pred_mask = model.predict(test_jpg)

# Calculate the dice coefficient for the predicted mask
dice_coef = dice_coefficient(test_png, pred_mask)

# Display the original image, mask, and predicted mask
cv2.imshow('Original Image', test_jpg[0])
cv2.imshow('Ground Truth Mask', test_png[0,:,:,0])
cv2.imshow('Predicted Mask', pred_mask[0,:,:,0])
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Dice Coefficient:", dice_coef)

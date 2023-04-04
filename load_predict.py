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

import cv2
import numpy as np
from tensorflow.keras.models import load_model

image_size = (128, 128)
CLASSES = 12

# Load the saved model with the best weights
model = load_model("C:\\Users\\HOME\\PycharmProjects\\CV_BuildingAnalytics\\model_checkpoint.h5")

# Load the test image and mask
test_jpg = cv2.imread('C:\\Users\\HOME\\PycharmProjects\\CV_BuildingAnalytics\\cmp_b0366.jpg')
test_png = cv2.imread('C:\\Users\\HOME\\PycharmProjects\\CV_BuildingAnalytics\\cmp_b0366.png', cv2.IMREAD_GRAYSCALE)

# Resize the image and mask to match the input shape of the model
test_jpg = cv2.resize(test_jpg, image_size)
test_png = cv2.resize(test_png, image_size)

# Normalize the image and mask pixel values
test_jpg = test_jpg / 255.0
test_png = test_png / 255.0

# Add an extra dimension to the image and mask arrays to match the model input shape
test_jpg = np.expand_dims(test_jpg, axis=0)
test_png = np.expand_dims(test_png, axis=-1)
test_png = np.expand_dims(test_png, axis=0)

# Make predictions on the test image and mask
pred_mask = model.predict(test_jpg)

# Calculate the dice coefficient for the predicted mask
dice_coef = dice_coefficient(test_png, pred_mask)

# Display the original image, mask, and predicted mask
cv2.imshow('Original Image', test_jpg[0])
cv2.imshow('Ground Truth Mask', test_png[0,:,:,0])
cv2.imshow('Predicted Mask', create_mask(pred_mask)[0,:,:,0])
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Dice Coefficient:", dice_coef)

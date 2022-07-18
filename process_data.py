from google.colab import drive
drive.mount('/content/drive')


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_PATH = "gs://"
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
IMAGE_SIZE = [224, 224]
IMAGE_SHAPE = 224
EPOCHS = 15

filenames = tf.io.gfile.glob(str(GCS_PATH + "/COVID-19_Radiography_Dataset/*/*"))

train_filenames, val_filenames = train_test_split(filenames, test_size=0.25,  random_state = 7)

train_list_ds = tf.data.Dataset.from_tensor_slices(train_filenames)
#val_list_ds = tf.data.Dataset.from_tensor_slices(val_filenames)
test_list_ds = tf.data.Dataset.from_tensor_slices(val_filenames)

TRAIN_IMG_COUNT = tf.data.experimental.cardinality(train_list_ds).numpy()
print("Training images count: " + str(TRAIN_IMG_COUNT))

VAL_IMG_COUNT = tf.data.experimental.cardinality(test_list_ds).numpy()
print("Validating images count: " + str(VAL_IMG_COUNT))

CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

COUNT_NORMAL = len([filename for filename in train_filenames if "Normal" in filename])
print("Normal images count in training set: " + str(COUNT_NORMAL))

COUNT_PNEUMONIA = len([filename for filename in train_filenames if "Viral Pneumonia" in filename])
print("Pneumonia images count in training set: " + str(COUNT_PNEUMONIA))

COUNT_COVID19 = len([filename for filename in train_filenames if "/COVID-19_Radiography_Dataset/COVID" in filename])
print("COVID19 images count in training set: " + str(COUNT_COVID19))

COUNT_OPACITY = len([filename for filename in train_filenames if "Lung_Opacity" in filename])
print("Lung Opacity images count in training set: " + str(COUNT_OPACITY))


def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASSES

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # resize the image to the desired size.
  return tf.image.resize(img, IMAGE_SIZE)

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

test_labels = [] # 5292
for f in test_list_ds:
    label = get_label(f)
    test_labels.append(label)

train_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# val_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

test_ds = test_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

labels = np.argmax(test_labels, axis = 1)
test_classes = np.array(test_labels)

test_X = []
test_y = []
for image, label in test_ds.take(5292):
    test_X.append(image.numpy())
    test_y.append(label.numpy())
np.array(test_X)
np.array(test_y)

np.save('/content/drive/MyDrive/test_X.npy', test_X)
np.save('/content/drive/MyDrive/test_y.npy', test_y) 
from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnetv2

from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet101

from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet201

from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficientb7

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception

from tensorflow.keras import Model, layers, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Concatenate, Conv2D, Flatten, Average, MaxPooling2D, Dropout, GlobalAveragePooling2D

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from plot_utils import plot_confusion_matrix, train_curves

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Device:", tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print("Number of replicas:", strategy.num_replicas_in_sync)
    
print(tf.__version__)

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

def prepare_for_training(ds, cache=True, shuffle = True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don"t
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


train_ds = prepare_for_training(train_ds)
# val_ds = prepare_for_training(val_ds)
test_ds = prepare_for_training(test_ds, shuffle=False)

def build_nn_ensemble_model():

    resnet101_model = load_model("resnet101.h5")
    densenet201_model = load_model("densenet201.h5")
    efficientb7_model = load_model("efficientb7.h5")
    xception_model = load_model("xception.h5")
    
    resnet101_model._name = "ResNet101"
    resnet101_model.trainable = False
    resnet101_base = resnet101_model.get_layer(name = "resnet101")

    densenet201_model._name = "Densenet201"
    densenet201_model.trainable = False
    densenet201_base = densenet201_model.get_layer(name = "densenet201")

    efficientb7_model._name = "EfficientNetB7"
    efficientb7_model.trainable = False
    efficientb7_base = efficientb7_model.get_layer(name = "efficientnetb7")

    xception_model._name = "Xception"
    xception_model.trainable = False
    xception_base = xception_model.get_layer(name = "xception")

    inputs = Input(shape=(224, 224, 3))

    resnet101_preprocess_input = preprocess_input_resnet101(inputs)
    resnet101_base_output = resnet101_base(resnet101_preprocess_input)
    resnet101_output = GlobalAveragePooling2D()(resnet101_base_output)

    densenet201_preprocess_input = preprocess_input_densenet201(inputs)
    densenet201_base_output = densenet201_base(densenet201_preprocess_input)
    densenet201_output = GlobalAveragePooling2D()(densenet201_base_output)

    efficientb7_preprocess_input = preprocess_input_efficientb7(inputs)
    efficientb7_base_output = efficientb7_base(efficientb7_preprocess_input)
    efficientb7_output = GlobalAveragePooling2D()(efficientb7_base_output)

    xception_preprocess_input = preprocess_input_xception(inputs)
    xception_base_output = xception_base(xception_preprocess_input)
    xception_output = GlobalAveragePooling2D()(xception_base_output)

    concat_output = Concatenate(axis=-1)([
                                        resnet101_output,
                                        densenet201_output,
                                        efficientb7_output,
                                        xception_output
                                        ])

    x = Dense(1024,activation ="relu")(concat_output)
    x = Dropout(0.5)(x)
    x = Dense(256,activation ="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(64,activation ="relu")(x)
    prediction = Dense(4, activation="softmax")(x)

    ensemble_model = Model(inputs, prediction)

    return ensemble_model



def build_conv_ensemble_model():

    resnet101_model = load_model("resnet101.h5")
    densenet201_model = load_model("densenet201.h5")
    efficientb7_model = load_model("efficientb7.h5")
    xception_model = load_model("xception.h5")
    
    resnet101_model._name = "ResNet101"
    resnet101_model.trainable = False
    resnet101_base = resnet101_model.get_layer(name = "resnet101")

    densenet201_model._name = "Densenet201"
    densenet201_model.trainable = False
    densenet201_base = densenet201_model.get_layer(name = "densenet201")

    efficientb7_model._name = "EfficientNetB7"
    efficientb7_model.trainable = False
    efficientb7_base = efficientb7_model.get_layer(name = "efficientnetb7")

    xception_model._name = "Xception"
    xception_model.trainable = False
    xception_base = xception_model.get_layer(name = "xception")

    inputs = Input(shape=(224, 224, 3))

    resnet101_preprocess_input = preprocess_input_resnet101(inputs)
    resnet101_base_output = resnet101_base(resnet101_preprocess_input)

    densenet201_preprocess_input = preprocess_input_densenet201(inputs)
    densenet201_base_output = densenet201_base(densenet201_preprocess_input)

    efficientb7_preprocess_input = preprocess_input_efficientb7(inputs)
    efficientb7_base_output = efficientb7_base(efficientb7_preprocess_input)

    xception_preprocess_input = preprocess_input_xception(inputs)
    xception_base_output = xception_base(xception_preprocess_input)

    concat_output = Concatenate(axis=-1)([
                                        resnet101_base_output,
                                        densenet201_base_output,
                                        efficientb7_base_output,
                                        xception_base_output
                                        ])

    x = Conv2D(2048, kernel_size=(2,2), strides=1, activation="relu")(concat_output)
    x = MaxPooling2D(pool_size=(2, 2), strides=1)(x)
    x = Conv2D(512, kernel_size=(2,2), strides=1, activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=1)(x)
    x = Conv2D(64, kernel_size=(2,2), strides=1, activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=1)(x)
    x = Flatten()(x)
    # x = Dense(32, activation="relu")(x)
    prediction = Dense(4, activation="softmax")(x)

    ensemble_model = Model(inputs, prediction)

    return ensemble_model



with strategy.scope():
    ensemble_model = build_nn_ensemble_model()
                     # build_conv_ensemble_model()
        
    ensemble_model.compile(optimizer = "adam",
                            loss = "categorical_crossentropy",
                            metrics=["accuracy"])

ensemble_model.summary()

ensemble_history = ensemble_model.fit(train_ds,
                                        epochs = 30,
                                        steps_per_epoch = TRAIN_IMG_COUNT // BATCH_SIZE,
                                        validation_data = test_ds,
                                        validation_steps = VAL_IMG_COUNT // BATCH_SIZE,
                                        verbose = 1,
                                        callbacks = [ModelCheckpoint("deep_ensemble.h5",
                                                                        verbose = 1,
                                                                        monitor = "val_accuracy",
                                                                        mod = "max",
                                                                        save_best_only = True)])

train_curves(ensemble_history, "Deep Ensemble CNNs")

ensemble_model = load_model("deep_ensemble.h5")    

preds = np.argmax(ensemble_model.predict(test_ds, steps = (VAL_IMG_COUNT // (BATCH_SIZE//1.4)), verbose = 1), axis = 1)
conf_matrix = confusion_matrix(labels, preds[:5292])
print(classification_report(labels, preds[:5292], target_names=["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"]))
plot_confusion_matrix(cm = conf_matrix, normalize = False,  target_names = ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"], title = "Deep Ensemble Model")
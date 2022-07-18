import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3

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

from tensorflow.keras.applications.mobilenet_v2  import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenet_v2

from tensorflow.keras import Model, layers, Input
from tensorflow.keras.layers import Dense, Concatenate, Conv2D, Flatten, Average, MaxPooling2D, Dropout, GlobalAveragePooling2D

IMAGE_SHAPE = 224

def build_vgg19_model():

    vgg19_base_model = VGG19(input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3), #(224, 224, 3),
                                include_top=False,
                                weights="imagenet")
    vgg19_base_model.trainable = True # 345

    vgg19_inputs = Input(shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
    vgg19_preprocess_input = preprocess_input_vgg19(vgg19_inputs)
    vgg19_output = vgg19_base_model(vgg19_preprocess_input)
    x = Flatten()(vgg19_output)
    x = Dense(512,activation ="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64,activation ="relu")(x)
    x = Dropout(0.2)(x)
    prediction = Dense(4,activation="softmax")(x)

    vgg19_model = Model(vgg19_inputs, prediction)

    return vgg19_model


def build_inceptionv3_model():

    inceptionv3_base_model = InceptionV3(input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3), #(224, 224, 3),
                                include_top=False,
                                weights="imagenet")
    inceptionv3_base_model.trainable = True # 345

    inceptionv3_inputs = Input(shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
    inceptionv3_preprocess_input = preprocess_input_inceptionv3(inceptionv3_inputs)
    inceptionv3_output = inceptionv3_base_model(inceptionv3_preprocess_input)
    x = Flatten()(inceptionv3_output)
    x = Dense(512,activation ="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64,activation ="relu")(x)
    x = Dropout(0.2)(x)
    prediction = Dense(4,activation="softmax")(x)

    inceptionv3_model = Model(inceptionv3_inputs, prediction)

    return inceptionv3_model


def build_resnet101_model():

    resnet101_base_model = ResNet101(input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3), #(224, 224, 3),
                                include_top=False,
                                weights="imagenet")
    resnet101_base_model.trainable = True # 345

    resnet101_inputs = Input(shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
    resnet101_preprocess_input = preprocess_input_resnet101(resnet101_inputs)
    resnet101_output = resnet101_base_model(resnet101_preprocess_input)
    x = Flatten()(resnet101_output)
    x = Dense(1024,activation ="relu")(x)
    x = Dropout(0.6)(x)
    x = Dense(512,activation ="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64,activation ="relu")(x)
    x = Dropout(0.2)(x)
    prediction = Dense(4,activation="softmax")(x)

    resnet101_model = Model(resnet101_inputs, prediction)

    return resnet101_model


def build_inception_resnetv2_model():  
    
    inception_resnetv2_base_model = InceptionResNetV2(input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3), #(224, 224, 3), 
                            include_top=False,
                            weights="imagenet")
    inception_resnetv2_base_model.trainable = True  #132

    inception_resnetv2_inputs = Input(shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
    inception_resnetv2_preprocess_input = preprocess_input_inception_resnetv2(inception_resnetv2_inputs)
    inception_resnetv2_output = inception_resnetv2_base_model(inception_resnetv2_preprocess_input)
    x = Flatten()(inception_resnetv2_output)
    x = Dense(512,activation ="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64,activation ="relu")(x)
    x = Dropout(0.2)(x)
    prediction = Dense(4,activation="softmax")(x)

    inception_resnetv2_model = Model(inception_resnetv2_inputs, prediction)
    return inception_resnetv2_model


def build_densenet201_model():

    densenet201_base_model = DenseNet201(input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3), #(224, 224, 3), 
                                    include_top=False,
                                    weights="imagenet")
    densenet201_base_model.trainable = True  #707

    densenet201_inputs = Input(shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
    densenet201_preprocess_input = preprocess_input_densenet201(densenet201_inputs)
    densenet201_output = densenet201_base_model(densenet201_preprocess_input)
    x = Flatten()(densenet201_output)
    x = Dense(512,activation ="relu")(x)
    x = Dense(64,activation ="relu")(x)
    prediction = Dense(4,activation="softmax")(x)

    densenet201_model = Model(densenet201_inputs, prediction)
    return densenet201_model


def build_xception_model():  
    
    xception_base_model = Xception(input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3), #(224, 224, 3), 
                            include_top=False,
                            weights="imagenet")
    xception_base_model.trainable = True  #132

    xception_inputs = Input(shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
    xception_preprocess_input = preprocess_input_xception(xception_inputs)
    xception_output = xception_base_model(xception_preprocess_input)
    x = Flatten()(xception_output)
    x = Dense(512,activation ="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64,activation ="relu")(x)
    x = Dropout(0.2)(x)
    prediction = Dense(4,activation="softmax")(x)

    xception_model = Model(xception_inputs, prediction)
    return xception_model


def build_efficientb7_model():

    efficientb7_base_model = EfficientNetB7(input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3), #(224, 224, 3),
                                    include_top=False, 
                                    weights="imagenet")
    efficientb7_base_model.trainable = True  #813

    efficientb7_inputs = Input(shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
    efficientb7_preprocess_input = preprocess_input_efficientb7(efficientb7_inputs)
    efficientb7_output = efficientb7_base_model(efficientb7_preprocess_input)
    x = Flatten()(efficientb7_output)
    x = Dense(512,activation ="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64,activation ="relu")(x)
    x = Dropout(0.2)(x)
    prediction = Dense(4,activation="softmax")(x)

    efficientb7_model = Model(efficientb7_inputs, prediction)
    return efficientb7_model
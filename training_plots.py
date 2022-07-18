import numpy as np
import matplotlib.pyplot as plt

conv_ensemble_history = np.load("conf_ensemble_history.npy",allow_pickle="TRUE").item()
resnet101_history = np.load("resnet101_history.npy",allow_pickle="TRUE").item()
densenet201_history = np.load("densenet201_history.npy",allow_pickle="TRUE").item()
xception_history = np.load("xception_history.npy",allow_pickle="TRUE").item()
efficientb7_history = np.load("efficientb7_history.npy",allow_pickle="TRUE").item()

resnet101_acc = resnet101_history["val_accuracy"]
densenet201_acc = densenet201_history["val_accuracy"]
xception_acc = xception_history["val_accuracy"]
efficientb7_acc = efficientb7_history["val_accuracy"]
conv_ensemble_acc = conv_ensemble_history["val_accuracy"]

plt.figure(figsize=(8, 6))

plt.plot(resnet101_acc, label="ResNet-101 Validation Accuracy")
plt.plot(densenet201_acc, label="DenseNet-201 Validation Accuracy")
plt.plot(xception_acc, label="Xception Validation Accuracy")
plt.plot(efficientb7_acc, label="EfficientNet-B7 Validation Accuracy")
plt.plot(conv_ensemble_acc, label="Ensemble Model Validation Accuracy")

plt.legend(loc="lower right")
plt.ylabel("Accuracy")
plt.xlabel("epochs")
plt.ylim([min(plt.ylim()),1])
plt.title("Validation Accuracy")
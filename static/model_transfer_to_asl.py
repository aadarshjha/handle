import os
from pickle import NONE
from telnetlib import SE
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import yaml
import pandas as pd
import gc
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from PIL import Image
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    MaxPool2D,
    Conv2D,
    BatchNormalization,
    Flatten,
)
from sklearn.metrics import confusion_matrix
from keras import Input, Model
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.vgg16 import VGG16

# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# gpus = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

PREFIX = "../../drive/MyDrive/handleData/"
PREFIX = "../data/leapGestRecog/"

# custom plotting
from plot import *

dim_x = 90
dim_y = 90

# collect the data
def read_data():
    imagepaths = []
    for root, _, files in os.walk("../data/leapGestRecog", topdown=False):
        for name in files:
            path = os.path.join(root, name)
            if path.endswith("png"):  # We want only the images
                imagepaths.append(path)

    # printing the number of images loaded:
    print("Number of images loaded: ", len(imagepaths))
    return imagepaths


def augment_data(imagepaths):

    X = []
    y = []

    # Loops through imagepaths to load images and labels into arrays
    for path in imagepaths:

        # opene image with PIL
        img = Image.open(path).convert("L")
        img = img.resize((dim_x, dim_y))
        img = np.array(img)
        X.append(img)

        # Processing label in image path
        category = path.split("/")[3]

        label = int(
            category.split("_")[0][1]
        )  # We need to convert 10_down to 00_down, or else it crashes
        y.append(label)

    X = np.array(X, dtype="float32")
    X = X.reshape(len(imagepaths), dim_y, dim_x, 1)
    print(X.shape)
    y = np.array(y)

    print("Images loaded: ", len(X))
    print("Labels loaded: ", len(y))

    # cache X and y
    print("Caching data...")
    np.save("X_augmented.npy", X)
    np.save("y_augmented.npy", y)

    return X, y


def create_model(mode, loss_fn, optimizer_algorithm, monitor_metric):
    model = None

    # load the weights of a ./hgr_domain.h5
    base_model = tf.keras.models.load_model("./asl_domain.h5")
    # freeze all layers except for the classifier
    for layer in base_model.layers[:-1]:
        layer.trainable = False

    # add a new classifier layer
    model = Sequential(
        [
            base_model,
            Flatten(),
            BatchNormalization(),
            Dense(64, activation="relu"),
            Dropout(0.5),
            BatchNormalization(),
            Dense(10, activation="softmax"),
        ]
    )
    # compile the model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def execute_training(
    X,
    y,
    mode,
    num_folds,
    epochs,
    batch_size,
    experiment_name,
    verbose,
    optimizer,
    loss,
):

    kfold = KFold(n_splits=num_folds, shuffle=True)

    fold_no = 1

    model_cache = []
    predictions_cache = []
    targets_cache = []

    train_loss = []
    train_acc = []

    val_loss = []
    val_acc = []

    precision_history = []
    recall_history = []
    f1_history = []
    accuracy_history = []
    cfx_history = []

    loss_fn = loss
    optimizer_algorithm = optimizer
    monitor_metric = ["accuracy"]

    # train across folds
    for train, test in kfold.split(X, y):

        model = create_model(mode, loss_fn, optimizer_algorithm, monitor_metric)

        print("For Fold: " + str(fold_no))
        X_val, X_test, y_val, y_test = train_test_split(
            X[test], y[test], test_size=0.5, random_state=42
        )

        X_train = X[train]
        y_train = y[train]

        X_train = tf.image.resize(X_train, (dim_x, dim_y))
        X_val = tf.image.resize(X_val, (dim_x, dim_y))
        X_test = tf.image.resize(X_test, (dim_x, dim_y))

        X_train = tf.image.grayscale_to_rgb(X_train)
        X_val = tf.image.grayscale_to_rgb(X_val)
        X_test = tf.image.grayscale_to_rgb(X_test)

        history = model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_val, y_val),
        )
        scores = model.evaluate(X_test, y_test, verbose=verbose)

        # save the predictions from the model.evaluate
        y_prob = model.predict(X_test)
        predictions = y_prob.argmax(axis=-1)

        # compute the precision, recall, f1 score, and accuracy
        precision = precision_score(y_test, predictions, average="weighted")
        recall = recall_score(y_test, predictions, average="weighted")
        f1 = f1_score(y_test, predictions, average="weighted")
        accuracy = accuracy_score(y_test, predictions)

        # cache the target values
        targets_cache.append(y_test)

        # create a confusion matrix
        cfx = confusion_matrix(y_test, predictions)
        cfx_history.append(cfx)

        # cache the above
        precision_history.append(precision)
        recall_history.append(recall)
        f1_history.append(f1)
        accuracy_history.append(accuracy)

        # cache the predictions for micro use
        predictions_cache.append(predictions)

        # save the training loss and accuracy
        train_loss.append(history.history["loss"])
        train_acc.append(history.history["accuracy"])

        # save the validation loss and accuracy
        val_loss.append(history.history["val_loss"])
        val_acc.append(history.history["val_accuracy"])

        # print the test loss and test accuracy
        print("Test loss: ", scores[0])
        print("Test accuracy: ", scores[1])

        # print the metrics:
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1: ", f1)
        print("Accuracy: ", accuracy)
        # print the confusion matrix
        print("Confusion Matrix", cfx)

        print("\n")

        model_cache.append(model)

        fold_no += 1

    return (
        model_cache,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
        predictions_cache,
        targets_cache,
        precision_history,
        recall_history,
        f1_history,
        accuracy_history,
        cfx_history,
    )


def execute_micro_macro_metrics(
    model_cache,
    predictions_cache,
    targets_cache,
    precision_history,
    recall_history,
    f1_history,
    accuracy_history,
    cfx_history,
    experiment_name,
):

    epoch_counter = 1
    for model in model_cache:
        model.save(PREFIX + "models/{}/{}.h5".format(experiment_name, epoch_counter))
        epoch_counter = epoch_counter + 1

    JSON_data = {}

    # macro averaging:
    macro_precision = np.mean(precision_history)
    macro_recall = np.mean(recall_history)
    macro_f1 = np.mean(f1_history)
    macro_accuracy = np.mean(accuracy_history)
    cfx_history_avg = np.mean(cfx_history, axis=0)
    cfx_history_avg_json = json.dumps(cfx_history_avg.tolist())

    # store into JSON:
    JSON_data["macro_metrics"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "accuracy": macro_accuracy,
        "cfx": cfx_history_avg_json,
    }

    # micro averaging:

    micro_precision = precision_score(
        np.concatenate(targets_cache),
        np.concatenate(predictions_cache),
        average="weighted",
    )
    micro_recall = recall_score(
        np.concatenate(targets_cache),
        np.concatenate(predictions_cache),
        average="weighted",
    )
    micro_f1 = f1_score(
        np.concatenate(targets_cache),
        np.concatenate(predictions_cache),
        average="weighted",
    )
    micro_accuracy = accuracy_score(
        np.concatenate(targets_cache), np.concatenate(predictions_cache)
    )

    # confusion matrix for micro averaging:
    cfx_micro = confusion_matrix(
        np.concatenate(targets_cache), np.concatenate(predictions_cache)
    )
    cfx_micro_json = json.dumps(cfx_micro.tolist())

    # store into JSON:
    JSON_data["micro_metrics"] = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": micro_f1,
        "accuracy": micro_accuracy,
        "cfx": cfx_micro_json,
    }

    # save the JSON_data to a file
    with open(
        PREFIX + "logs/" + experiment_name + "/" + experiment_name + "_data.json", "w"
    ) as outfile:
        json.dump(JSON_data, outfile)


def extract_hyperparameters(filename):
    hyperparams = None
    with open("experiments_hgr/{}.yaml".format(filename), "r") as f:
        hyperparameters = yaml.load(f, Loader=yaml.FullLoader)
    return hyperparameters


if __name__ == "__main__":

    # extract file name from command line input
    filename = sys.argv[1]
    hyperparameters = extract_hyperparameters(filename)

    if not os.path.exists(PREFIX + "models/" + hyperparameters["EXPERIMENT_NAME"]):
        os.makedirs(PREFIX + "models/" + hyperparameters["EXPERIMENT_NAME"])

    if not os.path.exists(PREFIX + "logs/" + hyperparameters["EXPERIMENT_NAME"]):
        os.makedirs(PREFIX + "logs/" + hyperparameters["EXPERIMENT_NAME"])

    # taking this precaution becuase the data takes a while to load.

    X = None
    y = None

    # if X_augmented.npy is in the current directory
    # and the file is not empty, skip this step
    if not os.path.exists("./X_augmented.npy"):
        print("Loading data...")
        # read the data
        imagepaths = read_data()

        # finalize data
        X, y = augment_data(imagepaths)
    else:
        # load the data
        print("Loading pre-saved data...")
        # X = np.load("../../drive/MyDrive/X_augmented.npy")
        # y = np.load("../../drive/MyDrive/y_augmented.npy")
        X = np.load("./X_augmented.npy")
        y = np.load("./y_augmented.npy")

        # execute the training pipeline
        (
            model_cache,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            predictions_cache,
            targets_cache,
            precision_history,
            recall_history,
            f1_history,
            accuracy_history,
            cfx_history,
        ) = execute_training(
            X,
            y,
            hyperparameters["CONFIG"]["MODE"],
            hyperparameters["CONFIG"]["NUM_FOLDS"],
            hyperparameters["CONFIG"]["EPOCHS"],
            hyperparameters["CONFIG"]["BATCH_SIZE"],
            hyperparameters["EXPERIMENT_NAME"],
            hyperparameters["CONFIG"]["VERBOSE"],
            hyperparameters["CONFIG"]["OPTIMIZER"],
            hyperparameters["CONFIG"]["LOSS"],
        )

        plot_training_validation(
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            hyperparameters["EXPERIMENT_NAME"],
            PREFIX,
        )
        execute_micro_macro_metrics(
            model_cache,
            predictions_cache,
            targets_cache,
            precision_history,
            recall_history,
            f1_history,
            accuracy_history,
            cfx_history,
            hyperparameters["EXPERIMENT_NAME"],
        )

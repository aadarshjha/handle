import os
from pickle import NONE
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import yaml
import pandas as pd
import skimage
import skimage.transform
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
import json
from sklearn.model_selection import train_test_split

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

PREFIX = "../drive/MyDrive/handleData/"


batch_size = 64
imageSize = 64
target_dims = (imageSize, imageSize, 3)
num_classes = 29

train_len = 87000
train_dir = "../drive/MyDrive/asl_alphabet_train/asl_alphabet_train/"


# custom plotting
from plot import *


def get_data(folder):
    X = np.empty((train_len, imageSize, imageSize, 3), dtype=np.float32)
    y = np.empty((train_len,), dtype=np.int)
    cnt = 0
    for folderName in os.listdir(folder):
        if not folderName.startswith("."):
            if folderName in ["A"]:
                label = 0
            elif folderName in ["B"]:
                label = 1
            elif folderName in ["C"]:
                label = 2
            elif folderName in ["D"]:
                label = 3
            elif folderName in ["E"]:
                label = 4
            elif folderName in ["F"]:
                label = 5
            elif folderName in ["G"]:
                label = 6
            elif folderName in ["H"]:
                label = 7
            elif folderName in ["I"]:
                label = 8
            elif folderName in ["J"]:
                label = 9
            elif folderName in ["K"]:
                label = 10
            elif folderName in ["L"]:
                label = 11
            elif folderName in ["M"]:
                label = 12
            elif folderName in ["N"]:
                label = 13
            elif folderName in ["O"]:
                label = 14
            elif folderName in ["P"]:
                label = 15
            elif folderName in ["Q"]:
                label = 16
            elif folderName in ["R"]:
                label = 17
            elif folderName in ["S"]:
                label = 18
            elif folderName in ["T"]:
                label = 19
            elif folderName in ["U"]:
                label = 20
            elif folderName in ["V"]:
                label = 21
            elif folderName in ["W"]:
                label = 22
            elif folderName in ["X"]:
                label = 23
            elif folderName in ["Y"]:
                label = 24
            elif folderName in ["Z"]:
                label = 25
            elif folderName in ["del"]:
                label = 26
            elif folderName in ["nothing"]:
                label = 27
            elif folderName in ["space"]:
                label = 28
            else:
                label = 29
            for image_filename in os.listdir(folder + folderName):
                img_file = cv2.imread(folder + folderName + "/" + image_filename)
                if img_file is not None:
                    img_file = skimage.transform.resize(
                        img_file, (imageSize, imageSize, 3)
                    )
                    img_arr = np.asarray(img_file).reshape(
                        (-1, imageSize, imageSize, 3)
                    )
                    X[cnt] = img_arr
                    y[cnt] = label
                    cnt += 1
    return X, y


def create_model(mode="CNN"):

    model = None
    if mode == "CNN":
        model = Sequential()
        model.add(Conv2D(32, (5, 5), activation="relu", input_shape=(64, 64, 3)))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(10, activation="softmax"))
    elif mode == "CNN_PRETRAINED":
        pass
    elif mode == "RESNET":
        model = keras.applications.resnet.ResNet50(
            include_top=False, weights=None, input_shape=(64, 64, 3)
        )
    elif mode == "RESNET_PRETRAINED":
        model = keras.applications.resnet.ResNet50(
            include_top=False, weights="imagenet", input_shaspe=(64, 64, 3)
        )
    elif mode == "MOBILENET":
        model = keras.applications.mobilenet.MobileNet(
            include_top=False, weights=None, input_shape=(64, 64, 3)
        )
    elif mode == "MOBILENET_PRETRAINED":
        model = keras.applications.mobilenet.MobileNet(
            include_top=False, weights="imagenet", input_shape=(64, 64, 3)
        )
    elif mode == "DENSENET":
        model = keras.applications.densenet.DenseNet121(
            include_top=False, weights=None, input_shape=(64, 64, 3)
        )
    elif mode == "DENSENET_PRETRAINED":
        model = keras.applications.densenet.DenseNet121(
            include_top=False, weights="imagenet", input_shape=(64, 64, 3)
        )
    else:
        # throw an error to the user
        raise Exception("Invalid model type")

    return model


def execute_training(
    X,
    y,
    experiment_name="exper1",
    num_folds=5,
    epochs=10,
    batch_size=32,
    verbose=False,
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    mode="CNN",
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

    # train across folds
    for train, test in kfold.split(X, y):

        model = create_model()
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

        print("For Fold: " + str(fold_no))
        X_val, X_test, y_val, y_test = train_test_split(
            X[test], y[test], test_size=0.5, random_state=42
        )

        history = model.fit(
            X[train],
            y[train],
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=(X_val, y_val),
        )
        scores = model.evaluate(X[test], y[test], verbose=verbose)

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

        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1: ", f1)
        print("Accuracy: ", accuracy)
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
    with open("experiments_asl/{}.yaml".format(filename), "r") as f:
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

    # read the data
    # imagepaths = read_data()

    # finalize data
    # X, y = augment_data(imagepaths)
    X, y = get_data(train_dir)

    # print(X)
    # print(y)

    # print("The shape of X_train is : ", X.shape)
    # print("The shape of y_train is : ", y.shape)

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
        hyperparameters["EXPERIMENT_NAME"],
        hyperparameters["CONFIG"]["NUM_FOLDS"],
        hyperparameters["CONFIG"]["EPOCHS"],
        hyperparameters["CONFIG"]["BATCH_SIZE"],
        hyperparameters["CONFIG"]["VERBOSE"],
        hyperparameters["CONFIG"]["OPTIMIZER"],
        hyperparameters["CONFIG"]["LOSS"],
        hyperparameters["CONFIG"]["MODE"],
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

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
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from sklearn.metrics import classification_report, confusion_matrix

# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# gpus = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

DRIVE = False

PREFIX = "../data/asl-mnist/"

# custom plotting
from plot import *


def read_data():
    train_df = pd.read_csv(
        PREFIX + "sign_mnist_train.csv",
    )
    test_df = pd.read_csv(
        PREFIX + "sign_mnist_test.csv",
    )

    return train_df, test_df


def augment_data(train_df, test_df):

    X = []
    y = []

    y_train = train_df["label"]
    y_test = test_df["label"]

    del train_df["label"]
    del test_df["label"]

    x_train = train_df.values
    x_test = test_df.values

    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    x_train = x_train / 255
    x_test = x_test / 255

    # Reshaping the data from 1-D to 3-D as required through input by CNN's
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,
    )  # randomly flip images

    # combine the x_trrarin and x_test
    X = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    print("Images loaded: ", len(X))
    print("Labels loaded: ", len(y))

    return X, y, datagen


def create_model(mode="CNN"):

    model = None
    if mode == "CNN":
        model = Sequential()
        model.add(
            Conv2D(
                75,
                (3, 3),
                strides=1,
                padding="same",
                activation="relu",
                input_shape=(28, 28, 1),
            )
        )
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2), strides=2, padding="same"))
        model.add(Conv2D(50, (3, 3), strides=1, padding="same", activation="relu"))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2), strides=2, padding="same"))
        model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2), strides=2, padding="same"))
        model.add(Flatten())
        model.add(Dense(units=512, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(units=24, activation="softmax"))
    elif mode == "CNN_PRETRAINED":
        pass
    elif mode == "RESNET":
        model = keras.applications.resnet.ResNet50(
            include_top=False, weights=None, input_shape=(28, 28, 1)
        )
    elif mode == "RESNET_PRETRAINED":
        model = keras.applications.resnet.ResNet50(
            include_top=False, weights="imagenet", input_shape=(28, 28, 1)
        )
    elif mode == "MOBILENET":
        model = keras.applications.mobilenet.MobileNet(
            include_top=False, weights=None, input_shape=(28, 28, 1)
        )
    elif mode == "MOBILENET_PRETRAINED":
        model = keras.applications.mobilenet.MobileNet(
            include_top=False, weights="imagenet", input_shape=(28, 28, 1)
        )
    elif mode == "DENSENET":
        model = keras.applications.densenet.DenseNet121(
            include_top=False, weights=None, input_shape=(28, 28, 1)
        )
    elif mode == "DENSENET_PRETRAINED":
        model = keras.applications.densenet.DenseNet121(
            include_top=False, weights="imagenet", input_shape=(28, 28, 1)
        )
    else:
        # throw an error to the user
        raise Exception("Invalid model type")

    return model


def execute_training(
    X,
    y,
    datagen,
    experiment_name="exper1",
    num_folds=5,
    epochs=10,
    batch_size=32,
    verbose=False,
    optimizer="adam",
    loss="categorical_crossentropy",
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
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        print("For Fold: " + str(fold_no))
        X_val, X_test, y_val, y_test = train_test_split(
            X[test], y[test], test_size=0.5, random_state=42
        )

        print("Training data: ", len(X[train]))
        print("Validation data: ", len(X_val))
        print("Test data: ", len(X_test))

        # fit the training data for the datagen
        datagen.fit(X[train])

        history = model.fit(
            datagen.flow(X[train], y[train], batch_size=batch_size),
            epochs=epochs,
            verbose=verbose,
            validation_data=(X_val, y_val),
        )
        scores = model.evaluate(X[test], y[test], verbose=verbose)

        # save the predictions from the model.evaluate
        y_prob = model.predict(X_test)
        predictions = y_prob.argmax(axis=-1)

        # compute the precision, recall, f1 score, and accuracy
        precision = precision_score(
            np.argmax(y_test, axis=1), predictions, average="weighted"
        )
        recall = recall_score(
            np.argmax(y_test, axis=1), predictions, average="weighted"
        )
        f1 = f1_score(np.argmax(y_test, axis=1), predictions, average="weighted")
        accuracy = accuracy_score(np.argmax(y_test, axis=1), predictions)

        # cache the target values
        targets_cache.append(y_test)

        # create a confusion matrix
        cfx = confusion_matrix(np.argmax(y_test, axis=1), predictions)
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
        np.argmax(np.concatenate(targets_cache), axis=1),
        np.concatenate(predictions_cache),
        average="weighted",
    )
    micro_f1 = f1_score(
        np.argmax(np.concatenate(targets_cache), axis=1),
        np.concatenate(predictions_cache),
        average="weighted",
    )
    micro_accuracy = accuracy_score(
        np.argmax(np.concatenate(targets_cache), axis=1),
        np.concatenate(predictions_cache),
    )

    # confusion matrix for micro averaging:
    cfx_micro = confusion_matrix(
        np.argmax(np.concatenate(targets_cache), axis=1),
        np.concatenate(predictions_cache),
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
    imagepaths = read_data()

    # finalize data
    X, y, datagen = augment_data(imagepaths[0], imagepaths[1])

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
        datagen,
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

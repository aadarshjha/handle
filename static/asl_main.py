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
    Flatten,
)
from sklearn.metrics import confusion_matrix
from keras import Input, Model
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet import ResNet50

# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# gpus = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

DRIVE = False

PREFIX = "../data/asl-mnist/"

# custom plotting
from plot import *


loss_fn = "categorical_crossentropy"
optimizer_algorithm = optimizers.RMSprop(learning_rate=1e-4)
monitor_metric = ["accuracy"]


def CNN_Model(
    input_shape=(
        300,
        300,
        3,
    ),
    n_out=24,
):
    model = Sequential()

    # Convolutional layer 1
    model.add(
        Conv2D(
            filters=8,
            kernel_size=(5, 5),
            padding="Same",
            activation="relu",
            input_shape=input_shape,
        )
    )
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Convolutional Layer 2
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="Same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Convolutional Layer 3
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="Same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Convolutional Layer 4
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="Same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(n_out, activation="softmax"))

    model.compile(loss=loss_fn, optimizer=optimizer_algorithm, metrics=monitor_metric)

    return model


def create_resnet50(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = ResNet50(
        weights="imagenet", include_top=False, input_tensor=input_tensor
    )

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation="relu")(x)
    x = Dropout(0.5)(x)
    final_output = Dense(n_out, activation="softmax", name="final_output")(x)
    model = Model(input_tensor, final_output)

    for layer in model.layers:
        layer.trainable = False

    for i in range(-5, 0):
        model.layers[i].trainable = True

    # optimizer = optimizers.Adam(lr=WARMUP_LEARNING_RATE)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def create_mobilenet(input_shape, n_out):
    base_model = MobileNet(input_shape=input_shape, include_top=False, weights=None)
    base_model.trainable = False
    model = Sequential(
        [
            base_model,
            GlobalAveragePooling2D(),
            Dense(1024, activation="relu"),
            Dense(1024, activation="relu"),
            Dense(512, activation="relu"),
            Dense(n_out, activation="sigmoid"),
        ]
    )
    model.compile(loss=loss_fn, optimizer=optimizer_algorithm, metrics=monitor_metric)
    return model


def create_mobilenet_pretrained(input_shape, n_out):

    base_model = MobileNet(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    model = Sequential(
        [
            base_model,
            GlobalAveragePooling2D(),
            Dense(1024, activation="relu"),
            Dense(1024, activation="relu"),
            Dense(512, activation="relu"),
            Dense(n_out, activation="sigmoid"),
        ]
    )
    model.compile(loss=loss_fn, optimizer=optimizer_algorithm, metrics=monitor_metric)
    return model


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
        fill_mode="nearest",
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False,
    )

    # combine the x_trrarin and x_test
    X = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    print("Images loaded: ", len(X))
    print("Labels loaded: ", len(y))

    return X, y, datagen


def create_model(mode):
    model = None
    if mode == "CNN":
        model = CNN_Model(
            (
                32,
                32,
                3,
            ),
            n_out=24,
        )
    # elif mode == "RESNET":
    #     model = keras.applications.resnet.ResNet50(
    #         include_top=False, weights=None, input_shape=(28, 28, 1)
    #     )
    elif mode == "RESNET_PRETRAINED":
        model = create_resnet50((120, 120, 3), 24)
    elif mode == "MOBILENET":
        model = create_mobilenet((32, 32, 3), 24)
    elif mode == "MOBILENET_PRETRAINED":
        model = create_mobilenet_pretrained((32, 32, 3), 24)
    # elif mode == "DENSENET":
    #     model = keras.applications.densenet.DenseNet121(
    #         include_top=False, weights=None, input_shape=(28, 28, 1)
    #     )
    # elif mode == "DENSENET_PRETRAINED":
    #     model = keras.applications.densenet.DenseNet121(
    #         include_top=False, weights="imagenet", input_shape=(28, 28, 1)
    #     )
    else:
        # throw an error to the user
        raise Exception("Invalid model type")

    return model


def execute_training(
    X,
    y,
    datagen,
    mode,
    experiment_name="exper1",
    num_folds=5,
    epochs=10,
    batch_size=32,
    verbose=False,
    optimizer="adam",
    loss="categorical_crossentropy",
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

        model = create_model(mode)

        print("For Fold: " + str(fold_no))
        X_val, X_test, y_val, y_test = train_test_split(
            X[test], y[test], test_size=0.5, random_state=42
        )

        X_train = X[train]
        y_train = y[train]

        X_train = tf.image.resize(X_train, (120, 120))
        X_val = tf.image.resize(X_val, (120, 120))
        X_test = tf.image.resize(X_test, (120, 120))

        X_train = tf.image.grayscale_to_rgb(X_train)
        X_val = tf.image.grayscale_to_rgb(X_val)
        X_test = tf.image.grayscale_to_rgb(X_test)

        # fit the training data for the datagen
        datagen.fit(X_train)

        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            verbose=verbose,
            validation_data=(X_val, y_val),
        )

        scores = model.evaluate(X_test, y_test, verbose=verbose)

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
        np.argmax(np.concatenate(targets_cache), axis=1),
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
        hyperparameters["CONFIG"]["MODE"],
        hyperparameters["EXPERIMENT_NAME"],
        hyperparameters["CONFIG"]["NUM_FOLDS"],
        hyperparameters["CONFIG"]["EPOCHS"],
        hyperparameters["CONFIG"]["BATCH_SIZE"],
        hyperparameters["CONFIG"]["VERBOSE"],
        hyperparameters["CONFIG"]["OPTIMIZER"],
        hyperparameters["CONFIG"]["LOSS"],
    )

    # plot_training_validation(
    #     train_loss,
    #     train_acc,
    #     val_loss,
    #     val_acc,
    #     hyperparameters["EXPERIMENT_NAME"],
    #     PREFIX,
    # )

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

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
from keras.applications import VGG16

from keras import backend as K

tf.config.run_functions_eagerly(False)  # or True

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

DRIVE = False

PREFIX = "../../drive/MyDrive/aslData/asl-mnist/"

# custom plotting
from plot import *

dim = 90


# Reset Keras Session
def reset_keras():
    sess = K.get_session()
    K.clear_session()
    sess.close()
    sess = K.get_session()


def CNN_Model(
    loss_fn,
    optimizer_algorithm,
    monitor_metric,
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


def create_resnet50(input_shape, n_out, loss_fn, optimizer_algorithm, monitor_metric):

    base_model = ResNet50(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    for layer in base_model.layers[:143]:
        layer.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(n_out, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_mobilenet(input_shape, n_out, loss_fn, optimizer_algorithm, monitor_metric):
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


def create_mobilenet_pretrained(
    input_shape, n_out, loss_fn, optimizer_algorithm, monitor_metric
):

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


def create_vgg16(input_shape, n_out, loss_fn, optimizer_algorithm, monitor_metric):
    old_model = VGG16(
        input_shape=input_shape, include_top=False, weights="imagenet", pooling="avg"
    )

    old_model.trainable = False

    model = Sequential([old_model, Dense(n_out, activation="sigmoid")])

    model.compile(loss=loss_fn, optimizer=optimizer_algorithm, metrics=monitor_metric)
    return model


def create_densenet_pretrained(
    input_shape, n_out, loss_fn, optimizer_algorithm, monitor_metric
):
    OldModel = DenseNet121(
        include_top=False, input_shape=input_shape, weights="imagenet"
    )

    for layer in OldModel.layers[:149]:
        layer.trainable = False
    for layer in OldModel.layers[149:]:
        layer.trainable = True

    model = Sequential()
    model.add(OldModel)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.7))
    model.add(BatchNormalization())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(n_out, activation="softmax"))

    model.compile(
        optimizer=optimizer_algorithm,
        loss=loss_fn,
        metrics=monitor_metric,
    )

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


def create_model(mode, loss_fn, optimizer_algorithm, monitor_metric):
    model = None
    if mode == "CNN":
        model = CNN_Model(
            loss_fn,
            optimizer_algorithm,
            monitor_metric,
            (
                dim,
                dim,
                3,
            ),
            n_out=24,
        )
    elif mode == "RESNET_PRETRAINED":
        model = create_resnet50(
            (dim, dim, 3), 24, loss_fn, optimizer_algorithm, monitor_metric
        )
    elif mode == "MOBILENET_PRETRAINED":
        model = create_mobilenet_pretrained(
            (dim, dim, 3), 24, loss_fn, optimizer_algorithm, monitor_metric
        )
    elif mode == "DENSENET_PRETRAINED":
        model = create_densenet_pretrained(
            (dim, dim, 3), 24, loss_fn, optimizer_algorithm, monitor_metric
        )
    elif mode == "VGG_PRETRAINED":
        model = create_vgg16(
            (dim, dim, 3), 24, loss_fn, optimizer_algorithm, monitor_metric
        )
    else:
        raise Exception("Invalid model type")

    return model


# def init_weight(model, weights):
#     ## we can uncomment the line below to reshufle the weights themselves so they are not exactly the same between folds
#     # weights = [np.random.permutation(x.flat).reshape(x.shape) for x in weights]
#     model.set_weights(weights)


def execute_training(
    X,
    y,
    datagen,
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

        X_train = tf.image.resize(X_train, (dim, dim))
        X_val = tf.image.resize(X_val, (dim, dim))
        X_test = tf.image.resize(X_test, (dim, dim))

        X_train = tf.image.grayscale_to_rgb(X_train)
        X_val = tf.image.grayscale_to_rgb(X_val)
        X_test = tf.image.grayscale_to_rgb(X_test)

        # fit the training data for the datagen
        datagen.fit(X_train)

        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            verbose=1,
            validation_data=(X_val, y_val),
        )

        scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)

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

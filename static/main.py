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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
import json
from sklearn.model_selection import train_test_split

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

PREFIX = "../../drive/MyDrive/handleData/"

# custom plotting
from plot import * 

# collect the data
def read_data(): 
    imagepaths = []
    for root, _, files in os.walk("../data/leapGestRecog", topdown=False): 
        for name in files:
            path = os.path.join(root, name)
            if path.endswith("png"): # We want only the images
                imagepaths.append(path)

    # printing the number of images loaded: 
    print("Number of images loaded: ", len(imagepaths))
    return imagepaths 

def augment_data(imagepaths): 

    X = [] 
    y = [] 

    # Loops through imagepaths to load images and labels into arrays
    for path in imagepaths:
        img = cv2.imread(path) # Reads image and returns np.array
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts into the corret colorspace (GRAY)
        img = cv2.resize(img, (320, 120)) # Reduce image size so training can be faster
        X.append(img)
    
        # Processing label in image path
        category = path.split("/")[3]
        label = int(category.split("_")[0][1]) # We need to convert 10_down to 00_down, or else it crashes
        y.append(label)

    X = np.array(X, dtype="uint8")
    X = X.reshape(len(imagepaths), 120, 320, 1) 
    y = np.array(y)

    print("Images loaded: ", len(X))
    print("Labels loaded: ", len(y))

    return X, y 

def create_model(mode="CNN"):

    model = None 
    if mode == "CNN":
        model = Sequential()
        model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(120, 320, 1))) 
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu')) 
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))
    elif mode == "RESNET":
        model = keras.applications.resnet.ResNet50(include_top=False, weights=None, input_shape=(120, 320, 1))
    elif mode == "RESNET_PRETRAINED": 
        model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(120, 320, 1))
    elif mode == "MOBILENET": 
        model = keras.applications.mobilenet.MobileNet(include_top=False, weights=None, input_shape=(120, 320, 1))
    elif mode == "DENSENET":
        model = keras.applications.densenet.DenseNet121(include_top=False, weights=None, input_shape=(120, 320, 1))
    else: 
        # throw an error to the user 
        raise Exception("Invalid model type")

    return model 


def execute_training(X, y, experiment_name = 'exper1', num_folds=5, epochs=10, batch_size=32, verbose=False, optimizer='adam', loss='sparse_categorical_crossentropy', mode='CNN'):

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
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])  

        print("For Fold: " + str(fold_no))
        X_val, X_test, y_val, y_test = train_test_split(X[test], y[test], test_size=0.5, random_state=42)

        history = model.fit(X[train], y[train], batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_val, y_val))
        scores = model.evaluate(X[test], y[test], verbose=verbose)

        # save the predictions from the model.evaluate
        y_prob = model.predict(X_test)
        predictions = y_prob.argmax(axis=-1)

        # compute the precision, recall, f1 score, and accuracy
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
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
        train_loss.append(history.history['loss'])
        train_acc.append(history.history['accuracy'])

        # save the validation loss and accuracy
        val_loss.append(history.history['val_loss'])
        val_acc.append(history.history['val_accuracy'])

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

    return model_cache, train_loss, train_acc, val_loss, val_acc, predictions_cache, targets_cache, precision_history, recall_history, f1_history, accuracy_history, cfx_history
    
def execute_micro_macro_metrics(model_cache, predictions_cache, targets_cache, precision_history, recall_history, f1_history, accuracy_history, cfx_history, experiment_name):

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
    JSON_data['macro_metrics'] = {'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1, 'accuracy': macro_accuracy, 'cfx': cfx_history_avg_json}

    # micro averaging:

    micro_precision = precision_score(np.concatenate(targets_cache), np.concatenate(predictions_cache), average="weighted")
    micro_recall = recall_score(np.concatenate(targets_cache), np.concatenate(predictions_cache), average="weighted")
    micro_f1 = f1_score(np.concatenate(targets_cache), np.concatenate(predictions_cache), average="weighted")
    micro_accuracy = accuracy_score(np.concatenate(targets_cache), np.concatenate(predictions_cache))

    # confusion matrix for micro averaging:
    cfx_micro = confusion_matrix(np.concatenate(targets_cache), np.concatenate(predictions_cache))
    cfx_micro_json = json.dumps(cfx_micro.tolist())
    
    # store into JSON:
    JSON_data['micro_metrics'] = {'precision': micro_precision, 'recall': micro_recall, 'f1': micro_f1, 'accuracy': micro_accuracy, 'cfx': cfx_micro_json}

    # save the JSON_data to a file
    with open(PREFIX + "logs/" + experiment_name + "/" + experiment_name + "_data.json", 'w') as outfile:
        json.dump(JSON_data, outfile)

def extract_hyperparameters(filename): 
    hyperparams = None 
    with open("experiments/{}.yaml".format(filename), "r") as f: 
        hyperparameters = yaml.load(f, Loader=yaml.FullLoader)
    return hyperparameters

if __name__ == "__main__":

    # extract file name from command line input 
    filename = sys.argv[1]
    hyperparameters = extract_hyperparameters(filename)

    if not os.path.exists(PREFIX + 'models/' + hyperparameters["EXPERIMENT_NAME"]):
        os.makedirs(PREFIX + 'models/' + hyperparameters["EXPERIMENT_NAME"])

    if not os.path.exists(PREFIX + 'logs/' + hyperparameters["EXPERIMENT_NAME"]):
        os.makedirs(PREFIX + 'logs/' + hyperparameters["EXPERIMENT_NAME"])

    # read the data
    imagepaths = read_data()

    # finalize data 
    X, y = augment_data(imagepaths)

    # execute the training pipeline
    model_cache, train_loss, train_acc, val_loss, val_acc, predictions_cache, targets_cache, precision_history, recall_history, f1_history, accuracy_history, cfx_history = \
        execute_training(X, y, hyperparameters["EXPERIMENT_NAME"], 
        hyperparameters["CONFIG"]["NUM_FOLDS"], hyperparameters["CONFIG"]["EPOCHS"], 
        hyperparameters["CONFIG"]["BATCH_SIZE"], hyperparameters["CONFIG"]["VERBOSE"], 
        hyperparameters["CONFIG"]["OPTIMIZER"], hyperparameters["CONFIG"]["LOSS"], hyperparameters["CONFIG"]["MODE"])
    
    plot_training_validation(train_loss, train_acc, val_loss, val_acc, hyperparameters["EXPERIMENT_NAME"], PREFIX)
    execute_micro_macro_metrics(model_cache, predictions_cache, targets_cache, precision_history, recall_history, f1_history, accuracy_history, cfx_history, hyperparameters["EXPERIMENT_NAME"])
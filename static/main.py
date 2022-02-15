import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import yaml
import time 
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix # Helps present results as a confusion-matrix


# Import of keras model and hidden layers for our convolutional network
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten

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
    else: 
        # throw an error to the user 
        raise Exception("Invalid model type")

    return model 


def execute_training(X, y, num_folds=5, epochs=10, batch_size=32, verbose=False):

    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_no = 1

    # log to a string buffer: 
    log_string = ""

    accuracy_per_fold = [] 
    loss_per_fold = []
    model_cache = [] 
    model_history = [] 
    model_scores = [] 

    # train across folds
    for train, test in kfold.split(X, y):
        
        best_accuracy = None 

        print("Fold: ", fold_no)

        fold_no += 1

        # create the model 
        model = create_model()
        
        # compile the model 
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  

        # add output to the string buffer: 
        log_string += "Fold: " + str(fold_no) + "\n"
        
        history = model.fit(X[train], y[train], batch_size=batch_size, epochs=epochs, verbose=verbose)
        scores = model.evaluate(X[test], y[test], verbose=verbose)

        # save the model in model_cache
        model_cache.append(model)
        model_history.append(history)
        model_scores.append(scores)

        # add output to the string buffer:
        log_string += "Test loss: " + str(scores[0]) + "\n"
        log_string += "Test accuracy: " + str(scores[1]) + "\n"

        # add accuracy, loss to array 
        accuracy_per_fold.append(scores[1])
        loss_per_fold.append(scores[0])

        # if this the best model, save then model
        if best_accuracy != None and scores[1] > best_accuracy:
            best_model = model
            best_model_history = history
            best_model_scores = scores

        # save a plot of the model accuracy and loss
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['loss'])
        plt.title('Model accuracy and loss')
        plt.ylabel('Metrics')
        plt.xlabel('Epoch')
        plt.legend(['Accuracy', 'Loss'], loc='upper left')
        plt.savefig("logs/model_accuracy_loss_" + str(fold_no) + ".png")

    # save best_model into a file: 
    if best_model != None:
        best_model.save("models/best_model.h5")
    


def execute_testing(model, X, y):
    # test the model that is created
    pass 

if __name__ == "__main__":

    # extract hyperparams from experiments/exper1.yaml
    with open("experiments/exper1.yaml") as f:
        hyperparams = yaml.load(f, Loader=yaml.FullLoader)

    print(hyperparams)


    # imagepaths = read_data()
    # X, y = augment_data(imagepaths)
    # execute_training(X, y)
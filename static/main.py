import os
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

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
    else: 
        # throw an error to the user 
        raise Exception("Invalid model type")

    return model 


def execute_training(X, y, experiment_name = 'exper1', num_folds=5, epochs=10, batch_size=32, verbose=False, optimizer='adam', loss='sparse_categorical_crossentropy'):

    kfold = KFold(n_splits=num_folds, shuffle=True)

    fold_no = 1

    model_cache = []
    predictions_cache = [] 

    train_loss = [] 
    train_acc = []

    val_loss = [] 
    val_acc = [] 

    precision_history = [] 
    recall_history = []
    f1_history = []
    accuracy_history = []

    # train across folds
    for train, test in kfold.split(X, y):

        model = create_model()
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])  

        # add output to the string buffer: 
        print("For Fold: " + str(fold_no))

        # split the testing data into testing and validation 50-50
        test_len = len(test)
        test_len_half = int(test_len/2)
        test_half_one = test[:test_len_half]
        test_half_two = test[test_len_half:]

        X_val, X_test = X[test_half_one], X[test_half_two]
        y_val, y_test = y[test_half_one], y[test_half_two]

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

        print("\n")

        model_cache.append(model)

        fold_no += 1

    return model_cache, train_loss, train_acc, val_loss, val_acc, predictions_cache, precision_history, recall_history, f1_history, accuracy_history
    
def execute_testing(model_cache, X, y, experiment_name='exper1'):

    # save all the models in the model_cache
    epoch_counter = 1 
    for model in model_cache:
        model.save("models/{}/{}.h5".format(experiment_name, epoch_counter))
        epoch_counter = epoch_counter + 1

    # cache data into a JSON
    JSON_data = {}  

    loss_history = []
    acc_history = []
    confusion_history = [] 
    recall_history = [] 
    precision_history = []
    f1_history = [] 

    # test all models in the model_cache array on the entire dataset
    counter = 1
    for model in model_cache:
        scores = model.evaluate(X, y, verbose=0)
        print("Fold: " + str(counter))
        print("Test loss: " + str(scores[0]))
        print("Test accuracy: " + str(scores[1]))

        loss_history.append(scores[0])
        acc_history.append(scores[1])

        predict_x=model.predict(X) 
        classes_x=np.argmax(predict_x,axis=1)

        # create confusion matrix and store in confusion_history
        cur_cfx = confusion_matrix(y, classes_x)
        confusion_history.append(cur_cfx)

        # compute precision score, recall score, and f1 score
        recall = recall_score(y, classes_x)
        precision = precision_score(y, classes_x)
        f1 = f1_score(y, classes_x)

        print("Recall: ", recall)
        print("Precision: ", precision)
        print("F1: ", f1)

        recall_history.append(recall)
        precision_history.append(precision)
        f1_history.append(f1)

        temp_obj = {"Test loss": scores[0], "Test accuracy": scores[1], "Confusion matrix": cur_cfx, "Recall": recall, "Precision": precision, "F1": f1}
        JSON_data["Fold {}".format(counter)] = temp_obj

        # print()

        counter += 1 

    # average the test loss and test accuracy across all folds and save into JSON
    JSON_data["Average"] = {"Test loss": np.mean(loss_history), "Test accuracy": np.mean(acc_history), "Confusion Matrix": np.mean(confusion_history, axis=0), "Recall": np.mean(recall_history), "Precision": np.mean(precision_history), "F1": np.mean(f1_history)}

    # take the average of the confusion matrices
    # confusion_matrix = np.mean(confusion_history, axis=0)

    # # plot the confusion matrix
    # plt.figure()
    # plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.colorbar()
    # tick_marks = np.arange(10)
    # plt.xticks(tick_marks, range(10))
    # plt.yticks(tick_marks, range(10))
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.tight_layout()
    # plt.savefig("confusion_matrix.png")
    # plt.close()

    # save the JSON_data to a file
    with open("logs/" + experiment_name + "/" + experiment_name + "_data.json", 'w') as outfile:
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

    if not os.path.exists('models/' + hyperparameters["EXPERIMENT_NAME"]):
        os.makedirs('models/' + hyperparameters["EXPERIMENT_NAME"])

    if not os.path.exists('logs/' + hyperparameters["EXPERIMENT_NAME"]):
        os.makedirs('logs/' + hyperparameters["EXPERIMENT_NAME"])

    # read the data
    imagepaths = read_data()

    # finalize data 
    X, y = augment_data(imagepaths)

    # execute the training pipeline
    model_cache, train_loss, train_acc, val_loss, val_acc,_,_,_,_,_ = execute_training(X, y, hyperparameters["EXPERIMENT_NAME"], 
        hyperparameters["CONFIG"]["NUM_FOLDS"], hyperparameters["CONFIG"]["EPOCHS"], 
        hyperparameters["CONFIG"]["BATCH_SIZE"], hyperparameters["CONFIG"]["VERBOSE"], 
        hyperparameters["CONFIG"]["OPTIMIZER"], hyperparameters["CONFIG"]["LOSS"])

    plot_training_validation(train_loss, train_acc, val_loss, val_acc, hyperparameters["EXPERIMENT_NAME"])

    # execute_testing(model_cache, X, y, hyperparameters["EXPERIMENT_NAME"])
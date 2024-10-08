import json
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

DATASET_PATH = "Data.json"

def load_data(dataset_path):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)

        # convert lists into numpy arrays
        x = np.array(data["mfcc"])
        y = np.array(data["labels"])

        return x, y

def prepare_datasets(test_size, validation_size):
    # load data
    X, y = load_data(DATASET_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # 3D array --> (130, 13, 1)
    X_train = X_train[..., np.newaxis] # 4D array -> (num_samples, 130, 13, 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_test, X_validation, y_train, y_test, y_validation

def build_model(input_shape):

    # create cnn
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output
    model.add(keras.layers.Flatten())

    # feed into dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(13, activation='softmax'))

    return model

def predict(model, X, y):

    X = X[np.newaxis, ...]

    # prediction is 2d array [ [0.1, 0.2,...] ] values represent scores for 10 genres
    prediction = model.predict(X) # X -> (1, 130, 13, 1) first index is number of predictions

    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1)
    print("The expected index is {}, and the predicted index is {}".format(y, predicted_index))

if __name__ == "__main__":

    # create the train, validation, and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # build the CNN network
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train network
    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)
    model.save("cnnModel.keras")

    # evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set: {}".format(test_accuracy))


    # make prediction on a sample
    X = X_test[100]
    y = y_test[100]
    predict(model, X, y)


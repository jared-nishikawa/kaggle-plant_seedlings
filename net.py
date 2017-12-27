#!/usr/bin/python

import tools
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

if __name__ == '__main__':
    np.random.seed(0)

    X_train = []
    Y_train = []
    df = pd.read_csv('train.csv')
    for row in df.iterrows():
        pixels = list(row[1])[:1024]
        X_train.append(pixels)

        label = row[1][-1]
        Y_train.append(tools.categorical(label))

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # Preprocess input data

    # Reshape input data
    X_train = X_train.reshape(X_train.shape[0], 1, 32, 32)
    #X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

    # Change type
    X_train = X_train.astype('float32')
    #X_test = X_test.astype('float32')

    # Define model architecture

    model = Sequential()

    # Declare input layer
    model.add(Conv2D(32, (3, 3), activation='relu', \
            input_shape=(1, 32, 32), data_format='channels_first'))
    model.summary()
    print model.output_shape

    # More layers
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # Fully connected dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(tools.labels), activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

    # Fit model on training data
    model.fit(X_train, Y_train,
            batch_size=32, epochs=100, verbose=1)

    # Evaluate
    scores = model.evaluate(X_train, Y_train)
    print "\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)

    # Save model
    model.save('model.h5')


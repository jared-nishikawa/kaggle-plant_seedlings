#!/usr/bin/python

import tools
import pandas as pd
import numpy as np
from keras.models import load_model

if __name__ == '__main__':
    model = load_model('model.h5')

    X_test = []
    filenames = []
    df = pd.read_csv('test.csv')
    for row in df.iterrows():
        pixels = list(row[1])[:1024]
        X_test.append(pixels)
        filenames.append(row[1][-1])

    X_test = np.array(X_test)

    X_test = X_test.reshape(X_test.shape[0], 1, 32, 32)
    X_test = X_test.astype('float32')

    # Predictions
    Z = model.predict(X_test, batch_size=32, verbose=1)

    #with open('submission.csv','w') as f:
    #    f.write('file,species\n')
    #    for ind,z in enumerate(Z):
    #        name = tools.categorize(z).replace("_", ' ')
    #        f.write(filenames[ind] + ',' + name + '\n')
    


# Plant Seedlings Classification

I thought [this Kaggle challenge](https://www.kaggle.com/c/plant-seedlings-classification/) would be a good way for me to test myself using what I knew about machine learning and Python's libraries.

I had previously written some simple PNG decoder functions in Python, so I grabbed that (png.py) and used it to:
- Read in the PNG images
- Convert to grayscale
- Highlight green areas to black, and convert the rest of the image to black
- Resize to 32x32

Then I can convert the resized pictures to a feature vector of 1024 (32*32) pixels of either black (0) or white (1).

For the neural network, I used Python's keras.  My model consisted of:
- A convolutional layer
- A pooling layer
- A flattening layer
- A dense layer (to output to a vector of length 12, the number of labels)

Fitting the model took about three hours, so I decided I would just run it once  and submit my results.  I could probably go back and tweak my data prep, or my neural net model, and increase my accuracy, but this was mainly for my own edification, and I'm satisfied.

My final model resulted in 67% accuracy on the Kaggle challenge.

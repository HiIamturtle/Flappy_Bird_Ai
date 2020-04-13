# Import Libraries
import random

import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from networks import Network
from layers import *
from activations import *


# Initializing the neural network
nn = Network(structure=[
    Dense(784, None),      # Input
    Dense(64, sigmoid),    # h1
    Dense(64, sigmoid),   # h2
    Dense(10, sigmoid),    # output layer
])

nn.init_network()


# Creating var for the saved weighs folder
save_dir = 'saved_mnist/'

# Loading the datasets
(train_img, train_labels), (test_img, test_labels) = tf.keras.datasets.mnist.load_data()

# Making arrays to manage the data
train_data = []
test_data = []


# Getting the name for new save file
def get_filename(directory):
    num = 0
    if os.path.isfile(directory + 'saved_weights.npz'):
        while True:
            num += 1
            file = directory + f'saved_weights({str(num)}).npz'
            if os.path.isfile(file):
                continue
            else:
                return file
    else:
        return directory + 'saved_weights.npz'


# Showing a random image and the networks guess
def show(images):
    rows = 4
    cols = 5

    # plot images
    fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 2 * rows))
    for index, i in enumerate(images):
        ax = axes[index // cols, index % cols]
        ax.imshow(i['Data'], cmap='gray')
        label = nn.feed_forward(i['Data'])
        ax.set_title(f'Guess: {np.argmax(label)}')
    plt.tight_layout()
    plt.show()


# Preparing the data
def prep_data():
    for i in range(len(train_img)):
        # Formatting the train image
        img = train_img[i] / 255
        label_vec = np.zeros(10,)
        label_vec[train_labels[i]] = 1
        train_data.append({'Data': img, 'Label': label_vec})

    for i in range(len(test_img)):
        # Formatting the train image
        img = test_img[i] / 255
        test_data.append({'Data': img, 'Label': test_labels[i]})


# Testing the model and returning an accuracy
def test():
    # Keeping count of how many are correct
    n_correct = 0

    # Looping over the test set
    for i in test_data:
        guess = np.argmax(nn.feed_forward(i['Data']))
        answer = i['Label']

        # Checking to see if guess was correct
        if guess == answer:
            n_correct += 1

    # Calculating and returning the percentage correct
    percent = (n_correct / len(test_data)) * 100

    return percent


# Updating the model based on example; "Training" the model
def train(n_epochs=10_000, batch_size=10, showing=False):
    # Randomizing the order of the train data
    random.shuffle(train_data)

    # Looping over for every epoch
    for e in range(n_epochs):
        # Creating an array to temporarily store the batch
        batch = []

        # Looping over each item in train_data
        for i in train_data:
            batch.append(i)

            # Checking if batch = batch size or its at the end of the data
            if len(batch) == batch_size or len(batch) == len(train_data):
                nn.train(batch=batch)
                batch = []

        percentage = test()
        print(f'Epoch: {e + 1}, Test Accuracy: {percentage}')

        if showing and e % 50 == 0:
            # show example with networks guess
            img_range = random.randint(0, len(test_data) - 20)
            show(test_data[img_range:img_range + 20])

    # Saving the network
    filename = get_filename(save_dir)
    nn.save(filename)


# Running the code
if __name__ == '__main__':
    # First prepare the data
    prep_data()

    # Training on the data
    train(n_epochs=20, batch_size=1)
    #print(test())

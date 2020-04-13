# Libraries
import numpy as np


# Gradient Decent
def gradient_decent(batch=None, network=None):

    # Setting delta weights to a matrix of zeros
    network.delta_weights()

    # Looping over each item in the batch
    for data in batch:
        # Transposing the information
        if isinstance(data['Data'], list):
            input_vec = np.array(data['Data'], ndmin=2).T
        elif isinstance(data['Data'], np.ndarray):
            input_vec = data['Data'].flatten()
            input_vec = np.array(input_vec, ndmin=2).T
        else:
            raise TypeError('Input needs to be either an np.array or a list')

        label_vec = np.array(data['Label'], ndmin=2).T

        # Running the feed forward algorithm with batch input
        net_out = np.array(network.feed_forward(input_vec, training=True), ndmin=2)

        # Reversing the structure for back propagation
        network.struct.reverse()

        # Calculating the errors
        last_error = net_out
        for index, i in enumerate(network.struct[:-1]):
            # Calculating the error
            last_error = i.calc_error(input=label_vec, last_error=last_error, index=index)

        # Calculating the deltas using gradient decent
        for index, i in enumerate(network.struct[1:]):
            index += 1

            # Calculating the gradient
            next_layer = network.struct[index - 1]
            gradient = next_layer.error * i.activated * (1 - i.activated)

            # If not the input layer
            if index is not len(network.struct) - 1:
                last_layer = network.struct[index + 1]
                i.d_w += network.lr * np.dot(gradient, last_layer.activated.T)
                i.d_b += network.lr * gradient
            # If the input layer
            else:
                i.d_w += network.lr * np.dot(gradient, input_vec.T)
                i.d_b += network.lr * gradient

        # Returning hte structure back to normal
        network.struct.reverse()

    # Updating the weights of each layer
    for i in network.struct[:-1]:
        i.w += i.d_w


# Genetic Algorithm
def gen_algorithm():
    pass

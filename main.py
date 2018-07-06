from nn import Neural_Network
from tensorflow.examples.tutorials.mnist import input_data

# Global vars
data_set = input_data.read_data_sets("/tmp/data/", one_hot=True)

# NN creation
neural_network = Neural_Network(500, 500, 500, 10, 100, 784)

# NN training
neural_network.train_me(data_set)

# For cnn:

# from cnn import Convolutional_Neural_Network
# convolutional_neural_network = Convolutional_Neural_Network(10, 128, 784)

# convolutional_neural_network.train_me(data_set)

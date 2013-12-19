#Alassane Ndiaye
#David Krueger
#Thomas Rohee

import gzip
import pickle
import numpy as np


class Mnist:
    def __init__(self):
        dataset = gzip.open('data/mnist.pkl.gz')
        data = pickle.load(dataset)

        self.train_inputs = data[0][0]
        self.train_labels = data[0][1]
        self.valid_inputs = data[1][0]
        self.valid_labels = data[1][1]
        self.test_inputs = data[2][0]
        self.test_labels = data[2][1]

        self.train_set = np.hstack((self.train_inputs, self.train_labels.reshape(self.train_labels.shape[0], 1)))
        self.test_set = np.hstack((self.test_inputs, self.test_labels.reshape(self.test_labels.shape[0], 1)))
        self.valid_set = np.hstack((self.valid_inputs, self.valid_labels.reshape(self.valid_labels.shape[0], 1)))

        self.train_cols = [0, 1]

        # Nombre de classes
        self.n_classes = 10
        # Nombre de points d'entrainement
        self.n_train = self.train_inputs.shape[0]




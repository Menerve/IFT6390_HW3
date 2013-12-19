#Alassane Ndiaye
#David Krueger
#Thomas Rohee

import numpy as np


class TwoMoons:

    def __init__(self):
        self.data = np.loadtxt('data/2moons.txt')

        self.train_cols = [0, 1]
        self.target_ind = [self.data.shape[1] - 1]

        # Nombre de classes
        self.n_classes = 2
        # Nombre de points d'entrainement
        self.n_train = 660

        # decommenter pour avoir des resultats non-deterministes
        np.random.seed(3395)

        inds = range(self.data.shape[0])
        np.random.shuffle(inds)
        self.train_inds = inds[:self.n_train]
        self.test_inds = inds[self.n_train:]

        self.train_set = self.data[self.train_inds, :]  # garder les bonnes lignes
        self.test_set = self.data[self.test_inds, :]
        self.test_inputs = self.test_set[:, :-1]
        self.test_labels = self.test_set[:, -1]


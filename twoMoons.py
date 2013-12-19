# Alassane Ndiaye
# David Krueger
# Thomas Rohee

from data.TwoMoons import TwoMoons
from displays import *
import mlp
import mlploop


def run():
    twoMoons = TwoMoons()

    print "On va entrainer sur ", twoMoons.n_train, " exemples d'entrainement"

    mu = 0.01
    n_hidden = [2, 10, 20, 50]
    weight_decay = [0, 0.001, 0.003, 0.1]
    n_updates = 100
    batch_size = 10

    for w in weight_decay:
        for n_dh in n_hidden:
            assert twoMoons.train_set.shape[0] % batch_size == 0

            print "Weight:", w, "Hidden number:", n_dh
            # Loops implementation
            # modelMlp = mlploop.MultilayerPerceptronLoop(twoMoons.train_set.shape[1] - 1, n_dh, twoMoons.n_classes, w, mu)
            # modelMlp.train(twoMoons.train_set, n_updates, batch_size)

            # Matrix implementation
            modelMlp = mlp.MultilayerPerceptron(twoMoons.train_set.shape[1] - 1, n_dh, twoMoons.n_classes, w, mu)
            modelMlp.train(twoMoons.train_set, n_updates, batch_size, twoMoons.test_set, twoMoons.test_set)

            error_rate(modelMlp, twoMoons)
            draw_decisions(modelMlp, twoMoons)
# Alassane Ndiaye
# David Krueger
# Thomas Rohee

from data.Mnist import Mnist
from displays import *
import mlp
import mlploop


def run():
    mnistData = Mnist()

    print "On va entrainer sur ", mnistData.n_train, " exemples d'entrainement"

    mu = 0.003
    n_hidden = [300]
    weight_decay = [0.003]
    n_updates = 100
    batch_size = 100

    for w in weight_decay:
        for n_dh in n_hidden:
            assert mnistData.train_set.shape[0] % batch_size == 0

            print "Weight:", w, "Hidden number:", n_dh
            # Loops implementation
            # modelMlp = mlploop.MultilayerPerceptronLoop(mnistData.train_set.shape[1] - 1, n_dh, mnistData.n_classes, w,mu)
            # modelMlp.train(mnistData.train_set, n_updates, batch_size)

            # Matrix implementation
            modelMlp = mlp.MultilayerPerceptron(mnistData.train_set.shape[1] - 1, n_dh, mnistData.n_classes, w, mu)
            modelMlp.train(mnistData.train_set, n_updates, batch_size, mnistData.test_set, mnistData.valid_set)

            error_rate(modelMlp, mnistData)
            draw_class_error_curves(n_updates)
            draw_loss_error_curves(n_updates)
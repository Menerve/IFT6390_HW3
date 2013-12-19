# coding=utf-8

# Alassane Ndiaye
# David Krueger
# Thomas Rohee

import time
import numpy as np


class MultilayerPerceptronLoop:
    def __init__(self, ninp, nhid, nout, l2, lr):
        # initialisation des attributs
        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout
        self.l2 = l2
        self.lr = lr

        # initialisation des paramètres
        # Couche 1
        w1bound = 1 / np.sqrt(self.ninp)
        self.W1 = np.asarray(np.random.uniform(low=-w1bound, high=w1bound, size=(self.nhid, self.ninp)))
        self.b1 = np.zeros((self.nhid, 1))

        # Couche 2
        w2bound = 1 / np.sqrt(self.nhid)
        self.W2 = np.asarray(np.random.uniform(low=-w2bound, high=w2bound, size=(self.nout, self.nhid)))
        self.b2 = np.zeros((self.nout, 1))

        self.params = [self.W1, self.b1, self.W2, self.b2]

    def onehot(self, target):
        onehot = np.zeros((self.nout, 1))
        onehot[target] = 1
        return onehot

    def fprop(self, input):
        input = np.asarray(input).reshape((1, input.size))
        self.ha = np.dot(self.W1, input.transpose()) + self.b1
        self.hs = np.tanh(self.ha)
        oa = np.dot(self.W2, self.hs) + self.b2
        self.os = np.exp(oa) / np.sum(np.exp(oa), axis=0)

        return self.os

    def bprop(self, input, target):
        # Gradients
        input = np.asarray(input).reshape((1, input.size))
        grad_oa = self.os - self.onehot(target)
        grad_b2 = grad_oa
        grad_W2 = np.dot(grad_oa, self.hs.transpose())
        grad_hs = np.dot(self.W2.transpose(), grad_oa)
        grad_ha = grad_hs * (1 - (np.tanh(self.ha) ** 2))
        grad_b1 = grad_ha
        grad_W1 = np.dot(grad_ha, input)

        return [grad_W1, grad_b1, grad_W2, grad_b2]

    def finite_diff(self, inputs, epsilon=10 ** -5):
        ratios = []
        if inputs.size == 3:
            inputs = np.asarray(inputs).reshape((-1, inputs.shape[0]))
        else:
            inputs = np.asarray(inputs).reshape((-1, inputs.shape[1]))

        for k in range(inputs.shape[0]):
            current_loss = - np.log(self.fprop(inputs[k, :-1])[inputs[k, -1]])
            grad_params = self.bprop(inputs[k, :-1], inputs[k, -1])
            for array, gradients in (
                (self.W1, grad_params[0]), (self.W2, grad_params[2]), (self.b1, grad_params[1]),
                (self.b2, grad_params[3])):
                for i in range(array.shape[0]):
                    for j in range(array.shape[1]):
                        array[i, j] += epsilon
                        loss_after_update = - np.log(self.fprop(inputs[k, :-1])[inputs[k, -1]])
                        array[i, j] -= epsilon
                        estimated_gradient = (loss_after_update - current_loss) / epsilon
                        gradients = np.asarray(gradients).reshape((-1, gradients.shape[0]))
                        gradient = gradients[i, j]
                        ratios.append(estimated_gradient / gradient)
        return ratios

    def train(self, train_data, nstages, batch_size=1):

        t1 = time.clock()
        for j in range((nstages * train_data.shape[0]) / batch_size):
            if j * batch_size % train_data.shape[0] == 0 and j != 0:
                t2 = time.clock()
                print "Took", t2 - t1, "to calculate epoch."
                print "Epoch:", j / (train_data.shape[0] / batch_size)
                t1 = time.clock()

            cost = [0.0, 0.0, 0.0, 0.0]
            start = j * batch_size % train_data.shape[0]
            batch_data = train_data[start: start + batch_size, :]

            for l in range(batch_data.shape[0]):
                # Propagation avant
                input = batch_data[l, :-1].reshape((1, train_data.shape[1] - 1))
                self.fprop(input)

                # Propagation arrière
                grad_params = self.bprop(input, batch_data[l, -1])

                for k in range(len(cost)):
                    cost[k] = cost[k] + grad_params[k]

            # Cout
            if self.l2 > 0:
                cost[0] += 2 * self.l2 * self.W1
                cost[2] += 2 * self.l2 * self.W2

            self.W1 -= self.lr * cost[0]
            self.b1 -= self.lr * cost[1]
            self.W2 -= self.lr * cost[2]
            self.b2 -= self.lr * cost[3]

            if j == 0 and batch_size == 1 and self.nhid == 2:
                print "grad_W1: ", cost[0]
                print "grad_b1: ", cost[1]
                print "grad_W2: ", cost[2]
                print "grad_b2: ", cost[3]
                ratios = self.finite_diff(train_data[10, :], epsilon=10 ** -5)
                print "Différence finie pour un exemple:", sum(ratios) / len(ratios)
            elif j == 0 and batch_size == 10 and self.nhid == 2:
                print "grad_W1: ", cost[0]
                print "grad_b1: ", cost[1]
                print "grad_W2: ", cost[2]
                print "grad_b2: ", cost[3]
                ratios = self.finite_diff(batch_data, epsilon=10 ** -5)
                print "Différence finie pour lot de 10:", sum(ratios) / len(ratios)

    def compute_predictions(self, test_data):
        # Predictions
        sorties = np.zeros((test_data.shape[0], test_data.shape[1]))
        for i in range(test_data.shape[0]):
            sorties[i, :] = self.fprop(test_data[i]).transpose()
        return sorties



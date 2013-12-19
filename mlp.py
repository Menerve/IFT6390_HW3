# coding=utf-8

# Alassane Ndiaye
# David Krueger
# Thomas Rohee

import time
import numpy as np


class MultilayerPerceptron:
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

    def onehot(self, targets):
        onehot = np.zeros((self.nout, targets.shape[0]))
        targets = np.array([int(i) for i in targets])
        # Indices of targets
        indices = (targets, range(targets.shape[0]))
        onehot[indices] = 1
        return onehot

    def fprop(self, inputs):
        self.ha = np.dot(self.W1, inputs.transpose()) + self.b1
        self.hs = np.tanh(self.ha)
        oa = np.dot(self.W2, self.hs) + self.b2
        self.os = np.exp(oa) / np.sum(np.exp(oa), axis=0)

        return self.os

    def bprop(self, inputs, targets):
        # Gradients
        grad_oa = self.os - self.onehot(targets)
        grad_b2 = grad_oa
        grad_W2 = np.dot(grad_oa, self.hs.transpose())
        grad_hs = np.dot(self.W2.transpose(), grad_oa)
        grad_ha = grad_hs * (1 - (np.tanh(self.ha) ** 2))
        grad_b1 = grad_ha
        grad_W1 = np.dot(grad_ha, inputs)

        return [grad_W1, grad_b1, grad_W2, grad_b2]

    def finite_diff(self, inputs, epsilon=10 ** -5):
        ratios = []
        for k in range(inputs.shape[0]):
            input = inputs[k, :-1].reshape((1, inputs.shape[1] - 1))
            current_loss = - np.log(self.fprop(input)[inputs[k, -1]])
            grad_params = self.bprop(input, inputs[k, -1])
            for array, gradients in (
                (self.W1, grad_params[0]), (self.W2, grad_params[2]), (self.b1, grad_params[1]),
                (self.b2, grad_params[3])):
                for i in range(array.shape[0]):
                    for j in range(array.shape[1]):
                        array[i, j] += epsilon
                        loss_after_update = - np.log(self.fprop(inputs[k, :-1])[inputs[k, -1]])
                        array[i, j] -= epsilon
                        estimated_gradient = (loss_after_update - current_loss) / epsilon
                        gradient = gradients[i, j]
                        ratios.append(estimated_gradient / gradient)
        return ratios

    def compute_classification_error(self, train_set, valid_set, test_set):
        result = []
        for dataset in (train_set, valid_set, test_set):
            predictions = np.argmax(self.compute_predictions(dataset[:, :-1]), axis=1)
            correct_predictions = np.sum(predictions != dataset[:, -1])
            average_error = float(correct_predictions) / float(dataset.shape[0])
            result.append(average_error)
        return result

    def compute_loss(self, train_set, valid_set, test_set):
        result = []
        for dataset in (train_set, valid_set, test_set):
            loss = 0
            for i in range(dataset.shape[0]):
                input = dataset[i, :-1].reshape((1, dataset.shape[1] - 1))
                loss += - np.log(self.fprop(input)[dataset[i, -1]])
            average_loss = loss / dataset.shape[0]
            result.append(average_loss[0])
        return result

    def stats(self, train_set, valid_set, test_set):
        classification_error = self.compute_classification_error(train_set, valid_set, test_set)
        loss = self.compute_loss(train_set, valid_set, test_set)
        sets = ["Train", "Valid", "Test"]
        for i, (error_rate, loss_rate) in enumerate(zip(classification_error, loss)):
            print sets[i], "Error rate:", error_rate, "Loss rate:", loss_rate

        errors_f_handle = file('errors_loss/errors.csv', 'a')
        loss_f_handle = file('errors_loss/loss.csv', 'a')
        class_array = np.array(classification_error)
        loss_array = np.array(loss)
        np.savetxt(errors_f_handle, class_array.reshape(1, class_array.shape[0]), fmt='%.5e', delimiter=',')
        np.savetxt(loss_f_handle, loss_array.reshape(1, loss_array.shape[0]), fmt='%.5e', delimiter=',')
        errors_f_handle.close()
        loss_f_handle.close()

    def train(self, train_data, nstages, batch_size, test_data, valid_data):

        t1 = time.clock()
        for j in range((nstages * train_data.shape[0]) / batch_size):


            cost = [0.0, 0.0, 0.0, 0.0]
            start = j * batch_size % train_data.shape[0]
            batch_data = train_data[start: start + batch_size, :]

            # Propagation avant
            self.fprop(batch_data[:, :-1])

            # Propagation arrière
            grad_params = self.bprop(batch_data[:, :-1], batch_data[:, -1])

            for k in range(len(cost)):
                cost[k] = cost[k] + grad_params[k]

            # Cout
            if self.l2 > 0:
                cost[0] += 2 * self.l2 * self.W1
                cost[2] += 2 * self.l2 * self.W2

            self.W1 -= self.lr * cost[0]
            self.b1 -= self.lr * np.sum(cost[1], axis=1).reshape((self.b1.shape[0], 1))
            self.W2 -= self.lr * cost[2]
            self.b2 -= self.lr * np.sum(cost[3], axis=1).reshape((self.nout, 1))

            if j == 0 and batch_size == 1 and self.nhid == 2:
                print "grad_W1: ", cost[0]
                print "grad_b1: ", cost[1]
                print "grad_W2: ", cost[2]
                print "grad_b2: ", cost[3]
            elif j == 0 and batch_size == 10 and self.nhid == 2:
                print "grad_W1: ", cost[0]
                print "grad_b1: ", np.sum(cost[1], axis=1).reshape((self.b1.shape[0], 1))
                print "grad_W2: ", cost[2]
                print "grad_b2: ", np.sum(cost[3], axis=1).reshape((self.nout, 1))

            if (j + 1) * batch_size % train_data.shape[0] == 0 and j != 0:
                t2 = time.clock()
                n_epoch = j / (train_data.shape[0] / batch_size) + 1
                print "Took", t2 - t1, "to calculate epoch."
                if n_epoch == 1:
                    with open('errors_loss/errors.csv', 'wb') as errors_f, open('errors_loss/loss.csv', 'wb') as loss_f:
                        errors_f.write(b'Train,Valid,Test\n')
                        loss_f.write(b'Train,Valid,Test\n')
                print "Epoch:", n_epoch
                self.stats(train_data, valid_data, test_data)
                t1 = time.clock()

    def compute_predictions(self, test_data):
        # Predictions
        sorties = self.fprop(test_data)
        return sorties.transpose()



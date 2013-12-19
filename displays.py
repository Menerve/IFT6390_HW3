# coding=utf-8

# Alassane Ndiaye
# David Krueger
# Thomas Rohee

import time

import pylab
import numpy as np

import utilitaires


def draw_decisions(model, dataset, grid_size=75):
    if len(dataset.train_cols) == 2:
        # Surface de decision
        utilitaires.gridplot(model, dataset.train_set, dataset.test_set, n_points=grid_size)
    else:
        print 'Trop de dimensions (', len(dataset.train_cols), ') pour pouvoir afficher la surface de decision'


def error_rate(model, dataset):
    # Obtenir ses prédictions
    t1 = time.clock()
    les_sorties = model.compute_predictions(dataset.test_inputs)
    t2 = time.clock()
    print 'Ca nous a pris ', t2 - t1, ' secondes pour calculer les predictions sur ', dataset.test_inputs.shape[
        0], ' points de test'

    classes_pred = np.argmax(les_sorties, axis=1)

    # Faire les tests
    err = 1.0 - np.mean(dataset.test_labels == classes_pred)
    print "L'erreur de test est de ", 100.0 * err, "%"


def draw_class_error_curves(epochs):
    class_errors = np.loadtxt("errors_loss/errors.csv", delimiter=",", skiprows=1)
    bins = np.arange(epochs)
    train_class_plot, = pylab.plot(bins, class_errors[:, 0])
    valid_class_plot, = pylab.plot(bins, class_errors[:, 1])
    test_class_plot, = pylab.plot(bins, class_errors[:, 2])

    pylab.xlabel('Epochs')
    pylab.ylabel('Classification error rate')
    pylab.title('Classification error for train, valid and test sets')
    pylab.legend([train_class_plot, valid_class_plot, test_class_plot], ['Train set', 'Valid set', 'Test set'])
    pylab.show()


def draw_loss_error_curves(epochs):
    loss_errors = np.loadtxt("errors_loss/loss.csv", delimiter=",", skiprows=1)
    bins = np.arange(epochs)
    train_loss_plot, = pylab.plot(bins, loss_errors[:, 0])
    valid_loss_plot, = pylab.plot(bins, loss_errors[:, 1])
    test_loss_plot, = pylab.plot(bins, loss_errors[:, 2])

    pylab.xlabel('Epochs')
    pylab.ylabel('Loss rate')
    pylab.title('Loss for train, valid and test sets')
    pylab.legend([train_loss_plot, valid_loss_plot, test_loss_plot], ['Train set', 'Valid set', 'Test set'])
    pylab.show()
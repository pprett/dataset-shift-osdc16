#!/usr/bin/python
"""
Checkerboard Example

Illustrates the phenomenon of covariate shift. That is, p(x,y) differs from training to testing phase. In particular, p(x) differs (given by the probability tabels) but p(y|x) remains fixed.

Example adopted from



"""

from __future__ import division
from abc import abstractmethod

import sys
import numpy as np

from sklearn.svm import SVC


def generate_data(sample_size=200, pd=[[0.4, 0.4], [0.1, 0.1]]):
    pd = np.array(pd)
    pd /= pd.sum()
    offset = 50
    bins = np.r_[np.zeros((1,)), np.cumsum(pd)]
    bin_counts = np.histogram(np.random.rand(sample_size), bins)[0]
    data = np.empty((0, 2))
    targets = []
    for ((i, j), p), count in zip(np.ndenumerate(pd), bin_counts):
        xs = np.random.uniform(low=0.0, high=50.0, size=count) + j * offset
        ys = np.random.uniform(low=0.0, high=50.0, size=count) + -i * offset
        data = np.vstack((data, np.c_[xs, ys]))
        if i == j:
            targets.extend([1] * count)
        else:
            targets.extend([-1] * count)
    return np.c_[data, targets]


class Model(object):
    def __init__(self):
        self.observers = []
        self.trainerr = "-"
        self.testerr = "-"
        self.surface = None
        self.train = np.zeros((0, 3))
        self.test = np.zeros((0, 3))
        self.sample_weight = np.zeros((0))

    def changed(self):
        for observer in self.observers:
            observer.update(self)

    def set_train(self, data):
        self.train = data

    def set_test(self, data):
        self.test = data

    def add_observer(self, observer):
        self.observers.append(observer)

    def set_testerr(self, testerr):
        self.testerr = testerr

    def set_trainerr(self, trainerr):
        self.trainerr = trainerr

    def set_surface(self, surface):
        self.surface = surface


class Controller(object):
    def __init__(self, model):
        self.model = model

    def generate_data(self):
        print("Controller: generate_data()")
        self.model.set_train(generate_data(pd=self.train_pd.get_pd()))
        self.model.set_test(generate_data(pd=self.test_pd.get_pd()))
        self.model.sample_weight = np.ones(self.model.train.shape[0])
        self.model.set_surface(None)
        self.model.set_testerr("-")
        self.model.set_trainerr("-")
        self.model.changed()

    def reweight(self, weight="none"):
        print("Controller: reweight(weight='%s')" % weight)
        self.model.set_surface(None)
        self.model.set_testerr("-")
        self.model.set_trainerr("-")
        if weight == "naive":
            p = self.test_pd.get_pd()
            q = self.train_pd.get_pd()
            weight_table = p / q

            X = self.model.train[:, :2]
            sample_weight = self.model.sample_weight
            for i, x in enumerate(X):
                if x[0] < 50.0 and x[1] >= 0.0:
                    sample_weight[i] = weight_table[0, 0]
                elif x[0] < 50.0 and x[1] < 0.0:
                    sample_weight[i] = weight_table[1, 0]
                elif x[0] >= 50.0 and x[1] >= 0.0:
                    sample_weight[i] = weight_table[0, 1]
                else:
                    sample_weight[i] = weight_table[1, 1]

        elif weight == "logreg":
            assert False
        else:
            sample_weight = np.ones(self.model.train.shape[0],
                                    dtype=np.float64)

        self.model.sample_weight = sample_weight
        self.model.changed()

    def classify(self, kernel="linear"):
        print("Controller: classify(kernel='%s')" % kernel)
        train = self.model.train

        samples = train[:, :2]
        labels = train[:, 2].ravel()

        try:
            sample_weight = self.model.sample_weight
        except AttributeError:
            sample_weight = np.ones(labels.shape, dtype=np.float64)

        # FIXME add hyperparameter tuning via CV.
        if kernel == 'linear':
            params = {'C': 0.1}
        elif kernel == 'rbf':
            params = {'C': 1., 'gamma': 0.0005}
        clf = SVC(kernel=kernel, probability=True, random_state=13,
                  **params)
        clf.fit(samples, labels, sample_weight=sample_weight)

        train_err = 1.0 - clf.score(samples,
                                    labels)
        test_err = 1.0 - clf.score(self.model.test[:, :2],
                                   self.model.test[:, 2].ravel())
        X1, X2, Z = self.decision_surface(clf)
        self.model.set_trainerr("%.2f" % train_err)
        self.model.set_testerr("%.2f" % test_err)
        self.model.set_surface((X1, X2, Z))
        self.model.changed()

    def decision_surface(self, clf):
        delta = 0.25
        x = np.arange(0.0, 100.1, delta)
        y = np.arange(-50.0, 50.1, delta)
        X1, X2 = np.meshgrid(x, y)
        XX = np.c_[X1.ravel(), X2.ravel()]
        Z = clf.predict_proba(XX)[:, 1].reshape(X1.shape)
        return X1, X2, Z

    def quit(self):
        sys.exit()

    def set_train_pd(self, train_pd):
        self.train_pd = train_pd

    def set_test_pd(self, test_pd):
        self.test_pd = test_pd


class View(object):
    """A view of the checkerboards app

    Attributes
    ----------
    train_pd : Table
        The training probability table
    testn_pd : Table
        The test probability table
    """
    def __init__(self, controller):
        self.controller = controller

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def update(self, model):
        pass


class Table(object):

    @abstractmethod
    def get_pd(self):
        pass

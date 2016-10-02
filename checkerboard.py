#!/usr/bin/python
"""
Checkerboard Example

Illustrates the phenomenon of covariate shift. That is, p(x,y) differs from training to testing phase. In particular, p(x) differs (given by the probability tabels) but p(y|x) remains fixed.

Example adopted from



"""

from __future__ import division
from abc import abstractmethod
import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import Tkinter as Tk
import sys
import numpy as np
from functools import partial

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
        print "Controller: generate_data()"
        self.model.set_train(generate_data(pd=self.train_pd.get_pd()))
        self.model.set_test(generate_data(pd=self.test_pd.get_pd()))
        self.model.sample_weight = np.ones(self.model.train.shape[0])
        self.model.set_surface(None)
        self.model.set_testerr("-")
        self.model.set_trainerr("-")
        self.model.changed()

    def reweight(self, weight="none"):
        print "Controller: reweight(weight='%s')" % weight
        self.model.set_surface(None)
        self.model.set_testerr("-")
        self.model.set_trainerr("-")
        if weight == "naive":
            p = np.array(self.test_pd.get_pd())
            q = np.array(self.train_pd.get_pd())
            print p
            print q
            weight_table = p / q
            print weight_table

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
        print "Controller: classify(kernel='%s')" % kernel
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
            params = {'C': 0.1, 'gamma': 0.001}
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

    def __init__(self, controller):
        self.controller = controller

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def update(self, model):
        pass


class TkView(View):

    def __init__(self, controller):
        super(TkView, self).__init__(controller)
        root = Tk.Tk()
        root.wm_title("Checkerboards")
        self.root = root

        f = Figure(figsize=(10, 5), dpi=100)
        canvas = FigureCanvasTkAgg(f, master=root)
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas, root)
        toolbar.update()
        canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.f = f
        self.toolbar = toolbar
        self.canvas = canvas
        self.hascolormaps = False
        self.trainerr_text = self.f.text(0.2, 0.05, "Errorrate = -")
        self.testerr_text = self.f.text(0.6, 0.05, "Errorrate = -")

        train_label = Tk.Label(root, text="Train Marginal Distribution:")
        train_label.pack(side=Tk.LEFT)
        train_pd = TkTable(np.array([[0.4, 0.4], [0.1, 0.1]]), root,
                           width=100, height=100)
        train_pd.pack(side=Tk.LEFT)
        test_label = Tk.Label(root, text="Test Marginal Distribution:")
        test_label.pack(side=Tk.LEFT)
        test_pd = TkTable(np.array([[0.4, 0.1], [0.4, 0.1]]), root,
                          width=100, height=100)
        test_pd.pack(side=Tk.LEFT)

        self.train_table = train_pd
        self.test_table = test_pd

        generate_button = Tk.Button(master=root,
                                    text='Generate Data',
                                    command=controller.generate_data)
        generate_button.pack(side=Tk.LEFT)

        reweight_button_box = Tk.Frame(master=root)
        reweight_button_label = Tk.Label(master=reweight_button_box,
                                         text="Reweight")
        reweight_button_label.pack()

        weight_none_button = Tk.Button(master=reweight_button_box,
                                       text='None',
                                       command=partial(controller.reweight,
                                                       weight="none"),
                                       width=10)
        weight_none_button.pack()

        weight_naive_button = Tk.Button(master=reweight_button_box,
                                        text='Naive',
                                        command=partial(controller.reweight,
                                                        weight="naive"),
                                        width=10)
        weight_naive_button.pack()
        reweight_button_box.pack(side=Tk.LEFT)

        classify_button_box = Tk.Frame(master=root)
        classify_button_label = Tk.Label(master=classify_button_box,
                                         text="Classify")
        classify_button_label.pack()
        svm_linear_button = Tk.Button(master=classify_button_box,
                                      text='Linear',
                                      command=partial(controller.classify,
                                                      kernel="linear"),
                                      width=10)
        svm_linear_button.pack()

        svm_rbf_button = Tk.Button(master=classify_button_box,
                                   text='RBF',
                                   command=partial(controller.classify,
                                                    kernel="rbf"),
                                   width=10)
        svm_rbf_button.pack()
        classify_button_box.pack(sid=Tk.RIGHT)

    def run(self):
        Tk.mainloop()

    def update(self, model):
        if hasattr(self, "train_plot"):
            self.train_plot.clear()

        if hasattr(self, "test_plot"):
            self.test_plot.clear()

        self.train_plot = self.f.add_subplot(121)
        self.test_plot = self.f.add_subplot(122)
        self.plot_data(self.train_plot, model.train,
                       sample_weight=model.sample_weight,
                       title="Training Distribution")
        self.plot_data(self.test_plot, model.test,
                       title="Test Distribution")
        self.plot_errors(model.trainerr, model.testerr)

        if model.surface is not None:
            CS = self.plot_decision_surface(self.train_plot, model.surface)
            CS = self.plot_decision_surface(self.test_plot, model.surface)
            self.plot_colormaps(CS)

        self.canvas.show()

    def plot_data(self, fig, data, sample_weight=None, title=""):
        if data.shape[0] > 0:
            pos_data = data[data[:, 2] == 1]
            neg_data = data[data[:, 2] == -1]
            if sample_weight is None:
                sample_weight = np.ones(data.shape[0])

            sample_weight = np.sqrt(sample_weight) * 10

            fig.scatter(pos_data[:, 0], pos_data[:, 1], c='w',
                        s=sample_weight[data[:, 2] == 1])
            fig.scatter(neg_data[:, 0], neg_data[:, 1], c='k',
                        s=sample_weight[data[:, 2] == -1])

        fig.set_ylim((-50, 50))
        fig.set_xlim((0, 100))
        fig.set_xticks([])
        fig.set_yticks([])
        fig.set_title(title)

    def plot_decision_surface(self, fig, surface):
        X1, X2, Z = surface
        levels = np.arange(0.0, 1.1, 0.1)
        CS = fig.contourf(X1, X2, Z, levels,
                          cmap=matplotlib.cm.bone,
                          origin='lower', alpha=0.7)
        fig.contour(X1, X2, Z, [0.5], colors="k", linestyles=["--"])
        return CS

    def plot_colormaps(self, CS):
        if not self.hascolormaps:
            self.f.colorbar(CS, ax=self.train_plot)
            self.f.colorbar(CS, ax=self.test_plot)
            self.hascolormaps = True

    def plot_errors(self, trainerr, testerr):
        self.trainerr_text.set_text("Errorrate = %s" % trainerr)
        self.testerr_text.set_text("Errorrate = %s" % testerr)


class Table(object):

    @abstractmethod
    def get_pd(self):
        pass


class TkTable(Table):
    def __init__(self, pd, *args, **kargs):
        master = Tk.Frame(*args, **kargs)
        self.master = master

        self.e1 = Tk.Entry(master, width=5)
        self.e1.insert(0, pd[0, 0])
        self.e2 = Tk.Entry(master, width=5)
        self.e2.insert(0, pd[0, 1])
        self.e3 = Tk.Entry(master, width=5)
        self.e3.insert(0, pd[1, 0])
        self.e4 = Tk.Entry(master, width=5)
        self.e4.insert(0, pd[1, 1])
        self.e1.grid(row=0, column=0)
        self.e2.grid(row=0, column=1)
        self.e3.grid(row=1, column=0)
        self.e4.grid(row=1, column=1)

    def get_pd(self):
        return [[float(self.e1.get()), float(self.e2.get())],
                 [float(self.e3.get()), float(self.e4.get())]]

    def pack(self, **kargs):
        self.master.pack(**kargs)

    def grid(self, **kargs):
        self.master.grid(**kargs)


# class BokehView(object):

#     def __init__(self, controller):
#         super(BokehView, self).__init__(controller)
#         from bokeh.io import output_file, show
#         from bokeh.layouts import widgetbox
#         from bokeh.models.widgets import Button

#         output_file("checkerboard.html")

#         self.gen_data_button = Button(label="Generate Data", button_type="success")

#     def run(self):
#         show(widgetbox(self.gen_data_button))

#     def update(self, model):
#         pass


def main(argv):

    model = Model()
    controller = Controller(model)
    view = TkView(controller)
    #view = BokehView(controller)
    model.add_observer(view)
    view.update(model)

    controller.set_train_pd(view.train_table)
    controller.set_test_pd(view.test_table)

    view.run()


if __name__ == "__main__":
    main(sys.argv)

import sys
import matplotlib
import Tkinter as Tk
import numpy as np

matplotlib.use('TkAgg')

from functools import partial
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
from matplotlib.figure import Figure

from checkerboard import View
from checkerboard import Table
from checkerboard import Controller
from checkerboard import Model


class TkView(View):
    """A view implementation using TK."""

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
        return np.array([[float(self.e1.get()), float(self.e2.get())],
                         [float(self.e3.get()), float(self.e4.get())]],
                        dtype=np.float)

    def pack(self, **kargs):
        self.master.pack(**kargs)

    def grid(self, **kargs):
        self.master.grid(**kargs)


def main(argv):
    model = Model()
    controller = Controller(model)
    view = TkView(controller)
    model.add_observer(view)
    view.update(model)

    controller.set_train_pd(view.train_table)
    controller.set_test_pd(view.test_table)

    view.run()


if __name__ == "__main__":
    main(sys.argv)

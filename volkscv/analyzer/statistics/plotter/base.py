import matplotlib.pyplot as plt

from ..base import Base
from ..utils import get_fig


class BasePlotter(Base):
    """ Base class of plotters.

    Args:
        data (Any): Data to be plotted.
        text (str): Name of figure.
        is_title (bool, optional): Whether set the text as title of the figure.
    """

    def __init__(self, data, text, is_title=False, axis_label=None):
        self.data = data
        self.text = text
        self.is_title = is_title
        self.axis_label = axis_label

    def attach_axis_label(self):
        if self.axis_label is None:
            return
        assert isinstance(self.axis_label, (list, tuple))
        assert len(self.axis_label) == 2
        plt.xlabel(self.axis_label[0])
        plt.ylabel(self.axis_label[1])

    def plot(self, func=None, *args, **kwargs):
        print(plt.get_figlabels())
        assert func is not None, 'You should provide a function to generate the figure'
        assert callable(func), 'func should be callable'
        func(*args, **kwargs)
        get_fig()


class Compose(Base):
    """Compose different plotters.

    Args:
        plotters (list): A list of plotter instances.
        text (str, 'optional): Name of figure.
        legend (list, optional): Legend of corresponding plotters.
        flag (bool, optional): If true, each plotter in self.plotters will
            generate a new figure. Default is False.

    Example:
        >>> import numpy as np
        >>> from volkscv.analyzer.statistics.plotter import OneDimPlotter
        >>> data_1 = [np.random.randint(0, 10) for _ in range(100)]
        >>> data_2 = [np.random.randint(0, 10) for _ in range(100)]
        >>> plotter1 = OneDimPlotter(data_1, 'data_1', plt.hist, bins=10)
        >>> plotter2 = OneDimPlotter(data_2, 'data_2', plt.hist, bins=10)
        >>> legend = ['data_1', 'data_2']
        >>> self = Compose([plotter1, plotter2], text='example', legend=legend)
        >>> self.plot()
        >>> # export
        >>> self.export('./result', save_mode='folder')
    """

    def __init__(self, plotters, text='Compose', legend=None, flag=False):
        self.plotters = plotters
        self.text = text
        self.legend = legend
        self.flag = flag

    def plot(self, func=None, **kwargs):
        """

        Args:
            func (callable): A callable function.
            kwargs (dict): Args of input func.
        """

        for idx, p in enumerate(self.plotters):
            if self.flag:
                self.figure(p.text)
                self.title(p.text)
            if func is None:
                p.plot()
            else:
                p.plot(func, **kwargs)
        if self.legend is not None:
            plt.legend(self.legend)

        get_fig()

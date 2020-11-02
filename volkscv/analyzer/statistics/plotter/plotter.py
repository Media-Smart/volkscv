import matplotlib.pyplot as plt

from .base import BasePlotter
from ..utils import get_fig


class OneDimPlotter(BasePlotter):
    """ Plot one dimensinal data.

    Args:
        data (np.ndarray): One dimensional data.
        text (str): Name of figure.
        func (callable, optional): Plot function.
        kwargs: Parameters of func.

    Example:
        >>> import random
        >>> import numpy as np
        >>> data = np.array([random.randint(0,100) for _ in range(100)])
        >>> self = OneDimPlotter(data, 'figure', plt.hist, bins=20)
        >>> self.plot()
        >>> # export the figure
        >>> self.export('./result', save_mode='folder')

    """

    def __init__(self, data, text, func=plt.hist, axis_label=None, **kwargs):
        super(OneDimPlotter, self).__init__(data, text, axis_label=axis_label)
        self.func = func
        self.kwargs = kwargs

    def plot(self, func=None, *args, **kwargs):
        if func is None:
            self.func(self.data, **self.kwargs)
        else:
            func(self.data, *args, **kwargs)
        self.attach_axis_label()
        get_fig()


class TwoDimPlotter(BasePlotter):
    """ Plot two dimensinal data.

    Args:
        data (list): Two dimensional data.
        text (str): Name of figure.
        func (callable, optional): Plot function.
        kwargs: Parameters of func.

    Example:
        >>> import random
        >>> import numpy as np
        >>> data = [np.array([random.randint(0,100) for _ in range(100)]),
        >>>         np.array([random.randint(0,100) for _ in range(100)])]
        >>> self = TwoDimPlotter(data, 'figure', plt.scatter, marker='-')
        >>> self.plot()
        >>> # export the figure
        >>> self.export('./result', save_mode='folder')
    """

    def __init__(self, data, text, func=plt.scatter, axis_label=None, **kwargs):
        super(TwoDimPlotter, self).__init__(data, text, axis_label=axis_label)
        self.func = func
        self.kwargs = kwargs

    def plot(self, func=None, *args, **kwargs):
        if func is None:
            self.func(self.data[0], self.data[1], **self.kwargs)
        else:
            func(self.data[0], self.data[1], *args, **kwargs)
        self.attach_axis_label()
        get_fig()


class SubPlotter(BasePlotter):
    """ Subplot a sequence of data.

    Args:
        data (dict): Data to be plotted.
        text (str): Name of figure.
        processor_type (str): Type of plotter processor.
        func (callable): Plot function.
        col (int): The columns of subplot figure.
        row (int): The rows of subplot figure.
        category (list, optional): Corresponding label of data.
        kwargs : Parameters of func.

    Example:
        >>> import numpy as np
        >>> import random
        >>> # one dim data
        >>> data = [[random.randint(0, 100) for __ in range(1000)] for _ in range(10)]
        >>> category = [str(i) for i in range(10)]
        >>> self = SubPlotter(data, 'figure1', 'one', plt.hist, 3, 4, category, bins=20)
        >>> self.plot()
        >>> # two dim data
        >>> data = {str(idx) : [[random.randint(0,100) for _ in range(1000)],
        >>>                     [random.randint(0, 100) for _ in range(1000)]]
        >>>         for idx in range(10)}
        >>> self = SubPlotter(data, 'figure2', 'two', plt.scatter, 3, 4,
        >>>                         category, marker='.', alpha=0.1)
        >>> self.plot()
        >>> # change the plot function
        >>> self.plot(plt.scatter, figsize=(10, 10), dpi=300, marker='*', alpha=0.1, linewidths=1)
        >>> # export the plot figure to a result folder
        >>> self.export('./result', save_mode='folder')
    """

    processors = {
        'one': OneDimPlotter,
        'two': TwoDimPlotter,
    }

    def __init__(self, data, text, processor_type, func, col, row, axis_label=None, category=None, **kwargs):
        assert isinstance(data, dict)
        super(SubPlotter, self).__init__(data, text, axis_label=axis_label)
        self.row = row
        self.col = col
        self.processor = []
        if category is not None:
            self.category = category
        else:
            self.category = list(data.keys())

        for key, value in data.items():
            key = str(key)
            if not hasattr(self, key):
                setattr(self, key, value)
                self.processor.append(self.processors[processor_type](value, func, **kwargs))

    def plot(self, func=None, figsize=(15, 15), dpi=150, **kwargs):
        exist_figlabels = plt.get_figlabels()
        if self.text in exist_figlabels:
            self.text = self.text + '_copy'
        self.figure(self.text, figsize=figsize, dpi=dpi)
        plt.clf()
        self.title(self.text)
        for idx, p in enumerate(self.processor):
            ax = plt.subplot(self.row, self.col, idx + 1)
            ax.set_title(str(self.category[idx]))
            if func is None:
                p.plot()
            else:
                p.plot(func, **kwargs)

        get_fig()

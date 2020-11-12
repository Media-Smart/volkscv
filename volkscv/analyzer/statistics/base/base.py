from abc import abstractmethod

import matplotlib.pyplot as plt

from ..utils import get_collect


class Base:
    """A series of interface."""

    @staticmethod
    def title(text, *args, **kwargs):
        plt.title(text, *args, **kwargs)

    @staticmethod
    def figure(*args, **kwargs):
        plt.figure(*args, **kwargs)

    @abstractmethod
    def plot(self, *args, **kwargs):
        pass

    @staticmethod
    def show():
        plt.show()

    @property
    def collect(self):
        return get_collect()

    def clear(self):
        """Clear the figure cache."""

        self.collect.clear()

    def savefig(self, name, path, *args, **kwargs):
        """Specify the name of cached figure and save it."""

        if name in self.collect._infos:
            ff = self.collect._infos[name]
            ff.savefig(path, *args, **kwargs)

    def export(self, save_path, *args, **kwargs):
        """Export the cached figure.
        Args:
            save_path (str): The directory path for saving file.
            save_mode (str):
                'folder': Export the cached figure into a folder.
                'png': Combine the cached figure into a png and export it.
                'pdf': Combine the cached figure and export it to a pdf.
            exist_ok (bool): If False, exist save path will report an error, else not.
            name (list): Only export those file whose file_name is in name.
        """

        self.collect.export(save_path, *args, **kwargs)

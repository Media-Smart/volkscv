import os
import threading
import warnings
from collections import OrderedDict
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages


class Collect(object):
    """ Collect figures and export to a file

    Args:
        args (Callable): A function, not neccessary

    Returns:
        obj (Collect)

    Example:
        >>> import random
        >>>
        >>> collect_files = Collect()
        >>>
        >>> def show(plot_data):
        >>>     plt.figure()
        >>>     plt.plot(plot_data)
        >>>     plt.show()
        >>> data = [random.randint(0, 10) for _ in range(1000)]
        >>> show(data)
        >>> name = plt.get_figlabels()[-1]
        >>> ff = plt.gcf()
        >>> collect_files.update_info(name, ff)
        >>> collect_files.export('./result', save_mode='folder')
    """
    _infos = OrderedDict()
    _instance_lock = threading.Lock()

    def check_name(self, name):
        if name in self._infos:
            # if the name has been collected, it will be replaced
            # by the newest one,
            self._infos.pop(name)

    def update_info(self, name, res):
        # collect the name and corresponding figure
        self.check_name(name)
        self._infos[name] = res

    def __new__(cls, *args, **kwargs):
        if not hasattr(Collect, "_instance"):
            with Collect._instance_lock:
                if not hasattr(Collect, "_instance"):
                    Collect._instance = object.__new__(cls)
        return Collect._instance

    def export(self, save_path, save_mode=None, exist_ok=True, *args, **kwargs):
        """ Export the saved files into a folder, pdf or a png.

        Args:
            save_path (str): File path or a dir path.
            save_mode (str, None): Save mode, 'pdf', 'png' or 'folder'.
            exist_ok (bool): If True, the exist file will be covered, else not export.
        """
        if len(self._infos) == 0:
            warnings.warn('Currently, there is no figure collected. '
                          'But it will still generate pdf file.',
                          category=UserWarning)
            return
        if save_mode is None:
            save_mode = 'folder'
        assert save_mode in ['pdf', 'png', 'folder'], "support export mode" \
                                                      " are ['pdf', 'png', 'folder']"
        dir_path = save_path if save_mode == 'folder' else os.path.dirname(save_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=exist_ok)
        if save_mode == 'pdf':
            self.export_pdf(save_path, *args, **kwargs)
        elif save_mode == 'png':
            self.export_png(save_path, *args, **kwargs)
        else:
            self.export_to_folder(save_path, *args, **kwargs)

    def figure_to_array(self, name=None):
        """ Transfer plt.figure to np.ndarray

        Args:
            name (list, optional): Transfer {name} file to np.ndarray if name is not None,
                else transfer all files to np.ndarray.

        Returns:
            shape (tuple): The max shape of figures.
            tmp_file (list): Transferred array of plt.figure.
            tmp_name (list): Corresponding names of tmp_file.
        """
        shape = [0, 0, 4]
        tmp_file = []
        tmp_name = []
        for key, value in self._infos.items():
            if name is not None and key not in name:
                continue
            buffer_ = BytesIO()
            value.savefig(buffer_, format='png')
            buffer_.seek(0)
            img = Image.open(buffer_)
            data = np.asarray(img)
            shape = list(map(lambda x: max(x), zip(shape, data.shape)))
            tmp_file.append(data)
            tmp_name.append(key)
        for k in tmp_name:
            self.clear(k)
        return shape, tmp_file, tmp_name

    def export_to_folder(self, save_path, name=None, cover=False):
        """ save the plt.figure into a folder

        Args:
            save_path (str): The dir path.
            name (str): If name is not None, export the {name} file, else export all.
            cover (bool): If True, the existed file will be covered,
                else generate a new file name.
        """
        _, tmp_file, tmp_name = self.figure_to_array(name)
        if tmp_file:
            for idx, (d, n) in enumerate(zip(tmp_file, tmp_name)):
                ll = Image.fromarray(d)
                s_i = str(idx) + '.png'
                while os.path.isfile(os.path.join(save_path, s_i)) and not cover:
                    s_i = s_i[:-4]
                    s_i = s_i + '_new.png'
                ll.save(os.path.join(save_path, s_i))

    def export_png(self, save_path, name=None):
        """ combine all the plt.figure as a png file and export to {save_path}

        Args:
            save_path (str): The file path.
            name (str): If name is not None, export the {name} file, else export all.
        """
        shape, tmp_file, _ = self.figure_to_array(name)
        if tmp_file:
            canvas = np.zeros((shape[0] * len(tmp_file), shape[1], shape[2]), dtype=np.uint8)
            start = 0
            for idx, d in enumerate(tmp_file):
                h_, w_, c_ = d.shape
                end = start + h_
                # canvas[idx * shape[0]:idx * shape[0] + h_, :w_, :c_] = d
                canvas[start:end, :w_, :c_] = d
                start = end
            canvas = canvas[:end, :, :]
            ll = Image.fromarray(canvas)
            ll.save(save_path)

    def export_pdf(self, save_path, name=None):
        """ export the plt.figure to a pdf file

        Args:
            save_path (str): The file path.
            name (str): If name is not None, export the {name} file, else export all.

        """
        with PdfPages(save_path) as pdf:

            if name is None:
                for key, value in self._infos.items():
                    pdf.savefig(value)
                self.clear()
            else:
                for n in name:
                    if self._infos.get(n, False):
                        ff = self._infos.get(n)
                        pdf.savefig(ff)
                        self.clear(n)

    def clear(self, v=None):
        if v is None:
            self._infos.clear()
        else:
            self._infos.pop(v)

    def __len__(self):
        return len(self._infos)


def get_collect():
    """
    Returns:
        obj (Collect)

    """

    return Collect()


def get_fig():
    """ get the current plt.figure and saved to obj (Collect)

    Examples:
        >>> import random
        >>> def show():
        >>>     plt.figure('example')
        >>>     plt.hist([random.randint(0,100) for _ in range(100)])
        >>>     get_fig()
        >>> show()
        >>> cc = get_collect()
        >>> cc._infos.get('example')
    """

    collect = get_collect()
    name = plt.get_figlabels()[-1]
    ff = plt.gcf()
    collect.update_info(name, ff)

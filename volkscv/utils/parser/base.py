from abc import abstractmethod, ABCMeta

import cv2

from .utils import read_imglist


class BaseParser(metaclass=ABCMeta):
    """The base class of parser, a help for data parser.

    All subclasses should implement the following APIs:v

    - ``__call__()``

    Args:
        imgs_folder (str, optional): Path of folder for Images. Default: ''.
        txt_file (str, optional): Required image paths. Default: None.

            Examples:
                xxx.jpg
                xxx.jpg
                xxxx.jpg

        extensions (str): Image extension. Default: 'jpg'.
    """

    def __init__(self,
                 imgs_folder='',
                 txt_file=None,
                 extension='jpg',
                 ):
        self.imgs_folder = imgs_folder
        self.txt_file = txt_file
        self.extension = extension

        if self.txt_file is not None:
            self.imgs_list, _ = read_imglist(self.txt_file)
            assert len(self.imgs_list), 'The txt file is empty.'
        else:
            self.imgs_list = None

        self._result = dict(img_names=None,
                            categories=None,
                            shapes=None,
                            bboxes=None,
                            labels=None,
                            segs=None,
                            scores=None,
                            bboxes_ignore=None,
                            labels_ignore=None, )

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, value):
        self._result.update(value)

    @staticmethod
    def _get_shape(fname):
        """Get image size.

        Args:
            fname (str): Absolute path of image.

        Returns:
            tuple: Image size.
        """

        img = cv2.imread(fname)
        return img.shape[0:2]

    @abstractmethod
    def __call__(self, need_shape):
        """Parse dataset.

        Args:
            need_shape (bool): Whether need shape attribute.

        Returns:
            dict: Annotations.
        """

        return self.result

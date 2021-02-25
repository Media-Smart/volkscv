import os

import numpy as np

from .base import BaseParser


class ImageParser(BaseParser):
    """class of parser for torchvision ImageFolder format

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    """

    def __init__(self, **kwargs):
        super(ImageParser, self).__init__(**kwargs)

        self.categories = os.listdir(self.imgs_folder)

    def __call__(self, need_shape=True):
        fname_list, shapes_list, labels_list = [], [], []
        for idx, label in enumerate(self.categories):
            subfolder = os.path.join(self.imgs_folder, label)
            if os.path.isdir(subfolder):
                for fname in os.listdir(subfolder):
                    if self.imgs_list is not None and fname not in self.imgs_list:
                        continue
                    fname = os.path.join(subfolder, fname)
                    height, width = self._get_shape(fname) if need_shape else (0, 0)
                    fname_list.append(fname)
                    shapes_list.append([width, height])
                    labels_list.append(self.categories.index(label))

        self.result = dict(
            img_names=np.array(fname_list),
            categories=np.array(self.categories),
            shapes=np.array(shapes_list),
            labels=np.array(labels_list),
        )

        return self.result

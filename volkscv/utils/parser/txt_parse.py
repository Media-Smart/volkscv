import os

import numpy as np

from .base import BaseParser
from .utils import read_imglist


class TXTParser(BaseParser):
    """Class of parser for classification TXT annotation file.

        xxx.png dog
        xxx.png cat
        xxxx.png dog

    Args:
        anno_path (str): Path of annotation file.
        categories (list or tuple): All categories of data.
    """

    def __init__(self,
                 categories=None,
                 **kwargs):
        super(TXTParser, self).__init__(**kwargs)

        self.categories = categories
        assert self.imgs_list is not None, \
            "For txt file parser, the imgs_list attribute shouldn't be None."

    def __call__(self, need_shape=True):
        fname_list = []
        labels_list = []
        shapes_list = []
        scores_list = []
        fnames, annos = read_imglist(self.txt_file)
        for fname, anno in zip(fnames, annos):
            fname = os.path.join(self.imgs_folder, fname)
            height, width = self._get_shape(fname) if need_shape else (0, 0)
            shapes_list.append([width, height])
            fname_list.append(fname)
            assert anno[0] in self.categories, \
                f'Label: {anno[0]} is not in categories.'
            labels_list.append(self.categories.index(anno[0]))
            if len(anno) > 1:
                scores_list.append(float(anno[1]))

        self.result = dict(
            img_names=np.array(fname_list),
            categories=np.array(self.categories),
            shapes=np.array(shapes_list),
            labels=np.array(labels_list),
            scores=np.array(scores_list) if len(scores_list) else None,
        )
        return self.result

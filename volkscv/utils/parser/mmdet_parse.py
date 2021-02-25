import os
import json
from collections import defaultdict

import numpy as np

from .base import BaseParser


class MMDETParser(BaseParser):
    """Class of parser for mmdetection bbox result format.

    Args:
        anno_path (str): Path of annotation file.
        imgid2filename (dict): Mapping relations between image id and filename.

            Examples:
                imgid2filename = dict(1='example1.jpg', 2='example2.jpg')

        categories (list or tuple): All categories of data.
        label_start (int): The first Label index. Default: 1.
    """

    def __init__(self,
                 anno_path,
                 imgid2filename,
                 categories=None,
                 label_start=1,
                 **kwargs):
        super(MMDETParser, self).__init__(**kwargs)

        self.categories = categories
        self.label_start = label_start
        self.imgid2filename = imgid2filename

        self.data = self.load_data(anno_path)

    def load_data(self, anno_path):
        data = json.load(open(anno_path, 'r'))
        result = defaultdict(list)
        for item in data:
            fname = self.imgid2filename[item['image_id']]
            x1, y1, w, h = item['bbox']
            bbox = list(map(float, [x1, y1, x1 + w, y1 + h]))
            result[fname].append(
                [bbox, item['score'], item['category_id'] - self.label_start])
        return result

    def __call__(self, need_shape=True):
        fname_list, shapes_list, bboxes_list, labels_list, scores_list, \
        segs_list = [], [], [], [], [], []
        for id, name in self.imgid2filename.items():
            if self.imgs_list is not None and name not in self.imgs_list:
                continue

            if name in self.data:
                bboxes, scores, category_ids = list(zip(*self.data[name]))
                bboxes = np.array(bboxes)
                scores = np.array(scores)
                category_ids = np.array(category_ids)
            else:
                bboxes = np.zeros((0, 4))
                scores = np.zeros((0,))
                category_ids = np.zeros((0,))
            fname = os.path.join(self.imgs_folder, name)
            height, width = self._get_shape(fname) if need_shape else (0, 0)

            fname_list.append(fname)
            shapes_list.append([width, height])
            bboxes_list.append(bboxes)
            scores_list.append(scores)
            labels_list.append(category_ids)

        self.result = dict(
            img_names=np.array(fname_list),
            categories=np.array(self.categories),
            shapes=np.array(shapes_list),
            bboxes=np.array(bboxes_list),
            labels=np.array(labels_list),
            segs=np.array(segs_list),
            scores=np.array(scores_list),
        )

        return self.result

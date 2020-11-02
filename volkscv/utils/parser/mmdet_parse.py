import json
import os
from collections import defaultdict

import numpy as np

from .base import BaseParser


class MMDETParser(BaseParser):
    """Class of parser for mmdetection bbox result format.

    Args:
        anno_path (str): Path of annotation file.
        categories (list or tuple): All categories of data.
        label_start (int): The first Label index. Default: 1.
    """

    def __init__(self,
                 anno_path,
                 categories=None,
                 label_start=1,
                 **kwargs):
        super(MMDETParser, self).__init__(**kwargs)

        self.categories = categories
        self.label_start = label_start

        self.data = json.load(open(anno_path, 'r'))

    def __call__(self, need_shape=True):
        fname_list = []
        shapes_list = []
        bboxes_list = []
        labels_list = []
        scores_list = []
        segs_list = []
        data = defaultdict(list)
        for item in self.data:
            if self.imgs_list is not None and \
                    item['image_id'] not in self.imgs_list:
                continue
            fname = os.path.join(self.imgs_folder,
                                 f"{item['image_id']}.{self.extension}")
            x1, y1, w, h = item['bbox']
            bbox = list(map(float, [x1, y1, x1 + w, y1 + h]))
            data[fname].append(
                [bbox, item['score'], item['category_id'] - self.label_start])

        for key, value in data.items():
            bbox = []
            score = []
            category_id = []
            height, width = self._get_shape(key) if need_shape else (0, 0)
            for v in value:
                bbox.append(v[0])
                score.append(v[1])
                category_id.append(v[2])
            fname_list.append(key)
            shapes_list.append([width, height])
            bboxes_list.append(np.array(bbox))
            scores_list.append(np.array(score))
            labels_list.append(np.array(category_id))

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

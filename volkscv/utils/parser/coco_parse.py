import json
import os
from collections import defaultdict

import numpy as np

from .base import BaseParser
from .utils import filter_imgs


class COCOParser(BaseParser):
    """Class of parser for COCO data format.

    Args:
        anno_path (str): Path of annotation file.
        ignore (bool): If set True, some qualified annotations will be ignored.
        min_size (int or float, optional): The minimum size of bounding
            boxes in the images. If the size of a bounding box is less than
            ``min_size``, it would be add to ignored field.
    """

    def __init__(self,
                 anno_path,
                 ignore=True,
                 min_size=None,
                 **kwargs):
        super(COCOParser, self).__init__(**kwargs)

        self.ignore = ignore
        self.min_size = min_size

        self.data = json.load(open(anno_path, 'r'))
        self.categories = [cat['name'] for cat in self.data['categories']]
        self.cat_ids = [cat['id'] for cat in self.data['categories']]
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        self.img2anns = defaultdict(list)
        for ann in self.data['annotations']:
            self.img2anns[ann['image_id']].append(ann)

    def _check_ignore(self, ann):
        """Check whether the box needs to be ignored or not.

        Args:
            ann: Annotation of a box.
        """

        return ann.get('ignore', False) or \
               filter_imgs(ann['bbox'], self.min_size, format='xywh') or \
               ann.get('iscrowd', False)

    def __call__(self, need_shape=True):
        fname_list = []
        shapes_list = []
        bboxes_list = []
        labels_list = []
        segs_list = []
        bboxes_ignore_list = []
        labels_ignore_list = []
        for img in self.data['images']:
            if self.imgs_list is not None and \
                    img['file_name'] not in self.imgs_list:
                continue

            img_id = img['id']
            fname = os.path.join(self.imgs_folder, img['file_name'])
            if img.get('width') and img.get('height'):
                height, width = img['height'], img['width']
            else:
                height, width = self._get_shape(fname) if need_shape else (0, 0)

            ann_info = [ann for ann in self.img2anns[img_id]]

            bboxes = []
            labels = []
            segs = []
            bboxes_ignore = []
            labels_ignore = []
            for i, ann in enumerate(ann_info):
                ignore = self.ignore and self._check_ignore(ann)
                x1, y1, w, h = ann['bbox']
                inter_w = max(0, min(x1 + w, width) - max(x1, 0))
                inter_h = max(0, min(y1 + h, height) - max(y1, 0))
                if inter_w * inter_h == 0:
                    continue
                if ann['area'] <= 0 or w < 1 or h < 1:
                    continue
                if ann['category_id'] not in self.cat_ids:
                    continue
                bbox = list(map(float, [x1, y1, x1 + w, y1 + h]))
                if ignore:
                    bboxes_ignore.append(bbox)
                    labels_ignore.append(self.cat2label[ann['category_id']])
                else:
                    bboxes.append(bbox)
                    labels.append(self.cat2label[ann['category_id']])
                    segs.append(ann['segmentation'])

            if not bboxes:
                bboxes = np.zeros((0, 4))
                labels = np.zeros((0,))
            else:
                bboxes = np.array(bboxes)
                labels = np.array(labels)

            if not bboxes_ignore:
                bboxes_ignore = np.zeros((0, 4))
                labels_ignore = np.zeros((0,))
            else:
                bboxes_ignore = np.array(bboxes_ignore)
                labels_ignore = np.array(labels_ignore)

            fname_list.append(fname)
            shapes_list.append([width, height])
            bboxes_list.append(bboxes)
            labels_list.append(labels)
            bboxes_ignore_list.append(bboxes_ignore)
            labels_ignore_list.append(labels_ignore)
            segs_list.append(np.array(segs))

        self.result = dict(
            img_names=np.array(fname_list),
            categories=np.array(self.categories),
            shapes=np.array(shapes_list),
            bboxes=np.array(bboxes_list),
            labels=np.array(labels_list),
            segs=np.array(segs_list),
            bboxes_ignore=np.array(bboxes_ignore_list),
            labels_ignore=np.array(labels_ignore_list),
        )

        return self.result

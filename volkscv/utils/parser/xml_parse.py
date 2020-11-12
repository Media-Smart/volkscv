import os
import xml.etree.ElementTree as ET

import numpy as np

from .base import BaseParser
from .utils import filter_imgs


class XMLParser(BaseParser):
    """Class of parser for XML annotation file.

    Args:
        categories (list or tuple): All categories of data.
        xml_prefix (str): Prefix for XML file path. Default: ''.
        ignore (bool): If set True, some qualified annotations will be ignored.
        min_size (int or float, optional): The minimum size of bounding
            boxes in the images. If the size of a bounding box is less than
            ``min_size``, it would be add to ignored field.
        offset (int): Box offset need to be subtracted. Default: 1.
    """

    def __init__(self,
                 categories=None,
                 xmls_folder='',
                 ignore=True,
                 min_size=None,
                 offset=1,
                 **kwargs):
        super(XMLParser, self).__init__(**kwargs)

        self.categories = categories
        self.xmls_folder = xmls_folder
        self.ignore = ignore
        self.min_size = min_size
        self.offset = offset
        assert self.xmls_folder, "'xmls_folder' shouldn't be empty."

        assert self.imgs_list is not None, \
            "For txt file parser, the imgs_list attribute shouldn't be None."

        self.cat2label = {cat: i for i, cat in enumerate(self.categories)}

    def __call__(self, need_shape=True):
        fname_list = []
        shapes_list = []
        bboxes_list = []
        labels_list = []
        bboxes_ignore_list = []
        labels_ignore_list = []
        for img_id in self.imgs_list:
            fname = os.path.join(self.imgs_folder, f'{img_id}.{self.extension}')
            xml_path = os.path.join(self.xmls_folder, f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                height, width = self._get_shape(fname) if need_shape else (0, 0)

            tree = ET.parse(xml_path)
            root = tree.getroot()
            bboxes = []
            labels = []
            bboxes_ignore = []
            labels_ignore = []
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name not in self.categories:
                    continue
                label = self.cat2label[name]
                difficult = int(obj.find('difficult').text)
                bnd_box = obj.find('bndbox')
                bbox = [
                    float(bnd_box.find('xmin').text),
                    float(bnd_box.find('ymin').text),
                    float(bnd_box.find('xmax').text),
                    float(bnd_box.find('ymax').text)
                ]

                if self.ignore:
                    ignore = filter_imgs(bbox, self.min_size, format='xyxy')
                    if difficult or ignore:
                        bboxes_ignore.append(bbox)
                        labels_ignore.append(label)
                    else:
                        bboxes.append(bbox)
                        labels.append(label)
                else:
                    bboxes.append(bbox)
                    labels.append(label)

            if not bboxes:
                bboxes = np.zeros((0, 4))
                labels = np.zeros((0,))
            else:
                bboxes = np.array(bboxes) - self.offset
                labels = np.array(labels)

            if not bboxes_ignore:
                bboxes_ignore = np.zeros((0, 4))
                labels_ignore = np.zeros((0,))
            else:
                bboxes_ignore = np.array(bboxes_ignore) - self.offset
                labels_ignore = np.array(labels_ignore)

            fname_list.append(fname)
            shapes_list.append([width, height])
            bboxes_list.append(bboxes)
            labels_list.append(labels)
            bboxes_ignore_list.append(bboxes_ignore)
            labels_ignore_list.append(labels_ignore)

        self.result = dict(
            img_names=np.array(fname_list),
            categories=np.array(self.categories),
            shapes=np.array(shapes_list),
            bboxes=np.array(bboxes_list),
            labels=np.array(labels_list),
            bboxes_ignore=np.array(bboxes_ignore_list),
            labels_ignore=np.array(labels_ignore_list),
        )

        return self.result

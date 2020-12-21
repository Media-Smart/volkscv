from functools import lru_cache

import cv2
import numpy as np

from .base import BaseVis
from .utils import draw_image, generate_mpl_figure


class ClsVis(BaseVis):
    """Class of visualization for classification task."""

    @lru_cache(maxsize=32)
    def img_process(self,
                    fname,
                    show_fpfn=False,
                    show_ori=False,
                    category_to_show=None,
                    show_score=True):
        img = cv2.imread(fname)
        data, labels, flag = self.get_single_data(fname, category_to_show)

        if not flag:
            return img, flag

        if show_fpfn:
            assert len(data) == 2, "Show fpfn need both ground truth file" \
                                   " and prediction file."
            if data['pred']['labels'] == data['gt']['labels']:
                return None, False

        imgs, title = {}, self.default_title.copy()
        if show_ori:
            imgs.update({'ori': img})

        for key in data.keys():
            img_, text = draw_image(img, key, data,
                                    self.categories,
                                    show_score=show_score)
            imgs.update({key: img_})
            title.update({key: f'{title[key]}  {text}'})

        img_ = generate_mpl_figure(imgs, fname, title)

        return img_, flag

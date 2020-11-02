from functools import lru_cache

import cv2

from .base import BaseVis
from .utils import draw_image, generate_mpl_figure


class ClsVis(BaseVis):
    """Class of visualization for classification task."""

    @lru_cache(maxsize=32)
    def img_process(self,
                    fname,
                    show_ori=False,
                    category_to_show=None,
                    show_score=True):
        img = cv2.imread(fname)
        data, labels, flag = self.get_single_data(fname, category_to_show)

        if not flag:
            return img, flag

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

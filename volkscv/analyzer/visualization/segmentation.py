from functools import lru_cache

import cv2

from .base import BaseVis
from .utils import draw_mask, generate_legend, generate_mpl_figure, cal_seg_fpfn


class SegVis(BaseVis):
    """Class of visualization for segmentation task."""

    @lru_cache(maxsize=32)
    def img_process(self,
                    fname,
                    show_ori=False,
                    show_fpfn=False,
                    show_score=True,
                    category_to_show=None,
                    score_thr=-1,
                    base_fontscale=0.5):
        img = cv2.imread(fname)
        data, labels, flag = self.get_single_data(fname, category_to_show)

        if not flag:
            return img, flag

        title = self.default_title.copy()
        draw_params = dict()
        if show_fpfn:
            fp_mask, fn_mask = cal_seg_fpfn(data, img.shape,
                                            self.colors,
                                            self.categories,
                                            category_to_show=category_to_show)
            draw_params.update({'fp': fp_mask, 'fn': fn_mask})

        imgs = {}
        if show_ori:
            imgs.update({'ori': img})
        for key in data.keys():
            img_, _ = draw_mask(img, key, data,
                                self.colors,
                                self.categories,
                                show_fpfn=show_fpfn,
                                show_score=show_score,
                                category_to_show=category_to_show,
                                score_thr=score_thr,
                                base_fontscale=base_fontscale,
                                **draw_params)
            imgs.update({key: img_})

        legend = generate_legend(self.colors, set(labels))
        img_ = generate_mpl_figure(imgs, fname, title, legend)

        return img_, flag

from functools import lru_cache

import cv2

from .base import BaseVis
from .utils import draw_bbox, cal_det_fpfn, generate_legend, generate_mpl_figure


class DetVis(BaseVis):
    """Class of visualization for detection task."""

    @lru_cache(maxsize=32)
    def img_process(self,
                    fname,
                    show_ori=False,
                    show_fpfn=False,
                    show_score=False,
                    show_fpfn_format='line',
                    show_ignore=False,
                    category_to_show=None,
                    iou_thr=0.5,
                    score_thr=-1,
                    base_thickness=1,
                    base_fontscale=0.5):
        img = cv2.imread(fname)
        data, labels, flag = self.get_single_data(fname, category_to_show)

        if not flag:
            return img, flag

        title = self.default_title.copy()

        draw_params = dict()
        if show_fpfn:
            fp, tp, fn, tn = cal_det_fpfn(data,
                                          self.categories,
                                          iou_thr=iou_thr,
                                          score_thr=score_thr,
                                          category_to_show=category_to_show)

            draw_params.update({'fp': fp, 'tp': tp, 'tn': tn, 'fn': fn})
            title.update({'gt': f"{title['gt']}  fn:{sum(fn)}"})
            title.update({'pred': f"{title['pred']}  fp:{sum(fp)}"})

        imgs = {}
        if show_ori:
            imgs.update({'ori': img})
        for key in data.keys():
            img_, _ = draw_bbox(img, key, data,
                                self.colors,
                                self.categories,
                                category_to_show=category_to_show,
                                show_score=show_score,
                                show_fpfn=show_fpfn,
                                show_fpfn_format=show_fpfn_format,
                                show_ignore=show_ignore,
                                score_thr=score_thr,
                                base_thickness=base_thickness,
                                base_fontscale=base_fontscale,
                                **draw_params)

            imgs.update({key: img_})

        legend = generate_legend(self.colors, set(labels))
        img_ = generate_mpl_figure(imgs, fname, title, legend)

        return img_, flag

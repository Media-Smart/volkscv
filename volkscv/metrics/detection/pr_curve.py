import os
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from .base import COCOAnalysis


class PRCurve(COCOAnalysis):

    def accumulate(self):
        super().accumulate()

        if self.areaRng is not None:
            warnings.warn(f'PR Curve for different area range is not supported yet!')

        accumulate_state = {
            'precision': self.precision,
            'recall': self.recall,
            'score': self.score,
        }
        return accumulate_state

    def get_valid_iou(self, ious):
        _ious = []
        if ious is None:
            if self.iou is not None:
                _ious = self.cocoEval.params.iouThrs
            else:
                _ious = (0.5,)
        else:
            if self.iou is not None:
                for iou in ious:
                    if iou in self.cocoEval.params.iouThrs:
                        _ious.append(iou)
                    else:
                        _ious.append(None)
                        warnings.warn(f'iou:({iou}) needs to be specified in Class initialization!')
            else:
                for iou in ious:
                    if iou in np.arange(0.5, 0.955, 0.05).round(2):
                        _ious.append(iou)
                    else:
                        _ious.append(None)
                        warnings.warn(f'No iou specified in Class initialization! '
                                      f'iou: {iou} needs to be the integral multiple of '
                                      f'0.05 in [0.5. 0.95] for default setting')
        if not _ious:
            raise ValueError('No suitable iou setting for pr curve drawing!')
        return _ious

    def export(self, export_path='.', with_anno=True, ious=None, colors=('crimson', ), **kwargs):
        ious = self.get_valid_iou(ious)
        assert len(ious) <= len(colors), \
            'number of colors is less than number of curves, please specify enough colors'
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        os.makedirs(export_path, exist_ok=True)

        for cat_id in range(self.precision.shape[2]):
            plt.figure(11, figsize=(9, 9), dpi=400)
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.grid(True)
            plt.plot(np.arange(0.0, 1.01, 0.01), np.arange(0.0, 1.01, 0.01), color='royalblue', linestyle='--')
            plt.annotate("balance line", xy=(0.5, 0.5), color='royalblue', rotation=45, xytext=(
                -20, 0), textcoords='offset points')

            for i, iou in enumerate(ious):
                if iou is not None:
                    line_kwargs = {
                        'label': f'iou: {iou}',
                        'color': colors[i],
                        'marker': '.',
                        'linestyle': '-',
                        'linewidth': 1,
                    }
                    line_kwargs.update(**kwargs)

                    if self.iou is None:
                        iou_id = round((iou-0.5) / 0.05)
                    else:
                        iou_id = np.argwhere(self.cocoEval.params.iouThrs == iou)[0][0]

                    plt.plot(np.arange(0.0, 1.01, 0.01), self.precision[iou_id, :, cat_id, 0, -1], **line_kwargs)
                    plt.legend(loc='lower left')
                    if with_anno:
                        for j, x in enumerate(np.arange(0.0, 1.01, 0.01)):
                            text = [round(x, 3),
                                    round(self.precision[iou_id, j, cat_id, 0, -1], 3),
                                    round(self.score[iou_id, j, cat_id, 0, -1], 3)]
                            plt.annotate(text, xy=(x, self.precision[iou_id, j, cat_id, 0, -1]),
                                         xytext=(x, self.precision[iou_id, j, cat_id, 0, -1] + 0.005),
                                         color=line_kwargs['color'], fontsize=3, rotation=80)
            plt.tight_layout()
            plt.savefig(os.path.join(export_path, timestamp + f'_pr_curve_of_cat_{cat_id}'), dpi=400)
            plt.close()

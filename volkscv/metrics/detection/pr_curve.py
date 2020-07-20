import os
import time

import numpy as np
import matplotlib.pyplot as plt

from .base import COCOAnalysis


class PRCurve(COCOAnalysis):

    def __init__(self, ious=None):
        super().__init__()
        self.ious = ious

    def compute(self, pred_path, target_path):
        super().compute(pred_path, target_path)
        # self.ap_iou = np.linspace(0.1, 0.95, np.round((0.95 - .1) / .05) + 1, endpoint=True).round(3)
        if self.ious is not None:
            self.cocoEval.params.iouThrs = np.sort(np.unique(np.array(self.ious)), axis=0)

    def accumulate(self):
        super().accumulate()
        accumulate_state = {
            'precision': self.precision,
            'recall': self.recall,
            'score': self.score,
        }
        return accumulate_state

    def export(self, export_path=None, with_anno=True, ious=None, colors=('crimson', ), **kwargs):

        if self.ious is not None:
            if ious is None:
                ious = self.ious
            else:
                for iou in ious:
                    assert iou in self.ious, f'iou:({iou}) needs to be specified in class initialization'
        else:
            if ious is None:
                ious = (0.5, )

        assert len(ious) <= len(colors), \
            'number of colors is less than number of curves, please specify enough colors'

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        if export_path is None:
            raise NotADirectoryError('export_path must be specified!')
        else:
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
                line_kwargs = {
                    'label': f'iou: {iou}',
                    'color': colors[i],
                    'marker': '.',
                    'linestyle': '-',
                    'linewidth': 1,
                }
                line_kwargs.update(**kwargs)

                if self.ious is None:
                    assert iou in np.arange(0.5, 0.955, 0.05).round(2), \
                        f'iou: {iou} is not supported, iou needs to be the integral multiple of 0.05 in [0.5. 0.95]'
                    iou_id = round((iou-0.5) / 0.05)
                else:
                    iou_id = np.argwhere(self.cocoEval.params.iouThrs == iou)[0][0]

                plt.plot(np.arange(0.0, 1.01, 0.01), self.precision[iou_id, :, cat_id, 0, 2], **line_kwargs)
                plt.legend(loc='lower left')
                if with_anno:
                    for j, x in enumerate(np.arange(0.0, 1.01, 0.01)):
                        text = [round(x, 3),
                                round(self.precision[iou_id, j, cat_id, 0, 2], 3),
                                round(self.score[iou_id, j, cat_id, 0, 2], 3)]
                        plt.annotate(text, xy=(x, self.precision[iou_id, j, cat_id, 0, 2]),
                                     xytext=(x, self.precision[iou_id, j, cat_id, 0, 2] + 0.005),
                                     color=line_kwargs['color'], fontsize=3, rotation=80)

            plt.tight_layout()
            plt.savefig(os.path.join(export_path, timestamp + f'_pr_curve_of_cat_{cat_id}'), dpi=400)
            plt.close()

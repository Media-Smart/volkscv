import os

import numpy as np
import matplotlib.pyplot as plt

from .base import COCOAnalysis


class PRCurve(COCOAnalysis):

    def accumulate(self):
        super().accumulate()
        accumulate_state = {
            'precision': self.precision,
            'recall': self.recall,
            'score': self.score,
        }
        return accumulate_state

    def export(self, export_path='.', with_anno=True, ious=(0.5, ), colors=('crimson', ), **kwargs):

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

                assert iou in np.arange(0.5, 0.955, 0.05).round(2), \
                    f'iou: {iou} is not supported, iou needs to be the integral multiple of 0.05 in [0.5. 0.95]'

                iou_id = round((iou-0.5) / 0.05)
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
            plt.savefig(os.path.join(export_path, f'pr_curve_of_cat_{cat_id}'), dpi=400)
            plt.close()

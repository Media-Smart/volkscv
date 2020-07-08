import os

import numpy as np
import matplotlib.pyplot as plt

from .base import BaseMetric
from .tools import precision_recall_curve


class PRCurve(BaseMetric):
    """
    Calculate value(precision, recall, thresholds) and export image of pr curve.
    Note: this implementation is restricted to the binary, multiclass classification
          task.

    Args:
        num_classes (int): number of classes. For num_classes=n, default
                           categorical indexes are 0 ~ n-1
    Examples:
        >>> pr_curve = PRCurve(num_classes=5)
        >>> for batch in dataloader:
        ...     # calculate pr value of current batch
        ...     pr_curve_batch = pr_curve(pred, target)
        >>> # calculate pr value of the epoch
        >>> pr_curve_epoch = pr_curve.accumulate()
        >>> # export image of pr curve
        >>> pr_curve.export(export_path='.', )
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__()

    def reset(self):
        self.y_true = None
        self.probas_pred = None

    def compute(self, pred, target):
        self._check_pred_range(pred)
        self._y_true_temp = target
        self._probas_pred_temp = pred
        return None

    def update(self, n=1):
        def concatenator(total, temp):
            if total is None:
                total = temp
            else:
                total = np.concatenate((total, temp), axis=0)
            return total

        self.y_true = concatenator(self.y_true, self._y_true_temp)
        self.probas_pred = concatenator(self.probas_pred, self._probas_pred_temp)

    def accumulate(self):
        self.accumulate_state = {}
        for cat_id in range(self.num_classes):
            precision, recall, thresholds = precision_recall_curve(self.y_true,
                                                                   self.probas_pred[:, cat_id],
                                                                   pos_label=cat_id)
            self.accumulate_state[str(cat_id)] = {
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds,
            }
        return self.accumulate_state

    def export(self, export_path='.', **kwargs):

        os.makedirs(export_path, exist_ok=True)

        for cat_id in self.accumulate_state.keys():

            plt.figure(11, figsize=(9, 9), dpi=400)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.grid(True)
            plt.plot(np.arange(0.0, 1.01, 0.01), np.arange(0.0, 1.01, 0.01),
                     color='royalblue', linestyle='--')
            plt.annotate("balance line", xy=(0.5, 0.5), color='royalblue',
                         rotation=45, xytext=(-20, 0), textcoords='offset points')

            line_kwargs = {
                'label': f'cat_id: {cat_id}',
                'color': 'crimson',
                'linestyle': '-',
                'linewidth': 1,
            }
            line_kwargs.update(**kwargs)

            plt.plot(self.accumulate_state[cat_id]['recall'],
                     self.accumulate_state[cat_id]['precision'],
                     **line_kwargs)

            plt.legend(loc='lower left')
            plt.tight_layout()
            plt.savefig(os.path.join(export_path, f'pr_curve_of_cat_{cat_id}'), dpi=400)
            plt.close()

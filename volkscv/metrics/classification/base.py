import numpy as np

from ..base import BaseMetric


BaseClsMetric = BaseMetric


class BaseScoreCurve(BaseClsMetric):

    def reset(self):
        self.y_true = None
        self.probas_pred = None

    def compute(self, pred, target):
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

    def check(self, pred, target):
        super().check(pred, target)
        self._check_pred_range(pred)

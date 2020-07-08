import numpy as np

from .base import BaseMetric
from .libs import average_precision_score


class APscore(BaseMetric):
    """
    Calculate average precision for classification tasks, this method is
    restricted to the binary classification task or multilabel-indicator
    classification task

    Args:
        num_classes (int): number of classes. For num_classes=n, default
                           categorical indexes are 0 ~ n-1

        mode (str): 'binary', 'multilabel-indicator'

    Examples:
        >>> for epoch in range(0, max_epoch):
        ...     ap_score = APscore(num_classes=n, mode=mode)
        ...     for batch in dataloader:
        ...         pred = model(...)
        ...         # calculate ap score of current batch
        ...         # for 'binary' mode, input format should be as follows:
        ...         #   pred = np.array([[0.8, 0.2], [0.3, 0.7], ...]),
        ...         #   target = np.array([0, 1, ...])
        ...         # while for  'multilabel-indicator' mode, it should be like:
        ...         #   pred = np.array([[0.7, 0.8, 0.5, 0.2, 0.05], ...]),
        ...         #       or np.array([[1, 1, 1, 0, 0], ...]),
        ...         #   target = np.array([[1, 1, 0, 0, 0], ...])
        ...         ap_score_batch = ap_score(pred, target)
        ...         # accumulate ap score from the start of current epoch till current batch
        ...         ap_score_tillnow = ap_score.accumulate()
        ...     # calculate ap score of the epoch
        ...     ap_score_epoch = ap_score.accumulate()
    """
    def __init__(self, num_classes=2, mode='binary'):
        self.num_classes = num_classes
        self.mode = mode
        super().__init__()

    def reset(self):
        self.y_true = None
        self.probas_pred = None
        self.num_examples = 0

    def compute(self, pred, target):
        self._num_examples_temp = target.shape[0]
        self._check_pred_range(pred)
        self._y_true_temp = target
        self._probas_pred_temp = pred
        return None

    def update(self, n=1):
        self.num_examples += self._num_examples_temp

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

        if self.mode == 'binary':
            for cat_id in range(self.num_classes):
                ap_score = average_precision_score(self.y_true,
                                                   self.probas_pred[:, cat_id],
                                                   pos_label=cat_id)
                self.accumulate_state[str(cat_id)] = ap_score

        elif self.mode == 'multilabel-indicator':
            ap_score = average_precision_score(self.y_true,
                                               self.probas_pred,
                                               pos_label=1,
                                               average=None)
            for cat_id in range(self.num_classes):
                self.accumulate_state[str(cat_id)] = ap_score[cat_id]
        else:
            raise KeyError(f'mode "{self.mode}" do not exist')

        return self.accumulate_state


class mAPscore(APscore):
    """
    Calculate multilabel average precision for binary classification, this
    method is restricted to the multilabel-indicator classification task

    Args:
        num_classes (int): number of classes. For num_classes=n, default
                           categorical indexes are 0 ~ n-1

        average (str): ['micro', 'macro' (default), 'samples', 'weighted']
            'micro':
                Calculate metrics globally by considering each element of the label
                indicator matrix as a label.
            'macro':
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            'weighted':
                Calculate metrics for each label, and find their average, weighted
                by support (the number of true instances for each label).
            'samples':
                Calculate metrics for each instance, and find their average.

            Will be ignored when y_true is binary.

        sample_weight (array): array-like of shape (n_samples,), default=None
            Sample weights.

    Examples:
        >>> for epoch in range(0, max_epoch):
        ...     map_score = mAPscore(num_classes=n)
        ...     for batch in dataloader:
        ...         pred = model(...)
        ...         # calculate map score of current batch
        ...         # inputs should be like:
        ...         #   pred = np.array([[0.7, 0.8, 0.5, 0.2, 0.05], ...]),
        ...         #       or np.array([[1, 1, 1, 0, 0], ...]),
        ...         #   target = np.array([[1, 1, 0, 0, 0], ...])
        ...         map_score_batch = map_score(pred, target)
        ...         # accumulate ap score from the start of current epoch till current batch
        ...         map_score_tillnow = map_score.accumulate()
        ...     # calculate ap score of the epoch
        ...     map_score_epoch = map_score.accumulate()
    """
    def __init__(self, num_classes, average='macro', sample_weight=None):
        super().__init__(num_classes=num_classes)
        self.kwargs = {
            'average': average,
            'sample_weight': sample_weight,
        }

    def accumulate(self):
        self.accumulate_state = {}
        ap_score = average_precision_score(self.y_true,
                                           self.probas_pred,
                                           pos_label=1,
                                           **self.kwargs)
        self.accumulate_state['mAP'] = ap_score

        return self.accumulate_state

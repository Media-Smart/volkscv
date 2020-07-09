import numpy as np

from .base import BaseMetric
from .libs import roc_auc_score


class AUCscore(BaseMetric):
    """
    Calculate auc for classification tasks, this method is restricted to
    the binary, multiclass, or multilabel-indicator classification task

    Args:
        num_classes (int): number of classes. For num_classes=n, default
                           categorical indexes are 0 ~ n-1

        mode (str): 'binary', 'multiclass', 'multilabel-indicator'

        multi_class (str): {'ovo', 'ovr'}, default='ovo'
            Multiclass only. Determines the type of configuration to use.
            'ovo':
                Computes the average AUC of all possible pairwise combinations of
                classes. Insensitive to class imbalance when average == 'macro'.
            'ovr':
                Computes the AUC of each class against the rest. This treats the
                multiclass case in the same way as the multilabel case.
                Sensitive to class imbalance even when average == 'macro',
                because class imbalance affects the composition of each of the
                'rest' groupings.

        average (str):{'micro', 'macro', 'samples', 'weighted'} or None,
                      default='macro'
                      If 'None', the scores for each class are returned.
                      Otherwise, this determines the type of averaging
                      performed on the data:
                      ( Note: multiclass ROC AUC currently only handles the
                      'macro' and 'weighted' averages, can not be None.)
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

            Will be ignored when 'target' is binary.

        sample_weight (array): array-like of shape (n_samples,), default=None
            Sample weights.

    Examples:
        >>> for epoch in range(0, max_epoch):
        ...     auc_score = AUCscore(num_classes=num_classes)
        ...     for batch in dataloader:
        ...         # calculate auc score of current batch
        ...         # for 'binary' mode, input format should be as follows:
        ...         #   pred = np.array([[0.8, 0.2], [0.3, 0.7], ...]),
        ...         #   target = np.array([0, 1, ...])
        ...         # while for 'multiclass' mode, it should be like:
        ...         #   pred = np.array([[0.6, 0.2, 0.05, 0.1, 0.05],
        ...         #                    [0.05, 0.2, 0.6, 0.1, 0.05], ...])
        ...         #          with sum of each sample <= 1
        ...         #   target = np.array([0, 2, ...])
        ...         # while for 'multilabel-indicator' mode, it should be like:
        ...         #   pred = np.array([[0.7, 0.8, 0.5, 0.2, 0.05], ...]),
        ...         #       or np.array([[1, 1, 1, 0, 0], ...]),
        ...         #   target = np.array([[1, 1, 0, 0, 0], ...])
        ...         auc_score_batch = auc_score(pred, target)
        ...     # calculate auc score of the epoch
        ...     auc_score_epoch = auc_score.accumulate()
    """
    def __init__(self, num_classes, mode='binary',
                 multi_class='ovo', average='macro', sample_weight=None):
        self.num_classes = num_classes
        self.mode = mode
        self.kwargs = {
            'multi_class': multi_class,
            'average': average,
            'sample_weight': sample_weight,
        }
        super().__init__()

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

    def accumulate(self):
        accumulate_state = {}

        if self.mode == 'binary':
            for cat_id in range(self.num_classes):
                auc_score = roc_auc_score(self.y_true,
                                          self.probas_pred[:, cat_id])
                accumulate_state[str(cat_id)] = auc_score

        else:
            if self.mode == 'multiclass':
                self._check_pred_sum(self.probas_pred)
            elif self.mode == 'multilabel-indicator':
                pass
            else:
                raise KeyError(f'mode "{self.mode}" do not exist')

            auc_score = roc_auc_score(self.y_true,
                                      self.probas_pred,
                                      **self.kwargs)
            accumulate_state['auc_score'] = auc_score

        return accumulate_state

    def check(self, pred, target):
        super().check(pred, target)
        self._check_pred_range(pred)

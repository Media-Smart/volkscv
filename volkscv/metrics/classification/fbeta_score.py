from .base import BaseScoreCurve
from .libs import fbeta_score


class Fbetascore(BaseScoreCurve):
    """
    Calculate Fbeta score for classification tasks, this method is restricted to
    the binary, multiclass, or multilabel-indicator classification task

    Args:
        num_classes (int): number of classes. For num_classes=n, default
                           categorical indexes are 0 ~ n-1

        mode (str): 'binary', 'multiclass', 'multilabel-indicator'

        beta (float): Determines the weight of recall in the combined score.

        average (str):{'micro', 'macro', 'samples', 'weighted'} or None,
                      default='macro'
                      If 'None', the scores for each class are returned.
                      Otherwise, this determines the type of averaging
                      performed on the data:
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
                 beta=1, average='macro', sample_weight=None):
        self.num_classes = num_classes
        self.mode = mode
        self.beta = beta
        self.kwargs = {
            'average': average,
            'sample_weight': sample_weight,
        }
        super().__init__()

    def compute(self, pred, target):
        self._y_true_temp = target
        if self.mode in ['binary', 'multiclass']:
            self._probas_pred_temp = pred.argsort(axis=1)[:, ::-1][:, 0]
        elif self.mode == 'multilabel-indicator':
            self._probas_pred_temp = pred
        else:
            raise KeyError(f'mode "{self.mode}" do not exist')

        return None

    def accumulate(self):
        accumulate_state = {}

        if self.mode == 'binary':
            for cat_id in range(self.num_classes):
                f_score = fbeta_score(self.y_true, self.probas_pred,
                                      beta=self.beta, pos_label=cat_id,
                                      average='binary')
                accumulate_state[str(cat_id)] = f_score

        else:
            if self.mode not in ['multiclass', 'multilabel-indicator']:
                raise KeyError(f'mode "{self.mode}" do not exist')

            f_score = fbeta_score(self.y_true,
                                  self.probas_pred,
                                  beta=self.beta,
                                  **self.kwargs)
            accumulate_state[f'f{self.beta}_score'] = f_score

        return accumulate_state

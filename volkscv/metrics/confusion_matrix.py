import numpy as np

from .base import BaseMetric


class BaseConfusionMatrix(BaseMetric):
    """
    Calculate confusion matrix for classification

    Args:
        num_classes (int): number of classes.

    Examples:
        >>> for epoch in range(0, max_epoch):
        ...     cfsmtx = BaseConfusionMatrix(num_classes=1000)
        ...     for batch in dataloader:
        ...         ...
        ...         # calculate confusion matrix of current batch
        ...         cfsmtx_current_batch = cfsmtx(pred, target)
        ...     # calculate confusion matrix of the epoch
        ...     cfsmtx_epoch = cfsmtx.accumulate()
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__()

    def reset(self):
        self.cfsmtx = np.zeros((self.num_classes,) * 2)

    def compute(self, pred, target):
        pred_index = np.argmax(pred, axis=1)
        mask = (target >= 0) & (target < self.num_classes)

        self.current_state = np.bincount(self.num_classes*target[mask].astype('int') + pred_index[mask],
                                         minlength=self.num_classes**2
                                         ).reshape(self.num_classes, self.num_classes)
        return self.current_state

    def update(self, n=1):
        self.cfsmtx += self.current_state

    def accumulate(self):
        accumulate_state = {
            'confusion matrix': self.cfsmtx
        }
        return accumulate_state

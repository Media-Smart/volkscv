import numpy as np

from .base import BaseMetric


class ConfusionMatrix(BaseMetric):
    """
    Calculate confusion matrix for classification

    Args:
        num_classes (int): number of classes.

    Examples:
        >>> for epoch in range(0, max_epoch):
        ...     cfsmtx = ConfusionMatrix(num_classes=1000)
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
        self.cfsmtx = np.zeros((self.num_classes, self.num_classes))
        self.num_examples = 0

    def compute(self, pred, target):
        self._num_examples_temp = target.shape[0]

        pred_index = np.argmax(pred, axis=1)
        self.current_state = np.bincount(target*self.num_classes + pred_index,
                                         minlength=self.num_classes**2
                                         ).reshape(self.num_classes, self.num_classes)
        return self.current_state

    def update(self, n=1):
        self.num_examples += self._num_examples_temp * n
        self.cfsmtx += self.current_state

    def accumulate(self):
        self.accumulate_state = self.cfsmtx
        return self.accumulate_state


class CMAccuracy(ConfusionMatrix):
    """
    Calculate accuracy based on confusion matrix for classification

    Args:
        num_classes (int): number of classes.

    Examples:
        >>> for epoch in range(0, max_epoch):
        ...     cmacc = CMAccuracy(num_classes=1000)
        ...     for batch in dataloader:
        ...         ...
        ...         # calculate confusion matrix based accuracy of current batch
        ...         cmacc_current_batch = cmacc(pred, target)
        ...     # calculate confusion matrix based accuracy of the epoch
        ...     cmacc_epoch = cmacc.accumulate()
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__(num_classes=self.num_classes)

    def accumulate(self):
        cmAccuracy = self.cfsmtx.diagonal().sum() / (self.cfsmtx.sum() + 1e-15)

        self.accumulate_state = {
            'cmAccuracy': cmAccuracy
        }
        return self.accumulate_state


class CMPrecisionRecall(ConfusionMatrix):
    """
    Calculate precision, recall for each classes based on confusion matrix for classification

    Args:
        num_classes (int): number of classes.

    Examples:
        >>> for epoch in range(0, max_epoch):
        ...     cmpr = CMPrecisionRecall(num_classes=1000)
        ...     for batch in dataloader:
        ...         ...
        ...         # calculate confusion matrix based precision, recall of current batch
        ...         cmpr_current_batch = cmpr(pred, target)
        ...     # calculate confusion matrix based precision, recall of the epoch
        ...     cmpr_epoch = cmpr.accumulate()
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__(num_classes=self.num_classes)

    def accumulate(self):
        def operator(dim):
            return self.cfsmtx.diagonal() / (self.cfsmtx.sum(axis=dim) + 1e-15)

        self.accumulate_state = {
            'cmPrecision': operator(0),
            'cmRecall': operator(1),
        }
        return self.accumulate_state

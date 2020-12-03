import numpy as np

from ..confusion_matrix import BaseConfusionMatrix


class ConfusionMatrix(BaseConfusionMatrix):
    """
    Calculate confusion matrix for segmentation

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
    @staticmethod
    def _check_match(pred, target):
        assert (pred.shape[0] == target.shape[0]) and \
               (pred.shape[2:4] == target.shape[1:3]), \
            "pred and target don't match"


class Accuracy(ConfusionMatrix):
    """
    Calculate accuracy based on confusion matrix for segmentation

    Args:
        num_classes (int): number of classes.
        average (str): {'pixel', 'class'}
            'pixel':
                calculate pixel wise average accuracy
            'class':
                calculate class wise average accuracy

    Examples:
        >>> for epoch in range(0, max_epoch):
        ...     cmacc = Accuracy(num_classes=10, average='pixel')
        ...     for batch in dataloader:
        ...         ...
        ...         # calculate confusion matrix based accuracy of current batch
        ...         cmacc_current_batch = cmacc(pred, target)
        ...     # calculate confusion matrix based accuracy of the epoch
        ...     cmacc_epoch = cmacc.accumulate()
    """
    def __init__(self, num_classes, average='pixel'):
        self.num_classes = num_classes
        self.average = average
        super().__init__(num_classes=self.num_classes)

    def accumulate(self):

        assert self.average in ('pixel', 'class'), \
            'Accuracy only support "pixel" & "class" wise average'

        if self.average == 'pixel':
            accuracy = self.cfsmtx.diagonal().sum() / (self.cfsmtx.sum() + 1e-15)

        elif self.average == 'class':
            accuracy_class = self.cfsmtx.diagonal() / self.cfsmtx.sum(axis=1)
            accuracy = np.nanmean(accuracy_class)

        accumulate_state = {
            'accuracy': accuracy
        }
        return accumulate_state


class IoU(ConfusionMatrix):
    """
    Calculate IoU for each class based on confusion matrix for segmentation

    Args:
        num_classes (int): number of classes.

    Examples:
        >>> for epoch in range(0, max_epoch):
        ...     iou = IoU(num_classes=10)
        ...     for batch in dataloader:
        ...         ...
        ...         # calculate confusion matrix based IoU of current batch
        ...         cmacc_current_batch = iou(pred, target)
        ...     # calculate confusion matrix based IoU of the epoch
        ...     cmacc_epoch = iou.accumulate()
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__(num_classes=self.num_classes)

    def accumulate(self):
        ious = self.cfsmtx.diagonal() / (self.cfsmtx.sum(axis=0) + self.cfsmtx.sum(axis=1) -
                                         self.cfsmtx.diagonal() + np.finfo(np.float32).eps)
        accumulate_state = {
            'IoUs': ious
        }
        return accumulate_state


class mIoU(IoU):
    """
    Calculate mIoU based on confusion matrix for segmentation

    Args:
        num_classes (int): number of classes.
        average (str): {'equal', 'frequency_weighted'}
            'equal':
                calculate mIoU in an equal class wise average manner
            'frequency_weighted':
                calculate mIoU in an frequency weighted class wise average manner

    Examples:
        >>> for epoch in range(0, max_epoch):
        ...     miou = mIoU(num_classes=10)
        ...     for batch in dataloader:
        ...         ...
        ...         # calculate confusion matrix based mIoU of current batch
        ...         cmacc_current_batch = miou(pred, target)
        ...     # calculate confusion matrix based mIoU of the epoch
        ...     cmacc_epoch = miou.accumulate()
    """
    def __init__(self, num_classes, average='equal'):
        self.num_classes = num_classes
        self.average = average
        super().__init__(num_classes=self.num_classes)

    def accumulate(self):
        assert self.average in ('equal', 'frequency_weighted'), \
            'mIoU only support "equal" & "frequency_weighted" average'

        ious = (super().accumulate())['IoUs']

        if self.average == 'equal':
            miou = np.nanmean(ious)
        elif self.average == 'frequency_weighted':
            pos_freq = self.cfsmtx.sum(axis=1) / self.cfsmtx.sum()
            miou = (pos_freq[pos_freq > 0]*ious[pos_freq > 0]).sum()

        accumulate_state = {
            'mIoU': miou
        }
        return accumulate_state


class DiceScore(ConfusionMatrix):
    """
    Calculate dice score based on confusion matrix for segmentation

    Args:
        num_classes (int): number of classes.

    Examples:
        >>> for epoch in range(0, max_epoch):
        ...     dicescore = DiceScore(num_classes=10)
        ...     for batch in dataloader:
        ...         ...
        ...         # calculate confusion matrix based dice score of current batch
        ...         cmacc_current_batch = dicescore(pred, target)
        ...     # calculate confusion matrix based dice score of the epoch
        ...     cmacc_epoch = dicescore.accumulate()
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__(self.num_classes)

    def accumulate(self):
        dice_score = 2.0*self.cfsmtx.diagonal() / (self.cfsmtx.sum(axis=1) +
                                                   self.cfsmtx.sum(axis=0) +
                                                   np.finfo(np.float32).eps)

        accumulate_state = {
            'dice_score': dice_score
        }
        return accumulate_state

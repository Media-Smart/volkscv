from .. import BaseConfusionMatrix


ConfusionMatrix = BaseConfusionMatrix


# class ConfusionMatrix(BaseConfusionMatrix):
#     def __init__(self, num_classes):
#         super().__init__(num_classes)


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

        accumulate_state = {
            'cmAccuracy': cmAccuracy
        }
        return accumulate_state


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

        accumulate_state = {
            'cmPrecision': operator(0),
            'cmRecall': operator(1),
        }
        return accumulate_state

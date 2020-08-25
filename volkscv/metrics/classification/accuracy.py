import numpy as np

from .base import BaseClsMetric


class TopKAccuracy(BaseClsMetric):
    """
    Calculate top k accuracy for classification

    Args:
        topk (tuple): indicate the accuracy of top ks needed to be calculated,
            eg, topk = (1, 3, 5), top1, top3, top5 accuracy will be calculated

    Examples:
        >>> for epoch in range(0, max_epoch):
        ...     topk_acc = TopKAccuracy(topk=(1, 3, 5))
        ...     for batch in dataloader:
        ...         pred = model(...)
        ...         acc_current_batch = topk_acc(pred, target)
        ...         # accumulate average acc from the start of current epoch till current batch
        ...         acc_current_average = topk_acc.accumulate()
        ...     # calculate acc of the epoch
        ...     acc_epoch = topk_acc.accumulate()
    """
    def __init__(self, topk=(1,)):
        self.topk = topk
        super(TopKAccuracy, self).__init__()

    def reset(self):
        self.sum = np.zeros(len(self.topk))
        self.count = 0

    def compute(self, pred, target):
        maxk = max(self.topk)
        assert maxk <= pred.shape[1], \
            "max k must be no bigger than the number of categories"

        batch_size = target.shape[0]
        pred_topk_index = pred.argsort(axis=1)[:, ::-1][:, 0:maxk].T
        correct = pred_topk_index == np.tile(target, (maxk, 1))
        self.current_state = []
        for k in self.topk:
            self.current_state.append(correct[:k].sum() / batch_size)

        return self.current_state

    def update(self, n=1):
        self.count += n
        self.sum += self.current_state * n

    def accumulate(self):
        acc = list(self.sum / (self.count + 1e-15))
        accumulate_state = {}
        for i, k in enumerate(self.topk):
            accumulate_state['top_' + str(k)] = acc[i]
        return accumulate_state

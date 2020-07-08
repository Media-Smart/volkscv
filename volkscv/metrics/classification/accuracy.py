import torch

import numpy as np

from .base import BaseMetric


class TopKAccuracybyTensor(BaseMetric):
    """
    compute top N accuracy for classification
    """
    def __init__(self, topk=(1,)):
        self.topk = topk
        super(TopKAccuracybyTensor, self).__init__()

    def reset(self):
        self.current_state = []
        self.accumulate_state = []
        self.sum = np.zeros(len(self.topk))
        self.count = 0

    def compute(self, pred, target):
        assert torch.is_tensor(pred) and torch.is_tensor(target), \
            "Only tensor is supported for computing accuracy"
        assert pred.size(0) == target.size(0), \
            "pred and target don't match"

        with torch.no_grad():
            maxk = max(self.topk)
            assert maxk <= pred.size(1), \
                "max k must be no bigger than the number of categories"

            batch_size = target.size(0)
            _, pred_topk_index = pred.topk(maxk, dim=1, largest=True, sorted=True)
            pred_topk_index = pred_topk_index.t()
            correct = pred_topk_index.eq(target.view(1, -1).expand_as(pred_topk_index))
            self.current_state = []
            for k in self.topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                self.current_state.append(float(correct_k.mul_(100.0 / batch_size)))
        return self.current_state

    def update(self, n=1):
        self.count += n
        self.sum += np.array(self.current_state)*n

    def accumulate(self):
        self.accumulate_state = list(self.sum / (self.count + 1e-15))
        return self.accumulate_state


class TopKAccuracy(BaseMetric):
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
        self.accumulate_state = {}
        for i, k in enumerate(self.topk):
            self.accumulate_state['top_' + str(k) + '_Accuracy'] = acc[i]
        return self.accumulate_state
